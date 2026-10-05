import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from ultimate.__main__ import main
from ultimate.agent import Agent
from ultimate.jev import (DEFAULT_JEV, QUESTIONS, Assessment, JevError, JevRouter,
                          NoRedirect, apply_assessment, evaluate_jev, parse_assessment)
from ultimate.policy import route
from ultimate.provider import Budget, BudgetExceeded
from ultimate.safety import Workspace
from test_ultimate import FakeProvider, answer


def response(choice='deep', confidence=.9, risk=.05, ambiguity=.05):
    probabilities = {t: .025 for t in ('fast', 'balanced', 'deep')}
    probabilities[choice] = .95
    return {'model': 'jev-1.13.0', 'answers': {
        'complexity': {'type': 'choice', 'choice': choice, 'probabilities': probabilities, 'confidence': confidence},
        'consequential': {'type': 'noul', 'noul': risk},
        'missing_requirements': {'type': 'noul', 'noul': ambiguity}},
        'usage': {'input_tokens': 100, 'output_tokens': 50}}

class Opener:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.result).encode())

class JevTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.budget = Budget(Path(self.tmp.name) / 'budget.sqlite')
    def router(self, result=None, error=None, config=None):
        opener = Opener(response() if result is None else result, error)
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-only-key'}):
            router = JevRouter(config or {}, self.budget, opener)
        return router, opener
    def test_documented_payload_and_usage(self):
        router, opener = self.router()
        assessment = router.assess([{'original_request': 'Investigate a bug'}])
        request, timeout = opener.calls[0]
        self.assertEqual(request.full_url, 'https://api.typesafe.ai/v1/systemone')
        body = json.loads(request.data)
        self.assertEqual(set(body), {'model', 'state', 'questions'})
        self.assertEqual(body['model'], 'jev-1.13.0')
        self.assertEqual(body['questions'], QUESTIONS)
        self.assertEqual(assessment.choice, 'deep')
        self.assertAlmostEqual(self.budget.spent, 100 * .042 / 1e6)
        self.assertEqual(timeout, 15)
    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'TYPESAFE_API_KEY'):
                JevRouter({}, self.budget)
    def test_zero_budget_prevents_request(self):
        self.budget.task_limit = .00000001
        router, opener = self.router()
        with self.assertRaises(BudgetExceeded): router.assess([])
        self.assertEqual(opener.calls, [])
    def test_network_failure_keeps_reservation_no_retry(self):
        router, opener = self.router(error=urllib.error.URLError('secret-provider-details'))
        with self.assertRaisesRegex(JevError, '^transport_or_json_error$'): router.assess([])
        self.assertEqual(len(opener.calls), 1)
        self.assertGreater(self.budget.spent, 0)
    def test_http_error_does_not_expose_body(self):
        router, opener = self.router(error=urllib.error.HTTPError('url', 429, 'secret-provider-details', {}, None))
        with self.assertRaisesRegex(JevError, '^http_429$'): router.assess([])
    def test_context_limit(self):
        router, opener = self.router()
        with self.assertRaisesRegex(JevError, 'context_limit'): router.assess(['x' * 30000])
        self.assertEqual(opener.calls, [])
    def test_secrets_block_network(self):
        router, opener = self.router()
        with self.assertRaises(ValueError): router.assess(['api_key=abcdefghijklmnop'])
        self.assertEqual(opener.calls, [])
    def test_version_mismatch(self):
        payload = response()
        payload['model'] = 'jev-9.0'
        router, _ = self.router(payload)
        with self.assertRaisesRegex(JevError, 'model_version_mismatch'): router.assess([])
    def test_alias_records_actual_version(self):
        router, _ = self.router(config={'model': 'jev-latest'})
        self.assertEqual(router.assess([]).model, 'jev-1.13.0')
    def test_invalid_usage_retains_reservation(self):
        payload = response()
        payload['usage']['input_tokens'] = True
        router, _ = self.router(payload)
        with self.assertRaisesRegex(JevError, 'invalid_usage'): router.assess([])
        self.assertGreater(self.budget.spent, 0)
    def test_overspend_stops(self):
        payload = response()
        payload['usage']['input_tokens'] = 1000000
        router, _ = self.router(payload)
        with self.assertRaises(BudgetExceeded): router.assess([])
        self.assertAlmostEqual(self.budget.spent, .042)
    def test_redirects_blocked(self):
        with self.assertRaisesRegex(JevError, 'redirect_blocked'):
            NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example')
    def test_threshold_config_validation(self):
        for value in (float('nan'), True, -1, 2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.router(config={'min_confidence': value})

class JevPolicyTests(unittest.TestCase):
    def test_upgrade_and_preserve_floor(self):
        d, ask = apply_assessment(route('Fix typo'), parse_assessment(response()), DEFAULT_JEV)
        self.assertEqual(d.tier, 'deep')
        self.assertFalse(ask)
        d, _ = apply_assessment(route('Fix authentication'), parse_assessment(response('fast')), DEFAULT_JEV)
        self.assertEqual(d.tier, 'deep')
    def test_confidence_is_not_selected_probability(self):
        payload = response('fast', confidence=.6)
        d, _ = apply_assessment(route('Fix typo'), parse_assessment(payload), DEFAULT_JEV)
        self.assertEqual(d.tier, 'balanced')
        payload['answers']['complexity'].update(confidence=.99, probabilities={'fast': .6, 'balanced': .3, 'deep': .1})
        d, _ = apply_assessment(route('Fix typo'), parse_assessment(payload), DEFAULT_JEV)
        self.assertEqual(d.tier, 'balanced')
    def test_risk_floor(self):
        d, _ = apply_assessment(route('Fix typo'), parse_assessment(response('fast', risk=.6)), DEFAULT_JEV)
        self.assertEqual((d.tier, d.risk), ('deep', 'high'))
    def test_missing_requirements(self):
        _, ask = apply_assessment(route('Implement behavior'), parse_assessment(response(ambiguity=.8)), DEFAULT_JEV)
        self.assertTrue(ask)
    def test_schema_rejections(self):
        mutations = [
            lambda r: r['answers']['complexity'].update(choice='unknown'),
            lambda r: r['answers']['complexity'].update(choice='fast'),
            lambda r: r['answers']['complexity'].update(confidence=float('nan')),
            lambda r: r['answers']['complexity'].update(confidence=True),
            lambda r: r['answers']['complexity'].update(probabilities={'fast': .5}),
            lambda r: r['answers']['complexity'].update(probabilities={'fast': .5, 'balanced': .5, 'deep': .5}),
            lambda r: r['answers']['consequential'].update(type='score'),
            lambda r: r['answers']['consequential'].update(noul=2),
            lambda r: r['answers'].pop('missing_requirements'),
        ]
        for mutate in mutations:
            payload = response()
            mutate(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(JevError): parse_assessment(payload)
        for payload in ([], None, {}, {'model': 'jev-1.13.0', 'answers': []}):
            with self.assertRaises(JevError): parse_assessment(payload)

class StubRouter:
    config = DEFAULT_JEV
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload or response(), error, []
    def assess(self, record):
        self.calls.append(copy.deepcopy(record))
        if self.error: raise self.error
        return parse_assessment(self.payload)

class JevAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
    def test_routes_real_execution_and_receives_context(self):
        (self.ws.root / 'a.py').write_text('x=1')
        router = StubRouter()
        provider = FakeProvider([answer()])
        result = Agent(provider, self.ws, route('Fix typo')).run('Fix typo', ['a.py'], jev_router=router)
        self.assertEqual(provider.tiers, ['deep'])
        self.assertEqual(router.calls[0][1]['initial_file']['content'], 'x=1')
        self.assertEqual(result['routing']['backend'], 'jev')
        self.assertFalse(self.ws.writable)
    def test_ambiguity_prevents_coding(self):
        provider = FakeProvider([])
        result = Agent(provider, self.ws, route('Implement it')).run('Implement it', jev_router=StubRouter(response(ambiguity=.9)))
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(provider.tiers, [])
    def test_failure_stops_by_default(self):
        provider = FakeProvider([])
        with self.assertRaises(JevError):
            Agent(provider, self.ws, route('Fix typo')).run('Fix typo', jev_router=StubRouter(error=JevError('http_429')))
        self.assertEqual(provider.tiers, [])
    def test_explicit_fallback_keeps_balanced_minimum(self):
        provider = FakeProvider([answer()])
        result = Agent(provider, self.ws, route('Fix typo')).run('Fix typo', jev_router=StubRouter(error=JevError('http_429')), router_fallback='rules')
        self.assertEqual(provider.tiers, ['balanced'])
        self.assertEqual(result['routing']['status'], 'fallback')
    def test_budget_cannot_be_bypassed_with_fallback(self):
        with self.assertRaises(BudgetExceeded):
            evaluate_jev(route('Fix typo'), [], StubRouter(error=BudgetExceeded('limit')), fallback='rules')
    def test_lock_skips_jev(self):
        router = StubRouter()
        result = Agent(FakeProvider([answer()]), self.ws, route('Fix typo'), locked=True).run('Fix typo', jev_router=router)
        self.assertEqual(router.calls, [])
        self.assertEqual(result['tier'], 'fast')
    def test_routers_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            Agent(FakeProvider([]), self.ws, route('Fix typo')).run('Fix typo', use_judge=True, jev_router=StubRouter())

class JevCLITests(unittest.TestCase):
    def test_jev_preview_needs_no_openai_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text('{"jev": {}}')
            argv = ['ultimate', 'route', 'Fix typo', '--router', 'jev', '--config', str(config), '--workspace', tmp, '--state-dir', str(Path(tmp) / 'state')]
            with patch('sys.argv', argv), patch('ultimate.__main__.JevRouter', return_value=StubRouter()), patch('ultimate.__main__.OpenAIProvider') as coding, contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 0)
                coding.assert_not_called()
                self.assertEqual(json.loads(output.getvalue())['routing']['backend'], 'jev')
    def test_locked_run_skips_jev_and_reports_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text('{}')
            argv = ['ultimate', 'run', 'Fix typo', '--router', 'jev', '--lock', 'fast',
                    '--config', str(config), '--workspace', tmp, '--state-dir', str(Path(tmp) / 'state')]
            with patch('sys.argv', argv), patch('ultimate.__main__.JevRouter') as jev, patch('ultimate.__main__.OpenAIProvider', return_value=FakeProvider([answer()])), contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 0)
                jev.assert_not_called()
                self.assertEqual(json.loads(output.getvalue())['routing'], {'backend': 'jev', 'status': 'skipped_model_lock'})

    def test_rules_preview_needs_no_config_or_keys(self):
        with patch('sys.argv', ['ultimate', 'route', 'Fix typo']), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
            self.assertEqual(json.loads(output.getvalue())['tier'], 'fast')
    def test_init_includes_jev(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            with patch('sys.argv', ['ultimate', 'init', '--config', str(config)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(config.read_text())['jev']['model'], 'jev-1.13.0')

if __name__ == '__main__':
    unittest.main()
