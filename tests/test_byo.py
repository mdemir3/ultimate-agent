"""Bring-your-own-model tests: compatible APIs, custom adapters, the fit test and per-tier providers."""
import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from ultimate.__main__ import main
from ultimate.agent import TOOLS, Agent
from ultimate.compatible import CompatibleProvider, CustomProvider, NoRedirect
from ultimate.fit import TASKS, accepted, format_report, run_task, select_tasks, summarize, write_project
from ultimate.policy import TIERS, route
from ultimate.provider import Budget, BudgetExceeded
from ultimate.safety import Workspace
from ultimate.ollama import OllamaProvider
from test_ultimate import FakeProvider, answer, call


def completion(content=None, tool_calls=(), finish='stop', usage=(100, 20)):
    """A chat-completions reply with text or tool calls, plus token usage."""
    message = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = [{'id': 'c%d' % i, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
                                 for i, (name, args) in enumerate(tool_calls)]
    return {'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1]}}

class Api:
    """Fake OpenAI-compatible server."""
    def __init__(self, *replies, error=None):
        self.replies, self.error, self.requests = list(replies), error, []
    def open(self, request, timeout):
        self.requests.append(request)
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.replies.pop(0)).encode())
    def body(self, index=0):
        return json.loads(self.requests[index].data)

MODELS = {'fast': 'small', 'balanced': 'medium', 'deep': 'large'}
PRICES = {model: {'input_usd_per_million': 1, 'output_usd_per_million': 2} for model in MODELS.values()}

class CompatibleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.budget = Budget(Path(self.tmp.name) / 'budget.sqlite')
    def provider(self, api, **config):
        config = {'base_url': 'https://api.example.com/v1', 'api_key_env': 'TEST_MODEL_KEY', 'models': MODELS,
                  'prices': PRICES, **config}
        with patch.dict(os.environ, {'TEST_MODEL_KEY': 'test-only'}):
            return CompatibleProvider(config, self.budget, opener=api)
    def test_payload_headers_and_native_turns(self):
        api = Api(completion('Done'))
        transcript = [{'original_request': 'Fix a.py'}, {'action': 'read_file', 'arguments': {}, 'observation': {'path': 'a.py'}},
                      {'controller': 'Verification failed.'}]
        self.provider(api).call('balanced', 'Rules', transcript, tools=TOOLS)
        request, body = api.requests[0], api.body()
        self.assertEqual(request.full_url, 'https://api.example.com/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-only')
        self.assertEqual((body['model'], body['max_tokens']), ('medium', 2048))
        self.assertNotIn('temperature', body)
        roles = [m['role'] for m in body['messages']]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(body['messages'][2]['tool_calls'][0]['id'], body['messages'][3]['tool_call_id'])
        self.assertEqual(body['tools'][0]['function']['name'], 'list_files')
    def test_tool_calls_and_budget_settlement(self):
        result = self.provider(Api(completion(tool_calls=[('read_file', {'path': 'a.py'})]))).call('fast', 'Rules', [], tools=TOOLS)
        self.assertEqual(result['output'], [{'type': 'function_call', 'name': 'read_file', 'arguments': '{"path": "a.py"}'}])
        self.assertAlmostEqual(self.budget.spent, (100 * 1 + 20 * 2) / 1e6)
    def test_text_tool_call_and_truncation(self):
        reply = self.provider(Api(completion('{"name": "run_check", "arguments": {}}'))).call('fast', 'Rules', [], tools=TOOLS)
        self.assertEqual(reply['output'][0]['name'], 'run_check')
        self.assertEqual(self.provider(Api(completion('Partial', finish='length'))).call('fast', 'Rules', [])['status'], 'incomplete')
    def test_schema_uses_response_format(self):
        api = Api(completion('{"tier": "deep"}'))
        self.provider(api).call('fast', 'Classify', [], schema={'type': 'object'})
        self.assertEqual(api.body()['response_format']['json_schema']['schema'], {'type': 'object'})
    def test_remote_needs_prices_https_and_key(self):
        with self.assertRaisesRegex(ValueError, 'compatible.prices'):
            self.provider(Api(), prices={'small': PRICES['small']})
        with self.assertRaisesRegex(ValueError, 'https'):
            self.provider(Api(), base_url='http://api.example.com/v1')
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'MISSING_MODEL_KEY'):
            CompatibleProvider({'base_url': 'https://api.example.com/v1', 'api_key_env': 'MISSING_MODEL_KEY',
                                'models': MODELS, 'prices': PRICES}, self.budget, opener=Api())
        with self.assertRaisesRegex(ValueError, 'models.deep'):
            self.provider(Api(), models={'fast': 'small', 'balanced': 'medium'})
    def test_local_server_needs_no_key_or_price(self):
        api = Api(completion('Done'))
        provider = self.provider(api, base_url='http://localhost:1234/v1', api_key_env='', prices={})
        provider.call('fast', 'Rules', [])
        self.assertIsNone(api.requests[0].get_header('Authorization'))
        self.assertEqual(self.budget.spent, 0)
    def test_free_remote_model_skips_budget(self):
        api = Api(completion('Done'))
        free = {model: {'input_usd_per_million': 0, 'output_usd_per_million': 0} for model in MODELS.values()}
        self.provider(api, prices=free).call('fast', 'Rules', [])
        self.assertEqual(self.budget.spent, 0)
    def test_budget_blocks_before_network(self):
        self.budget.task_limit = 1e-9
        api = Api(completion('unused'))
        with self.assertRaises(BudgetExceeded):
            self.provider(api).call('fast', 'Rules', [])
        self.assertEqual(api.requests, [])
    def test_http_error_detail_without_retry_or_secrets(self):
        def failing(message):
            body = io.BytesIO(json.dumps({'error': {'message': message}}).encode())
            return Api(error=urllib.error.HTTPError('https://api.example.com', 404, 'error', {}, body))
        api = failing('model not found')
        with self.assertRaisesRegex(ValueError, 'model not found.*cost reservation retained'):
            self.provider(api).call('fast', 'Rules', [])
        self.assertEqual(len(api.requests), 1)
        with self.assertRaises(ValueError) as caught:
            self.provider(failing('bad api_key = abcdefghijklmnop')).call('fast', 'Rules', [])
        self.assertNotIn('abcdefghijklmnop', str(caught.exception))
    def test_redirects_are_refused(self):
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, 'Found', {}, 'https://elsewhere.example.com')

class ScriptedAdapter:
    """A custom adapter, loaded by tests as test_byo:ScriptedAdapter."""
    models = {'fast': 'mine-small', 'balanced': 'mine', 'deep': 'mine-large'}
    def __init__(self, options):
        self.replies, self.seen = list(options.get('replies', [])), []
    def complete(self, tier, messages, tools, schema=None):
        self.seen.append((tier, messages, tools))
        return self.replies.pop(0)

class NoComplete:
    """An adapter without complete(), to test the load error."""
    def __init__(self, options):
        pass

class CustomTests(unittest.TestCase):
    def test_adapter_drives_the_agent(self):
        replies = [{'tool_calls': [{'name': 'list_files', 'arguments': {}}]}, {'text': 'Nothing to change.'}]
        provider = CustomProvider({'adapter': 'test_byo:ScriptedAdapter', 'options': {'replies': replies}})
        with tempfile.TemporaryDirectory() as tmp:
            result = Agent(provider, Workspace(tmp), route('Explain the project')).run('Explain the project')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(provider.models['balanced'], 'mine')
        tier, messages, tools = provider.adapter.seen[1]
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'tool'])
        self.assertEqual(tools[0]['type'], 'function')
    def test_load_errors(self):
        for spec, message in (('nope', 'module:ClassName'), ('missing_module_xyz:Model', 'Cannot load'),
                              ('test_byo:NoComplete', 'complete')):
            with self.assertRaisesRegex(ValueError, message):
                CustomProvider({'adapter': spec})
    def test_text_tool_call_counts_as_call(self):
        replies = [{'text': 'Reading first: {"name": "read_file", "arguments": {"path": "a.py"}}'}]
        provider = CustomProvider({'adapter': 'test_byo:ScriptedAdapter', 'options': {'replies': replies}})
        self.assertEqual(provider.call('fast', 'Rules', [], tools=TOOLS)['output'][0]['name'], 'read_file')
    def test_malformed_reply(self):
        provider = CustomProvider({'adapter': 'test_byo:ScriptedAdapter', 'options': {'replies': [{'answer': 'x'}]}})
        with self.assertRaisesRegex(ValueError, 'text'):
            provider.call('fast', 'Rules', [])

class Solver:
    """A scripted model that applies each fit task's reference solution."""
    models = {tier: 'solver' for tier in TIERS}
    def call(self, tier, instructions, transcript, tools=None, schema=None):
        task = next(t for t in TASKS if t['prompt'] == transcript[0]['original_request'])
        written = {e['observation'].get('written') for e in transcript if 'action' in e}
        for entry in transcript[1:]:
            path = entry.get('initial_file', {}).get('path')
            if path and path not in written:
                return call('edit_file', path=path, content=task['solution'][path], expected_sha256=entry['initial_file']['sha256'])
        return answer('Applied the fix.')

class Idle:
    """A scripted model that never edits anything, so every fit task should fail."""
    models = {tier: 'idle' for tier in TIERS}
    def call(self, *args, **kwargs):
        return answer('I looked at it.')

class FitTests(unittest.TestCase):
    def test_tasks_route_to_their_level_and_checks_discriminate(self):
        for task in TASKS:
            self.assertEqual(route(task['prompt'], len(task['files'])).tier, task['level'], task['name'])
            with tempfile.TemporaryDirectory() as start, tempfile.TemporaryDirectory() as solved:
                write_project(Path(start), task['project'])
                write_project(Path(solved), {**task['project'], **task['solution']})
                self.assertFalse(accepted(Path(start), task['accept']), task['name'])
                self.assertTrue(accepted(Path(solved), task['accept']), task['name'])
    def test_scores_come_from_hidden_checks(self):
        for task in TASKS:
            solved, idle = run_task(Solver(), task), run_task(Idle(), task)
            self.assertTrue(solved['passed'], task['name'])
            self.assertEqual(solved['status'], 'checks_passed' if task['check'] else 'unverified')
            self.assertEqual((idle['passed'], idle['status']), (False, 'answered'))
    def test_errors_are_recorded_but_budget_stops(self):
        class Broken:
            def call(self, *args, **kwargs):
                raise ValueError('boom')
        class Broke:
            def call(self, *args, **kwargs):
                raise BudgetExceeded('Budget limit reached; no request was sent.')
        result = run_task(Broken(), TASKS[0])
        self.assertEqual((result['status'], result['passed'], result['model']), ('error: boom', False, None))
        with self.assertRaises(BudgetExceeded):
            run_task(Broke(), TASKS[0])
    def test_summary_and_report(self):
        results = [{'title': t['title'], 'level': t['level'], 'final_tier': t['level'], 'escalated': False, 'model': 'm',
                    'status': 'checks_passed', 'passed': passed, 'seconds': 1.0, 'cost_usd': 0.0}
                   for t, passed in zip(TASKS, (True, True, True, False, False, False))]
        summary = summarize(results)
        self.assertEqual([s['verdict'] for s in summary['tiers'].values()], ['good fit', 'partial fit', 'not a fit yet'])
        self.assertEqual(summary['advice'], 'Use this setup for fast tasks; route balanced and deep tasks to a stronger model.')
        report = format_report(results, summary, 'provider test', 1)
        self.assertIn('By tier:', report)
        self.assertIn('--runs 3', report)
    def test_select_tasks(self):
        self.assertEqual([t['name'] for t in select_tasks(['deep'])], ['login', 'cache'])
        self.assertEqual(len(select_tasks(['typo', 'login'])), 2)
        with self.assertRaisesRegex(ValueError, 'nope'):
            select_tasks(['nope'])

class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.json'
    def main(self, *args, **patches):
        argv = ['ultimate', *args, '--config', str(self.config)]
        if args[0] != 'init':
            argv += ['--state-dir', str(self.root / 'state')]
        with contextlib.ExitStack() as stack:
            mocks = {name: stack.enter_context(patch('ultimate.__main__.' + name, return_value=value))
                     for name, value in patches.items()}
            stack.enter_context(patch('sys.argv', argv))
            out = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            err = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = main()
        return code, out.getvalue(), err.getvalue(), mocks
    def test_fit_report_and_json(self):
        code, out, err, _ = self.main('fit', '--provider', 'ollama', '--tasks', 'typo,bugfix', '--json',
                                      str(self.root / 'fit.json'), OllamaProvider=Solver())
        self.assertEqual(code, 0)
        self.assertIn('This setup handled every tier tested.', out)
        self.assertIn('[fit] 2/2', err)
        report = json.loads((self.root / 'fit.json').read_text())
        self.assertEqual(report['summary']['tiers']['balanced']['passed'], 1)
    def test_fit_model_override(self):
        code, _, _, mocks = self.main('fit', '--provider', 'ollama', '--model', 'coder:7b', '--tasks', 'typo', OllamaProvider=Solver())
        self.assertEqual(code, 0)
        self.assertEqual(mocks['OllamaProvider'].call_args[0][0]['models'], {tier: ['coder:7b'] for tier in TIERS})
        self.config.write_text('{"provider": "openai"}')
        self.assertEqual(self.main('fit', '--model', 'x')[0], 2)
    def test_fit_needs_config_for_remote_providers(self):
        code, _, err, _ = self.main('fit', '--provider', 'compatible')
        self.assertEqual(code, 2)
        self.assertIn('init --provider compatible', err)
    def test_run_with_compatible_provider(self):
        self.config.write_text('{"provider": "compatible"}')
        stub = Idle()
        code, out, _, _ = self.main('run', 'Explain the project', '--workspace', str(self.root), CompatibleProvider=stub)
        result = json.loads(out)
        self.assertEqual((code, result['provider'], result['model']), (0, 'compatible', 'idle'))
    def test_init_writes_every_provider_section(self):
        code, out, _, _ = self.main('init', '--provider', 'compatible')
        written = json.loads(self.config.read_text())
        self.assertEqual((code, written['provider']), (0, 'compatible'))
        self.assertTrue({'models', 'ollama', 'compatible', 'custom'} <= set(written))

class Stub(FakeProvider):
    """A FakeProvider labeled with one model name for every tier."""
    def __init__(self, label, responses=()):
        super().__init__(responses)
        self.models = {tier: label for tier in TIERS}

class TierProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'provider': 'ollama', 'tier_providers': {'deep': 'compatible'}}))
    def main(self, *args, local=None, hosted=None):
        argv = ['ultimate', *args, '--config', str(self.config), '--workspace', str(self.root), '--state-dir', str(self.root / 'state')]
        with patch('sys.argv', argv), patch('ultimate.__main__.OllamaProvider', return_value=local) as ollama, \
                patch('ultimate.__main__.CompatibleProvider', return_value=hosted) as compatible, \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = main()
        self.ollama, self.compatible, self.err = ollama, compatible, err.getvalue()
        return code, json.loads(out.getvalue()) if out.getvalue().startswith('{') else None
    def test_deep_tasks_go_to_their_own_provider(self):
        code, result = self.main('run', 'Find why login occasionally fails', local=Stub('local'), hosted=Stub('hosted', [answer('Found it.')]))
        self.assertEqual((code, result['tier'], result['provider'], result['model']), (0, 'deep', 'compatible', 'hosted'))
        self.assertEqual(result['providers'], {'fast': 'ollama', 'balanced': 'ollama', 'deep': 'compatible'})
        self.assertEqual(self.ollama.call_args[1]['tiers'], ('fast', 'balanced'))
        self.assertEqual(self.compatible.call_args[1]['tiers'], ('deep',))
    def test_provider_flag_uses_one_provider_for_every_tier(self):
        code, result = self.main('run', 'Find why login occasionally fails', '--provider', 'ollama', local=Stub('local', [answer()]))
        self.assertEqual((code, result['provider'], result['model']), (0, 'ollama', 'local'))
        self.assertNotIn('providers', result)
        self.compatible.assert_not_called()
    def test_preview_names_each_tiers_model_without_contacting_it(self):
        self.config.write_text(json.dumps({'provider': 'ollama', 'tier_providers': {'deep': 'compatible'},
                                           'compatible': {'models': {'deep': 'hosted-large'}}}))
        code, result = self.main('route', 'Find why login occasionally fails')
        self.assertEqual((code, result['provider'], result['model']), (0, 'compatible', 'hosted-large'))
        self.compatible.assert_not_called()
        self.ollama.assert_not_called()
    def test_invalid_tier_providers(self):
        for setting in ({'expert': 'ollama'}, {'deep': 'gpt'}, ['deep']):
            self.config.write_text(json.dumps({'provider': 'ollama', 'tier_providers': setting}))
            self.assertEqual(self.main('route', 'Add a search feature')[0], 2, setting)
            self.assertIn('tier_providers', self.err)
    def test_jev_refused_when_any_tier_is_local(self):
        self.assertEqual(self.main('route', 'Add a search feature', '--router', 'jev')[0], 2)
        self.assertIn('TypeSafe', self.err)
    def test_fit_model_needs_one_provider(self):
        argv = ['ultimate', 'fit', '--model', 'x', '--config', str(self.config), '--state-dir', str(self.root / 'state')]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(), 2)
        self.assertIn('--provider', err.getvalue())
    def test_escalation_hands_the_transcript_to_the_deep_provider(self):
        from ultimate.__main__ import TieredProvider
        local, hosted = Stub('local', [call('request_escalation', reason='Needs investigation.')]), Stub('hosted', [answer('Done')])
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(TieredProvider({'fast': local, 'balanced': local, 'deep': hosted}), Workspace(tmp), route('Implement a feature'))
            result = agent.run('Implement a feature')
        self.assertEqual((local.tiers, hosted.tiers, result['escalated']), (['balanced'], ['deep'], True))
    def test_providers_validate_only_their_tiers(self):
        from test_ollama import Server
        provider = CompatibleProvider({'base_url': 'https://api.example.com/v1', 'models': {'deep': 'large'},
                                       'prices': {'large': PRICES['large']}}, Budget(self.root / 'b.sqlite'), opener=Api(), tiers=('deep',))
        self.assertEqual(provider.models, {'deep': 'large'})
        local = OllamaProvider({}, opener=Server(installed=('qwen2.5-coder:14b',), tools=('qwen2.5-coder:14b',)), tiers=('deep',))
        self.assertEqual(local.models, {'deep': 'qwen2.5-coder:14b'})

if __name__ == '__main__':
    unittest.main()
