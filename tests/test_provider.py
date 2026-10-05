"""OpenAI adapter tests (payload, budget, errors) and end-to-end agent workflows on real files."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from ultimate.provider import Budget, OpenAIProvider
from ultimate.agent import Agent
from ultimate.policy import route
from ultimate.safety import Workspace
from test_ultimate import FakeProvider, call, answer

class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = {'models': {tier: {'id': 'configured-test-model', 'input_usd_per_million': 1,
                        'output_usd_per_million': 2, 'input_token_limit': 64000,
                        'reasoning_effort': 'low'} for tier in ('fast', 'balanced', 'deep')}}
        self.budget = Budget(Path(self.tmp.name) / 'budget.sqlite')
    def provider(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'}):
            return OpenAIProvider(self.config, self.budget)
    def test_payload_and_settlement(self):
        result = {'output': [], 'usage': {'input_tokens': 100, 'output_tokens': 50}, 'status': 'completed'}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as urlopen:
            self.provider().call('fast', 'Instructions', [{'original_request': 'Hello'}])
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/responses')
        body = json.loads(request.data)
        self.assertFalse(body['store'])
        self.assertEqual(body['model'], 'configured-test-model')
        self.assertAlmostEqual(self.budget.spent, .0002)
    def test_no_key(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError): OpenAIProvider(self.config, self.budget)
    def test_invalid_rates(self):
        self.config['models']['fast']['input_usd_per_million'] = None
        with self.assertRaises(ValueError): self.provider()
    def test_budget_blocks_network(self):
        self.budget.task_limit = .000001
        with patch('urllib.request.urlopen') as request:
            with self.assertRaises(ValueError): self.provider().call('fast', 'Instructions', [])
            request.assert_not_called()
    def test_failure_not_retried_and_reservation_retained(self):
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('offline')) as request:
            with self.assertRaises(ValueError): self.provider().call('fast', 'Instructions', [])
            self.assertEqual(request.call_count, 1)
        self.assertGreater(self.budget.spent, 0)
    def test_context_blocks_network(self):
        self.config['models']['fast']['input_token_limit'] = 1
        with patch('urllib.request.urlopen') as request:
            with self.assertRaises(ValueError): self.provider().call('fast', 'Instructions', [])
            request.assert_not_called()

class IntegrationTests(unittest.TestCase):
    def test_readonly_disallows_model_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FakeProvider([call('edit_file', path='a.py', content='x=1', expected_sha256='NEW'), answer()])
            a = Agent(p, Workspace(tmp), route('Implement a feature'))
            a.run('Implement a feature')
            self.assertFalse((Path(tmp) / 'a.py').exists())
            self.assertIn('error', a.history[-1]['observation'])
    def test_recovery_persisted_before_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.py').write_text('old')
            ws = Workspace(root, True)
            ws.recovery_dir = root / '.ultimate'
            ws.recovery_dir.mkdir()
            ws.edit('a.py', 'new', ws.read('a.py')['sha256'])
            manifest = json.loads((ws.recovery_dir / 'recovery.json').read_text())
            self.assertEqual((ws.recovery_dir / 'originals' / manifest['a.py']).read_text(), 'old')
    def test_environment_failure_does_not_escalate(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp, True, [sys.executable, '-c', 'import ultimate_missing_dependency'])
            p = FakeProvider([call('run_check')])
            result = Agent(p, ws, route('Fix a bug')).run('Fix a bug')
            self.assertEqual(result['status'], 'environment_blocked')
            self.assertFalse(result['escalated'])
    def test_complete_edit_check_workflow(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sum.py').write_text('def add(a, b): return a - b\n')
            ws = Workspace(root, True, [sys.executable, '-c', 'from sum import add; assert add(2, 3) == 5'])
            p = FakeProvider([call('read_file', path='sum.py'),
                 call('edit_file', path='sum.py', content='def add(a, b): return a + b\n', expected_sha256=ws.read('sum.py')['sha256']),
                 call('run_check'), answer('Corrected addition.')])
            result = Agent(p, ws, route('Fix addition')).run('Fix addition')
            self.assertEqual(result['status'], 'checks_passed')
            self.assertEqual(result['changed_files'], ['sum.py'])
