"""Core tests: routing rules, workspace guardrails, the budget ledger and the agent loop.

FakeProvider stands in for a model; call() and answer() build its scripted replies.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from ultimate.agent import Agent
from ultimate.policy import route, judge_upgrade
from ultimate.provider import Budget
from ultimate.safety import Workspace, ensure_no_secrets


def call(name, **args):
    """A scripted reply in which the model calls one tool."""
    return {'status': 'completed', 'output': [{'type': 'function_call', 'name': name, 'arguments': json.dumps(args)}]}

def answer(text='Done'):
    """A scripted reply in which the model gives its final text answer."""
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}

class FakeProvider:
    """Plays a model by replaying scripted replies in order; records the tier of each step."""
    def __init__(self, responses):
        self.responses, self.tiers = iter(responses), []
    def call(self, tier, *args, **kwargs):
        self.tiers.append(tier)
        return next(self.responses)

class RoutingTests(unittest.TestCase):
    def test_mechanical(self):
        self.assertEqual(route('Fix a typo in the README').tier, 'fast')
    def test_risk_overrides_simple(self):
        self.assertEqual(route('Rename authentication permissions').tier, 'deep')
    def test_unknown_is_balanced(self):
        self.assertEqual(route('Make it work').tier, 'balanced')
    def test_hard_short_prompt(self):
        self.assertEqual(route('Fix deadlock').tier, 'deep')
    def test_judge_cannot_lower_floor(self):
        self.assertEqual(judge_upgrade(route('Fix authentication'), {'tier': 'fast'}).tier, 'deep')
    def test_lock_cannot_lower_floor(self):
        with self.assertRaises(ValueError): route('Fix payment bug', lock='fast')
    def test_empty(self):
        with self.assertRaises(ValueError): route(' ')

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = Workspace(self.root, True)
        (self.root / 'a.py').write_text('x = 1\n')
    def test_path_escape(self):
        for p in ('../a.py', '/tmp/a.py', '.env', '.git/config', 'secret.json'):
            with self.assertRaises(ValueError): self.ws.path(p)
    def test_symlink(self):
        (self.root / 'b.py').symlink_to(self.root / 'a.py')
        with self.assertRaises(ValueError): self.ws.read('b.py')
    def test_hardlink(self):
        os.link(self.root / 'a.py', self.root / 'b.py')
        with self.assertRaises(ValueError): self.ws.read('b.py')
    def test_conflict_and_snapshot(self):
        hashed = self.ws.read('a.py')['sha256']
        with self.assertRaises(ValueError): self.ws.edit('a.py', 'x=2', 'bad')
        self.ws.edit('a.py', 'x=2', hashed)
        self.assertEqual(self.ws.snapshots['a.py'], b'x = 1\n')
    def test_readonly(self):
        with self.assertRaises(ValueError): Workspace(self.root).edit('a.py', 'x=2', 'NEW')
    def test_secret(self):
        with self.assertRaises(ValueError): ensure_no_secrets('api_key = "abcdefghijklmnop"')
    def test_check_and_invalidation(self):
        import sys
        self.ws.check = [sys.executable, '-c', 'print("ok")']
        self.assertTrue(self.ws.verify()['passed'])
        self.ws.edit('a.py', 'x=2', self.ws.read('a.py')['sha256'])
        self.assertIsNone(self.ws.last_check)

class BudgetTests(unittest.TestCase):
    def test_persistent_daily_and_task_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b.sqlite'
            b = Budget(path, 1, 2)
            b.reserve(.8)
            with self.assertRaises(ValueError): b.reserve(.3)
            c = Budget(path, 2, 1)
            with self.assertRaises(ValueError): c.reserve(.3)
    def test_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = Budget(Path(tmp) / 'b.sqlite', 1, 2)
            row = b.reserve(.8)
            b.settle(row, .8, .1)
            b.reserve(.8)
            self.assertAlmostEqual(b.spent, .9)
    def test_invalid_budgets(self):
        for value in (float('nan'), -1, float('inf')):
            with self.assertRaises(ValueError): Budget('unused', value)

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name, True)
    def test_escalation_preserves_state_and_is_bounded(self):
        p = FakeProvider([call('request_escalation', reason='complex'), call('request_escalation', reason='again'), answer()])
        a = Agent(p, self.ws, route('Implement a feature'))
        result = a.run('Implement a feature')
        self.assertEqual(p.tiers, ['balanced', 'deep', 'deep'])
        self.assertTrue(result['escalated'])
        self.assertEqual(a.history[0]['original_request'], 'Implement a feature')
    def test_lock_blocks_escalation(self):
        p = FakeProvider([call('request_escalation', reason='complex'), answer()])
        result = Agent(p, self.ws, route('Fix typo'), locked=True).run('Fix typo')
        self.assertFalse(result['escalated'])
    def test_no_verification_means_unverified(self):
        p = FakeProvider([call('edit_file', path='a.py', content='x=2', expected_sha256='NEW'), answer('Tests passed')])
        self.assertEqual(Agent(p, self.ws, route('Implement a feature')).run('Implement a feature')['status'], 'unverified')
    def test_controller_verification(self):
        import sys
        self.ws.check = [sys.executable, '-c', 'assert True']
        p = FakeProvider([call('edit_file', path='a.py', content='x=2', expected_sha256='NEW'), answer()])
        self.assertEqual(Agent(p, self.ws, route('Implement a feature')).run('Implement a feature')['status'], 'checks_passed')
    def test_two_failures_escalate(self):
        import sys
        self.ws.check = [sys.executable, '-c', 'raise SystemExit(1)']
        p = FakeProvider([call('run_check'), call('run_check'), answer()])
        Agent(p, self.ws, route('Fix a bug')).run('Fix a bug')
        self.assertEqual(p.tiers, ['balanced', 'balanced', 'deep'])
    def test_unknown_tool_blocked(self):
        p = FakeProvider([call('shell', command='touch BAD'), answer()])
        a = Agent(p, self.ws, route('Implement a feature'))
        a.run('Implement a feature')
        self.assertIn('error', a.history[-1]['observation'])
        self.assertFalse((self.ws.root / 'BAD').exists())
    def test_malformed_tool_arguments(self):
        p = FakeProvider([{'output': [{'type': 'function_call', 'name': 'read_file', 'arguments': '{'}]}, answer()])
        a = Agent(p, self.ws, route('Read code'))
        a.run('Read code')
        self.assertIn('error', a.history[-1]['observation'])
    def test_step_limit(self):
        p = FakeProvider([call('list_files')])
        self.assertEqual(Agent(p, self.ws, route('Read code'), max_steps=1).run('Read code')['status'], 'incomplete')

if __name__ == '__main__':
    unittest.main()
