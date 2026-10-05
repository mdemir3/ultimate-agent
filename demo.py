"""Offline demonstration: scripted model responses, real file edits and checks."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from ultimate.agent import Agent
from ultimate.jev import Assessment, DEFAULT_JEV
from ultimate.policy import needs_clarification, route
from ultimate.safety import Workspace

class DemoProvider:
    def __init__(self, workspace):
        self.workspace = workspace
        self.step = 0
    def call(self, tier, instructions, transcript, **kwargs):
        self.step += 1
        print('  Model tier: ' + tier)
        if self.step == 1:
            name, args = 'read_file', {'path': 'calculator.py'}
        elif self.step == 2:
            name, args = 'request_escalation', {'reason': 'Demonstrate a safe handoff with preserved task history.'}
        elif self.step == 3:
            name, args = 'edit_file', {'path': 'calculator.py', 'content': 'def add(a, b):\n    return a + b\n',
                                      'expected_sha256': self.workspace.read('calculator.py')['sha256']}
        elif self.step == 4:
            name, args = 'run_check', {}
        else:
            return {'status': 'completed', 'output': [{'type': 'message', 'content': [
                    {'type': 'output_text', 'text': 'Fixed addition and passed the authorized check.'}]}]}
        return {'status': 'completed', 'output': [{'type': 'function_call', 'name': name, 'arguments': json.dumps(args)}]}

class DemoRouter:
    config = DEFAULT_JEV

    def assess(self, record):
        return Assessment('jev-1.13.0', 'balanced',
                          {'fast': .025, 'balanced': .95, 'deep': .025}, .9, .05, .05)

def main():
    parser = argparse.ArgumentParser(description='Scripted offline Ultimate demonstration.')
    parser.add_argument('--router', choices=['rules', 'jev'], default='rules')
    args = parser.parse_args()
    print('ULTIMATE MODE — offline demo')
    print('No API calls. Coding responses and Jev assessments are scripted; edits and checks are real.\n')
    for prompt in ('Fix this spelling mistake', 'Add search to this page', 'Find why login occasionally fails', 'Fix it'):
        if needs_clarification(prompt):
            print('%s → ASK (no target named; add details or --file)' % prompt)
            continue
        d = route(prompt)
        print('%s → %s (%s)' % (prompt, d.tier.upper(), d.reasons[0]))
    print('\nDemonstrating read → switch → edit → verify in a temporary project:')
    with tempfile.TemporaryDirectory(prefix='ultimate-demo-') as tmp:
        (Path(tmp) / 'calculator.py').write_text('def add(a, b):\n    return a - b\n')
        ws = Workspace(tmp, True, [sys.executable, '-c', 'from calculator import add; assert add(2, 3) == 5'])
        result = Agent(DemoProvider(ws), ws, route('Fix the addition bug'),
                       emit=lambda event: print('  ' + json.dumps(event))).run(
                           'Fix the addition bug', jev_router=DemoRouter() if args.router == 'jev' else None)
        print('\n' + json.dumps(result, indent=2))
        return 0 if result['status'] == 'checks_passed' else 1

if __name__ == '__main__':
    sys.exit(main())
