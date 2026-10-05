"""Fit test: run a model setup through small coding tasks, scored by acceptance checks the model never sees.

Each task builds a tiny project in a temporary folder and lets the agent work on it like a normal
run. Then a hidden acceptance script runs; the task passes only if that script passes, whatever
the agent reported.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .agent import Agent
from .policy import TIERS, route
from .provider import BudgetExceeded
from .safety import Workspace

# Starting files for the tasks. {name}, {token} and {key} are filled in per task.
CALCULATOR = 'def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n'
CALCULATOR_TESTS = '''import unittest
from calculator import add, multiply


class CalculatorTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)
'''
SHOP = '''def {name}(prices):
    """Return the sum of all prices."""
    return sum(prices)


def checkout(prices, discount=0):
    return round({name}(prices) * (1 - discount), 2)
'''
LOGIN = '''import time

SESSIONS = {{}}


def login(user, pw, users):
    """Return a session token for valid credentials, else None."""
    if users.get(user) != pw:
        return None
    token = {token}
    SESSIONS[token] = user
    return token
'''
PRICES = '''_cache = {{}}


def get_price(item, currency, base_prices, rates):
    """Return the price of item in currency, caching results."""
    key = {key}
    if key in _cache:
        return _cache[key]
    price = round(base_prices[item] * rates[currency], 2)
    _cache[key] = price
    return price
'''

# Each task: the level it is written for, a project, an optional visible test command, a hidden
# acceptance script run in the project after the agent stops, and a reference solution for self-tests.
TASKS = [
    {'name': 'typo', 'level': 'fast', 'title': 'Fix a spelling mistake',
     'prompt': 'Fix the spelling mistake in README.md', 'files': ['README.md'], 'check': False,
     'project': {'README.md': '# Calculator\n\nA tiny calculater for adding and multiplying numbers.\n'},
     'accept': "text = open('README.md').read()\nassert 'calculater' not in text\n"
               "assert 'A tiny calculator for adding and multiplying numbers.' in text\nassert text.startswith('# Calculator')\n",
     'solution': {'README.md': '# Calculator\n\nA tiny calculator for adding and multiplying numbers.\n'}},
    {'name': 'rename', 'level': 'fast', 'title': 'Rename a function and its call',
     'prompt': 'Rename calc_total to calculate_total in shop.py, including where it is called', 'files': ['shop.py'], 'check': True,
     'project': {'shop.py': SHOP.format(name='calc_total'), 'tests/test_shop.py':
                 'import unittest\nfrom shop import checkout\n\n\nclass ShopTests(unittest.TestCase):\n'
                 '    def test_checkout(self):\n        self.assertEqual(checkout([10, 5]), 15)\n\n'
                 '    def test_discount(self):\n        self.assertEqual(checkout([10, 10], 0.25), 15)\n'},
     'accept': "import shop\nassert not hasattr(shop, 'calc_total')\nassert shop.calculate_total([1, 2]) == 3\n"
               "assert shop.checkout([10, 10], 0.25) == 15\n",
     'solution': {'shop.py': SHOP.format(name='calculate_total')}},
    {'name': 'bugfix', 'level': 'balanced', 'title': 'Fix a bug covered by tests',
     'prompt': 'Fix the addition bug in calculator.py and make the tests pass', 'files': ['calculator.py'], 'check': True,
     'project': {'calculator.py': CALCULATOR.replace('a + b', 'a - b'), 'tests/test_calculator.py': CALCULATOR_TESTS},
     'accept': 'from calculator import add, multiply\nassert add(2, 3) == 5 and add(-2, 2) == 0\nassert multiply(3, 4) == 12\n',
     'solution': {'calculator.py': CALCULATOR}},
    {'name': 'feature', 'level': 'balanced', 'title': 'Add a small feature',
     'prompt': 'Add a subtract(a, b) function to calculator.py that returns a minus b', 'files': ['calculator.py'], 'check': True,
     'project': {'calculator.py': CALCULATOR, 'tests/test_calculator.py': CALCULATOR_TESTS},
     'accept': 'from calculator import add, subtract\nassert subtract(5, 3) == 2 and subtract(0, 4) == -4\nassert add(2, 3) == 5\n',
     'solution': {'calculator.py': CALCULATOR + '\n\ndef subtract(a, b):\n    return a - b\n'}},
    {'name': 'login', 'level': 'deep', 'title': 'Find an intermittent login bug',
     'prompt': 'Find why login occasionally fails and fix it', 'files': ['login.py'], 'check': True,
     'project': {'login.py': LOGIN.format(token='str(int(time.time()))'), 'tests/test_login.py':
                 "import unittest\nfrom login import SESSIONS, login\n\nUSERS = {'alice': 'pw-a', 'bob': 'pw-b'}\n\n\n"
                 "class LoginTests(unittest.TestCase):\n    def test_wrong_password(self):\n"
                 "        self.assertIsNone(login('alice', 'nope', USERS))\n\n"
                 "    def test_each_login_keeps_its_own_session(self):\n        first = login('alice', 'pw-a', USERS)\n"
                 "        second = login('bob', 'pw-b', USERS)\n        self.assertEqual(SESSIONS[first], 'alice')\n"
                 "        self.assertEqual(SESSIONS[second], 'bob')\n"},
     'accept': "import login\nusers = {'u%d' % i: 'pw%d' % i for i in range(50)}\n"
               "tokens = [login.login(u, p, users) for u, p in users.items()]\n"
               "assert None not in tokens and len(set(tokens)) == 50\n"
               "assert all(login.SESSIONS[t] == u for t, u in zip(tokens, users))\nassert login.login('u1', 'wrong', users) is None\n",
     'solution': {'login.py': LOGIN.format(token='uuid.uuid4().hex').replace('import time', 'import uuid')}},
    {'name': 'cache', 'level': 'deep', 'title': 'Find a stale-cache bug',
     'prompt': 'Find why get_price sometimes returns stale prices and fix it', 'files': ['prices.py'], 'check': True,
     'project': {'prices.py': PRICES.format(key='item'), 'tests/test_prices.py':
                 "import unittest\nfrom prices import get_price\n\nBASE = {'apple': 2.0}\nRATES = {'USD': 1.0, 'EUR': 0.5}\n\n\n"
                 "class PriceTests(unittest.TestCase):\n    def test_eur(self):\n"
                 "        self.assertEqual(get_price('apple', 'EUR', BASE, RATES), 1.0)\n\n"
                 "    def test_usd(self):\n        self.assertEqual(get_price('apple', 'USD', BASE, RATES), 2.0)\n"},
     'accept': "import prices\nbase, rates = {'apple': 2.0, 'pear': 3.0}, {'USD': 1.0, 'EUR': 0.5}\n"
               "assert prices.get_price('apple', 'EUR', base, rates) == 1.0\n"
               "assert prices.get_price('apple', 'USD', base, rates) == 2.0\n"
               "assert prices.get_price('pear', 'EUR', base, rates) == 1.5\n",
     'solution': {'prices.py': PRICES.format(key='(item, currency)')}},
]

def select_tasks(names=None):
    """Pick tasks by name or level, e.g. ['deep'] or ['typo', 'login']."""
    if not names:
        return list(TASKS)
    chosen = [t for t in TASKS if t['name'] in names or t['level'] in names]
    unknown = set(names) - {t['name'] for t in TASKS} - set(TIERS)
    if unknown or not chosen:
        raise ValueError('Unknown fit task(s): %s. Use task names (%s) or levels (fast, balanced, deep).'
                         % (', '.join(sorted(unknown)) or 'none', ', '.join(t['name'] for t in TASKS)))
    return chosen

def write_project(root, files):
    """Create the task's starting files under root."""
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content)

def accepted(root, script):
    """Run the hidden acceptance script in the project after the agent stops; the model never sees it."""
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'SYSTEMROOT')}
    try:
        # -c puts the project folder first on the import path.
        return subprocess.run([sys.executable, '-c', script], cwd=root, env=env, capture_output=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        return False

def run_task(provider, task, max_steps=12, budget=None, emit=None):
    """Run one task on a fresh copy of its project and return one result row."""
    with tempfile.TemporaryDirectory(prefix='ultimate-fit-') as tmp:
        root = Path(tmp) / 'project'
        write_project(root, task['project'])
        check = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'] if task['check'] else None
        workspace, events = Workspace(root, True, check), []
        def record(event):
            events.append(event)
            if emit:
                emit({**event, 'task': task['name']})
        decision = route(task['prompt'], len(task['files']))  # The same routing rules as a normal run.
        # Measure spending and time for this task alone.
        spent, started = budget.spent if budget else 0.0, time.monotonic()
        try:
            result = Agent(provider, workspace, decision, max_steps, emit=record).run(task['prompt'], task['files'])
            status, tier, escalated = result['status'], result['tier'], result['escalated']
        except BudgetExceeded:
            raise  # A budget stop ends the whole fit test; the caller reports partial results.
        except Exception as exc:  # One failing model call should not end the whole fit test.
            status, tier, escalated = 'error: %s' % str(exc)[:160], decision.tier, False
        models = getattr(provider, 'models', None)
        return {'task': task['name'], 'title': task['title'], 'level': task['level'], 'routed_tier': decision.tier,
                'final_tier': tier, 'escalated': escalated,
                'model': models.get(tier) if isinstance(models, dict) else None,
                'status': status, 'passed': accepted(root, task['accept']),
                'steps': sum(1 for e in events if e.get('event') == 'tool'),
                'seconds': round(time.monotonic() - started, 1),
                'cost_usd': round((budget.spent if budget else 0.0) - spent, 6)}

def verdict(passed, runs):
    """Turn a pass rate into a label: 80% or more is a good fit, 50% or more a partial fit."""
    rate = passed / runs
    return 'good fit' if rate >= 0.8 else 'partial fit' if rate >= 0.5 else 'not a fit yet'

def summarize(results):
    """Count passes per tier and suggest which tiers this setup can handle."""
    tiers = {}
    for level in TIERS:
        runs = [r for r in results if r['level'] == level]
        if runs:
            passed = sum(r['passed'] for r in runs)
            tiers[level] = {'passed': passed, 'runs': len(runs), 'verdict': verdict(passed, len(runs))}
    good = [level for level, s in tiers.items() if s['verdict'] == 'good fit']
    rest = [level for level in tiers if level not in good]
    if good and not rest:
        advice = 'This setup handled every tier tested.'
    elif good:
        advice = 'Use this setup for %s tasks; route %s tasks to a stronger model.' % (' and '.join(good), ' and '.join(rest))
    else:
        advice = 'This setup is not yet a fit for any tier tested; try a stronger model.'
    return {'tiers': tiers, 'advice': advice, 'runs': len(results),
            'seconds': round(sum(r['seconds'] for r in results), 1),
            'cost_usd': round(sum(r['cost_usd'] for r in results), 6)}

def format_report(results, summary, setup, runs_per_task=1):
    """Format the results as the plain-text scorecard printed at the end."""
    rows = [('%s (%s)' % (r['title'], r['level']), r['final_tier'] + ('*' if r['escalated'] else ''), (r['model'] or '-')[:32],
             'PASS' if r['passed'] else 'FAIL', r['status'][:24], '%.0fs' % r['seconds']) for r in results]
    header = ('Task', 'Tier', 'Model', 'Result', 'Agent status', 'Time')
    # Pad each column to its widest cell.
    widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]
    lines = ['Ultimate fit test — ' + setup, '']
    lines += ['  '.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [header] + rows]
    lines += ['', 'By tier:']
    for level, s in summary['tiers'].items():
        lines.append('  %-9s %d/%d  %s' % (level, s['passed'], s['runs'], s['verdict']))
    lines += ['', summary['advice'],
              '%d runs, %.0fs, $%.4f estimated. * = escalated to deep. Results come from hidden acceptance checks, '
              'not the agent\'s own report.' % (summary['runs'], summary['seconds'], summary['cost_usd'])]
    if runs_per_task == 1:
        lines.append('One run per task is a small sample; use --runs 3 for steadier results.')
    return '\n'.join(lines)
