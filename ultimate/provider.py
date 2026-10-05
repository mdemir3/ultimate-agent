"""The OpenAI adapter and the spending budget.

Budget keeps a small SQLite ledger so per-task and per-day limits hold across runs. Before
each paid request it reserves the most that request could cost; afterwards it replaces the
reservation with the actual cost the API reports.
"""
import datetime
import json
import math
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

class BudgetExceeded(ValueError):
    """Raised before sending a request that would go over a spending limit."""

class Budget:
    """Spending limits in US dollars: one per task and one per UTC day, shared by runs with the same state folder."""
    def __init__(self, path, task_limit=2.0, daily_limit=10.0):
        if not all(math.isfinite(x) and x > 0 for x in (task_limit, daily_limit)):
            raise ValueError('Budget limits must be positive finite numbers.')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_limit, self.daily_limit, self.spent = task_limit, daily_limit, 0.0
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, day TEXT, amount REAL)')

    def reserve(self, amount):
        """Record the worst-case cost of a request before sending it; refuse if a limit would be exceeded."""
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError('Invalid cost reservation.')
        day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        with sqlite3.connect(self.path, timeout=10) as db:
            # Lock the ledger so two runs cannot both pass the check and overspend together.
            db.execute('BEGIN IMMEDIATE')
            total = db.execute('SELECT COALESCE(SUM(amount),0) FROM charges WHERE day=?', (day,)).fetchone()[0]
            if self.spent + amount > self.task_limit or total + amount > self.daily_limit:
                raise BudgetExceeded('Budget limit reached; no request was sent.')
            row = db.execute('INSERT INTO charges(day,amount) VALUES (?,?)', (day, amount)).lastrowid
        self.spent += amount
        return row

    def settle(self, row, reservation, actual):
        """Replace a reservation with the actual cost once the API reports token usage."""
        # Never credit more than reserved. Unexpected excess is charged in full.
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE charges SET amount=? WHERE id=?', (actual, row))
        self.spent += actual - reservation

class OpenAIProvider:
    """Calls OpenAI's Responses API, using each tier's model ID and prices from the config."""
    def __init__(self, config, budget, tiers=('fast', 'balanced', 'deep')):
        self.config, self.budget = config, budget
        self.key = os.environ.get('OPENAI_API_KEY')
        if not self.key:
            raise ValueError('Set OPENAI_API_KEY in your terminal; never paste it into chat or config.')
        for tier in tiers:
            model = config.get('models', {}).get(tier, {})
            if not model.get('id'):
                raise ValueError('Configure an API model ID for every tier in your config.')
            for rate in ('input_usd_per_million', 'output_usd_per_million'):
                value = model.get(rate)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Configure current positive token prices for every model.')
        self.models = {tier: config['models'][tier]['id'] for tier in tiers}

    def call(self, tier, instructions, transcript, tools=None, schema=None):
        """Send one agent step to the tier's model and return the raw Responses API result."""
        spec = self.config['models'][tier]
        # The whole task record travels as one JSON user message.
        payload = {'model': spec['id'], 'instructions': instructions,
                   'input': [{'role': 'user', 'content': json.dumps(transcript, ensure_ascii=False)}],
                   'max_output_tokens': 2048, 'store': False}
        if spec.get('reasoning_effort'):
            payload['reasoning'] = {'effort': spec['reasoning_effort']}
        if tools:
            payload['tools'] = tools
            payload['parallel_tool_calls'] = False  # One tool call at a time, as the agent loop requires.
        # The LLM judge passes a JSON schema so the answer is just the tier.
        if schema:
            payload['text'] = {'format': {'type': 'json_schema', 'name': 'assessment',
                                          'strict': True, 'schema': schema}}
        data = json.dumps(payload, ensure_ascii=False).encode()
        # UTF-8 byte count plus margin is intentionally conservative for text requests.
        input_bound = len(data) + 2048
        if input_bound > spec.get('input_token_limit', 64000):
            raise ValueError('Context limit reached; narrow the task or use fewer files.')
        # Reserve the most this request could cost; nothing is sent if that would break the budget.
        reservation = (input_bound * spec['input_usd_per_million'] + 2048 * spec['output_usd_per_million']) / 1e6
        row = self.budget.reserve(reservation)
        request = urllib.request.Request('https://api.openai.com/v1/responses', data=data,
                  headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError('Provider returned HTTP %s. No automatic retry; cost reservation retained.' % exc.code) from None
        except (OSError, ValueError):
            raise ValueError('Provider request failed. No automatic retry; cost reservation retained.') from None
        # Settle to the actual cost when the API reports token usage.
        usage = result.get('usage')
        if usage and all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')):
            actual = (usage['input_tokens'] * spec['input_usd_per_million'] + usage['output_tokens'] * spec['output_usd_per_million']) / 1e6
            self.budget.settle(row, reservation, actual)
        return result

def response_text(result):
    """Join the text parts of a Responses-style result: the model's answer when it called no tool."""
    return '\n'.join(c.get('text', '') for item in result.get('output', [])
                     if item.get('type') == 'message' for c in item.get('content', [])
                     if c.get('type') == 'output_text')
