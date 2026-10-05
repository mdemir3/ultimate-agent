"""OpenAI Responses adapter with durable, conservative cost reservations."""
import datetime
import json
import math
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

class BudgetExceeded(ValueError):
    pass

class Budget:
    def __init__(self, path, task_limit=2.0, daily_limit=10.0):
        if not all(math.isfinite(x) and x > 0 for x in (task_limit, daily_limit)):
            raise ValueError('Budget limits must be positive finite numbers.')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_limit, self.daily_limit, self.spent = task_limit, daily_limit, 0.0
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, day TEXT, amount REAL)')

    def reserve(self, amount):
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError('Invalid cost reservation.')
        day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute('BEGIN IMMEDIATE')
            total = db.execute('SELECT COALESCE(SUM(amount),0) FROM charges WHERE day=?', (day,)).fetchone()[0]
            if self.spent + amount > self.task_limit or total + amount > self.daily_limit:
                raise BudgetExceeded('Budget limit reached; no request was sent.')
            row = db.execute('INSERT INTO charges(day,amount) VALUES (?,?)', (day, amount)).lastrowid
        self.spent += amount
        return row

    def settle(self, row, reservation, actual):
        # Never credit more than reserved. Unexpected excess is charged in full.
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE charges SET amount=? WHERE id=?', (actual, row))
        self.spent += actual - reservation

class OpenAIProvider:
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
        spec = self.config['models'][tier]
        payload = {'model': spec['id'], 'instructions': instructions,
                   'input': [{'role': 'user', 'content': json.dumps(transcript, ensure_ascii=False)}],
                   'max_output_tokens': 2048, 'store': False}
        if spec.get('reasoning_effort'):
            payload['reasoning'] = {'effort': spec['reasoning_effort']}
        if tools:
            payload['tools'] = tools
            payload['parallel_tool_calls'] = False
        if schema:
            payload['text'] = {'format': {'type': 'json_schema', 'name': 'assessment',
                                          'strict': True, 'schema': schema}}
        data = json.dumps(payload, ensure_ascii=False).encode()
        # UTF-8 byte count plus margin is intentionally conservative for text requests.
        input_bound = len(data) + 2048
        if input_bound > spec.get('input_token_limit', 64000):
            raise ValueError('Context limit reached; narrow the task or use fewer files.')
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
        usage = result.get('usage')
        if usage and all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')):
            actual = (usage['input_tokens'] * spec['input_usd_per_million'] + usage['output_tokens'] * spec['output_usd_per_million']) / 1e6
            self.budget.settle(row, reservation, actual)
        return result

def response_text(result):
    return '\n'.join(c.get('text', '') for item in result.get('output', [])
                     if item.get('type') == 'message' for c in item.get('content', [])
                     if c.get('type') == 'output_text')
