"""Bring your own model: any OpenAI-compatible Chat Completions API, or a Python adapter class.

CompatibleProvider works with services that copy OpenAI's chat API, and with local servers such
as LM Studio or vLLM. CustomProvider wraps a class you write, for a model with any other interface.
"""
import importlib
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .chat import chat_messages, chat_tools, error_detail, reply_output, setting
from .policy import TIERS

# Requests to these hosts stay on this machine, so plain http and missing prices are allowed.
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}
DEFAULT_COMPATIBLE = {
    'base_url': '',
    'api_key_env': '',
    'models': {'fast': '', 'balanced': '', 'deep': ''},
    'prices': {},
    'context_tokens': 32768,
    'max_output_tokens': 2048,
    'temperature': None,
    'timeout_seconds': 120,
}
DEFAULT_CUSTOM = {'adapter': '', 'options': {}}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the API key to another host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('The model API redirected the request; set compatible.base_url to the final URL.')

class CompatibleProvider:
    """Calls <base_url>/chat/completions with each tier's model. Remote APIs must use https and list prices."""
    def __init__(self, config, budget, emit=None, opener=None, tiers=TIERS):
        if not isinstance(config, dict) or not all(isinstance(config.get(k, {}), dict) for k in ('models', 'prices')):
            raise ValueError('Configure compatible as an object with models and prices objects.')
        self.config = c = {**DEFAULT_COMPATIBLE, **config}
        url = urlparse(c['base_url']) if isinstance(c['base_url'], str) else None
        if not url or url.scheme not in ('http', 'https') or not url.hostname:
            raise ValueError('Set compatible.base_url to the API root, such as https://api.groq.com/openai/v1.')
        self.local = url.hostname in LOCAL_HOSTS
        if url.scheme == 'http' and not self.local:
            raise ValueError('Use https for a remote compatible.base_url so the API key is not sent in clear text.')
        self.endpoint = c['base_url'].rstrip('/') + '/chat/completions'
        # Each tier needs a model ID from your provider's catalog.
        self.models = {}
        for tier in tiers:
            model = c['models'].get(tier)
            if not isinstance(model, str) or not model:
                raise ValueError('Set compatible.models.%s to a model ID from your provider.' % tier)
            self.models[tier] = model
        # Remote APIs need a price per model so budgets apply; 0 marks a free model.
        self.prices = {}
        for model in set(self.models.values()):
            price = c['prices'].get(model)
            if price is None and self.local:
                continue
            rates = [price.get(k) for k in ('input_usd_per_million', 'output_usd_per_million')] if isinstance(price, dict) else [None]
            if not all(isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) and r >= 0 for r in rates):
                raise ValueError('Set compatible.prices["%s"] to its input_usd_per_million and output_usd_per_million '
                                 '(use 0 for a free model).' % model)
            self.prices[model] = rates
        # The API key comes from the environment variable named in api_key_env, never from the config file.
        key_env = c['api_key_env']
        if not isinstance(key_env, str) or (key_env and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env)):
            raise ValueError('Set compatible.api_key_env to the name of an environment variable, or "" for none.')
        self.key = os.environ.get(key_env) if key_env else None
        if key_env and not self.key:
            raise ValueError('Set %s in your terminal; never paste API keys into chat or config.' % key_env)
        setting(c, 'context_tokens', 2048, 10000000, section='compatible')
        setting(c, 'max_output_tokens', 256, c['context_tokens'] // 2, section='compatible')
        if c['temperature'] is not None:
            setting(c, 'temperature', 0, 2, (int, float), section='compatible')
        setting(c, 'timeout_seconds', 1, 3600, (int, float), section='compatible')
        self.budget, self.emit = budget, emit or (lambda event: None)
        # Local servers bypass system proxies; remote requests never follow redirects.
        self.opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if self.local else urllib.request.ProxyHandler(), NoRedirect())

    def call(self, tier, instructions, transcript, tools=None, schema=None):
        """Send one agent step and return Responses-style output; priced models reserve budget first."""
        c, model = self.config, self.models[tier]
        payload = {'model': model, 'messages': chat_messages(instructions, transcript), 'max_tokens': c['max_output_tokens']}
        if c['temperature'] is not None:
            payload['temperature'] = c['temperature']
        if tools:
            payload['tools'] = chat_tools(tools)
        if schema:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'assessment', 'strict': True, 'schema': schema}}
        data = json.dumps(payload, ensure_ascii=False).encode()
        # Some local servers truncate long prompts silently. ~2 UTF-8 bytes per token is an estimate, not a bound.
        if len(data) // 2 + c['max_output_tokens'] > c['context_tokens']:
            raise ValueError('Context limit reached; narrow the task, use fewer files, or raise compatible.context_tokens.')
        rates, row, reservation = self.prices.get(model), None, 0.0
        if rates and any(rates):
            # The UTF-8 byte count conservatively bounds text input tokens, as in the OpenAI adapter.
            reservation = (len(data) * rates[0] + c['max_output_tokens'] * rates[1]) / 1e6
            row = self.budget.reserve(reservation)
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        retained = '; cost reservation retained' if row else ''
        started = time.monotonic()
        try:
            with self.opener.open(urllib.request.Request(self.endpoint, data=data, headers=headers),
                                  timeout=c['timeout_seconds']) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError('Model API returned HTTP %s%s. No automatic retry%s.' % (exc.code, error_detail(exc), retained)) from None
        except OSError as exc:
            if isinstance(getattr(exc, 'reason', exc), socket.timeout):
                raise ValueError('Model API did not respond within %s seconds. No automatic retry%s.' % (c['timeout_seconds'], retained)) from None
            raise ValueError('Cannot reach the model API at %s. No automatic retry%s.' % (c['base_url'], retained)) from None
        except ValueError:
            raise ValueError('Model API returned a response that is not JSON.') from None
        choices = result.get('choices') if isinstance(result, dict) else None
        message = choices[0].get('message') if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError('Model API returned an unexpected response.')
        usage = result.get('usage') if isinstance(result.get('usage'), dict) else {}
        tokens = [usage.get(k) if type(usage.get(k)) is int and usage.get(k) >= 0 else None
                  for k in ('prompt_tokens', 'completion_tokens')]
        # Settle the reservation to the actual usage when the API reports it.
        if row and None not in tokens:
            self.budget.settle(row, reservation, (tokens[0] * rates[0] + tokens[1] * rates[1]) / 1e6)
        self.emit({'event': 'model_call', 'provider': 'compatible', 'tier': tier, 'model': model,
                   'input_tokens': tokens[0], 'output_tokens': tokens[1], 'seconds': round(time.monotonic() - started, 1)})
        output = reply_output(message.get('tool_calls'), message.get('content'), tools)
        # finish_reason "length" means max_output_tokens cut the answer off.
        return {'status': 'incomplete' if choices[0].get('finish_reason') == 'length' else 'completed', 'output': output}

class CustomProvider:
    """Wrap your own class, loaded from custom.adapter as "module:ClassName" and created with custom.options.
    It needs complete(tier, messages, tools, schema) taking OpenAI chat-format messages and tools,
    returning {"text": str} or {"tool_calls": [{"name": str, "arguments": dict}]} (add "truncated": true if cut off)."""
    def __init__(self, config, emit=None, tiers=TIERS):
        spec = config.get('adapter') if isinstance(config, dict) else None
        if not isinstance(spec, str) or not re.fullmatch(r'[\w.]+:\w+', spec):
            raise ValueError('Set custom.adapter to "module:ClassName", importable from the current folder or PYTHONPATH.')
        # Import the class named in the config and create it with custom.options.
        module_name, class_name = spec.split(':')
        try:
            adapter_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError('Cannot load custom.adapter %s (%s).' % (spec, exc)) from None
        options = config.get('options', {})
        self.adapter = adapter_class(options if isinstance(options, dict) else {})
        if not callable(getattr(self.adapter, 'complete', None)):
            raise ValueError('custom.adapter %s needs a complete(tier, messages, tools, schema) method.' % spec)
        # Optional: the adapter's own model name per tier, shown in results.
        names = getattr(self.adapter, 'models', None)
        self.models = {tier: str(names[tier]) if isinstance(names, dict) and tier in names else spec for tier in tiers}
        self.emit = emit or (lambda event: None)

    def call(self, tier, instructions, transcript, tools=None, schema=None):
        """Pass one agent step to the adapter and convert its reply."""
        started = time.monotonic()
        reply = self.adapter.complete(tier, chat_messages(instructions, transcript), chat_tools(tools) if tools else None, schema)
        self.emit({'event': 'model_call', 'provider': 'custom', 'tier': tier, 'model': self.models[tier],
                   'seconds': round(time.monotonic() - started, 1)})
        calls = reply.get('tool_calls') if isinstance(reply, dict) else None
        if calls:
            if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
                raise ValueError('The custom adapter returned malformed tool_calls.')
            output = reply_output([{'function': call} for call in calls], None, tools)
        elif isinstance(reply, dict) and isinstance(reply.get('text'), str):
            # As with the other providers, a tool call written as JSON text counts as that call.
            output = reply_output(None, reply['text'], tools)
        else:
            raise ValueError('The custom adapter must return {"text": ...} or {"tool_calls": [...]}.')
        return {'status': 'incomplete' if reply.get('truncated') else 'completed', 'output': output}
