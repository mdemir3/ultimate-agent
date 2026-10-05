"""Local Ollama adapter. Requests go only to the configured host; there is no cost ledger."""
import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .chat import chat_messages, chat_tools, error_detail, reply_output, setting
from .policy import TIERS

DEFAULT_OLLAMA = {
    'host': 'http://127.0.0.1:11434',
    # Each tier uses the first installed candidate that supports tool calling. In local tests
    # (README), llama3.1/3.2 and qwen2.5-coder:3b completed no tasks, so they are not defaults.
    'models': {
        'fast': ['qwen2.5-coder:7b'],
        'balanced': ['qwen2.5-coder:7b'],
        'deep': ['qwen2.5-coder:14b', 'qwen2.5-coder:7b'],
    },
    'num_ctx': 16384,
    'max_output_tokens': 2048,
    'temperature': 0.2,
    'timeout_seconds': 300,
}

def tag(name):
    return name if ':' in name else name + ':latest'

class OllamaProvider:
    def __init__(self, config, emit=None, opener=None, tiers=TIERS):
        if not isinstance(config, dict) or not isinstance(config.get('models', {}), dict):
            raise ValueError('Configure ollama as an object with a models object.')
        self.config = c = {**DEFAULT_OLLAMA, **config, 'models': {**DEFAULT_OLLAMA['models'], **config.get('models', {})}}
        host = urlparse(c['host']) if isinstance(c['host'], str) else None
        if not host or host.scheme not in ('http', 'https') or not host.netloc or host.path not in ('', '/'):
            raise ValueError('Configure ollama.host as a URL such as http://127.0.0.1:11434.')
        self.host = c['host'].rstrip('/')
        setting(c, 'num_ctx', 2048, 1048576)
        setting(c, 'max_output_tokens', 256, c['num_ctx'] // 2)
        setting(c, 'temperature', 0, 2, (int, float))
        setting(c, 'timeout_seconds', 1, 3600, (int, float))
        self.emit = emit or (lambda event: None)
        # Bypass system proxies so local requests stay on this machine.
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        installed = {m.get('name') for m in self.request('/api/tags', None, 10).get('models', []) if isinstance(m, dict)}
        self.models, capable = {}, {}
        for tier in tiers:
            candidates = c['models'][tier]
            candidates = [candidates] if isinstance(candidates, str) else candidates
            if not isinstance(candidates, list) or not candidates or not all(isinstance(n, str) and n for n in candidates):
                raise ValueError('Configure ollama.models.%s as a model name or a list of names.' % tier)
            for name in map(tag, candidates):
                if name in installed and name not in capable:
                    capabilities = self.request('/api/show', {'model': name}, 10).get('capabilities')
                    capable[name] = isinstance(capabilities, list) and 'tools' in capabilities
                if capable.get(name):
                    self.models[tier] = name
                    break
            else:
                raise ValueError('No installed Ollama model with tool support for the %s tier (tried %s). '
                                 'Install one with `ollama pull %s`, or edit ollama.models in your config.'
                                 % (tier, ', '.join(candidates), candidates[0]))

    def request(self, path, payload, timeout):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(self.host + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Local server errors (e.g. not enough memory for a model) are useful to show.
            raise ValueError('Ollama returned HTTP %s%s. No automatic retry.' % (exc.code, error_detail(exc))) from None
        except OSError as exc:
            if isinstance(getattr(exc, 'reason', exc), socket.timeout):
                raise ValueError('Ollama did not respond within %s seconds. No automatic retry; '
                                 'raise ollama.timeout_seconds or use a smaller model.' % timeout) from None
            raise ValueError('Cannot reach Ollama at %s. Start it with `ollama serve` or open the Ollama app.' % self.host) from None
        except ValueError:
            raise ValueError('Ollama returned a response that is not JSON.') from None
        if not isinstance(result, dict):
            raise ValueError('Ollama returned an unexpected response.')
        return result

    def call(self, tier, instructions, transcript, tools=None, schema=None):
        c, model = self.config, self.models[tier]
        payload = {'model': model, 'stream': False, 'messages': chat_messages(instructions, transcript, 'ollama'),
                   'options': {'num_ctx': c['num_ctx'], 'num_predict': c['max_output_tokens'], 'temperature': c['temperature']}}
        if tools:
            payload['tools'] = chat_tools(tools)
        if schema:
            payload['format'] = schema
        # Ollama truncates prompts longer than num_ctx instead of failing, which could drop
        # the original request. ~2 UTF-8 bytes per token is a conservative estimate, not a bound.
        if len(json.dumps(payload, ensure_ascii=False).encode()) // 2 + c['max_output_tokens'] > c['num_ctx']:
            raise ValueError('Context limit reached; narrow the task, use fewer files, or raise ollama.num_ctx.')
        result = self.request('/api/chat', payload, c['timeout_seconds'])
        message = result.get('message')
        if not isinstance(message, dict):
            raise ValueError('Ollama returned an unexpected response.')
        duration = result.get('total_duration')
        self.emit({'event': 'model_call', 'provider': 'ollama', 'tier': tier, 'model': model,
                   'input_tokens': result.get('prompt_eval_count'), 'output_tokens': result.get('eval_count'),
                   'seconds': round(duration / 1e9, 1) if isinstance(duration, (int, float)) else None})
        output = reply_output(message.get('tool_calls'), message.get('content'), tools)
        # done_reason "length" means max_output_tokens cut the answer off.
        status = 'completed' if result.get('done') is True and result.get('done_reason') != 'length' else 'incomplete'
        return {'status': status, 'output': output}
