"""Ollama adapter tests (model choice, payloads, reply parsing), the routing examples and command-line use."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

from ultimate.__main__ import main
from ultimate.agent import TOOLS, Agent
from ultimate.ollama import OllamaProvider
from ultimate.policy import TIERS, needs_clarification, route
from ultimate.safety import Workspace
from test_ultimate import FakeProvider, answer


def chat_call(name, **args):
    """An Ollama chat reply in which the model calls one tool."""
    return {'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args}}]},
            'done': True, 'done_reason': 'stop', 'prompt_eval_count': 10, 'eval_count': 5, 'total_duration': 2e9}

def chat_text(text, done_reason='stop'):
    """An Ollama chat reply with plain text."""
    return {'message': {'role': 'assistant', 'content': text}, 'done': True, 'done_reason': done_reason}

class Server:
    """Fake local Ollama: scripted chat replies and recorded requests."""
    def __init__(self, chats=(), installed=('qwen2.5-coder:7b',), tools=('qwen2.5-coder:7b',), error=None):
        self.chats, self.installed, self.tools, self.error = list(chats), installed, tools, error
        self.requests = []
    def open(self, request, timeout):
        path = urlparse(request.full_url).path
        body = json.loads(request.data) if request.data else None
        self.requests.append((path, body))
        if self.error and path == '/api/chat':
            raise self.error
        if path == '/api/tags':
            result = {'models': [{'name': name} for name in self.installed]}
        elif path == '/api/show':
            result = {'capabilities': ['completion'] + (['tools'] if body['model'] in self.tools else [])}
        else:
            result = self.chats.pop(0)
        return io.BytesIO(json.dumps(result).encode())
    def paths(self, path):
        return [body for p, body in self.requests if p == path]

class ModelSelectionTests(unittest.TestCase):
    def test_first_installed_tool_model_per_tier(self):
        server = Server(installed=('qwen2.5-coder:7b', 'llama3.1:latest'), tools=('qwen2.5-coder:7b', 'llama3.1:latest'))
        provider = OllamaProvider({'models': {'deep': ['qwen2.5-coder:14b', 'llama3.1']}}, opener=server)
        self.assertEqual(provider.models, {'fast': 'qwen2.5-coder:7b', 'balanced': 'qwen2.5-coder:7b', 'deep': 'llama3.1:latest'})
        self.assertEqual(len(server.paths('/api/show')), 2)
    def test_models_without_tool_support_are_skipped(self):
        server = Server(installed=('codellama:latest', 'llama3.1:latest'), tools=('llama3.1:latest',))
        provider = OllamaProvider({'models': {tier: ['codellama', 'llama3.1'] for tier in TIERS}}, opener=server)
        self.assertEqual(set(provider.models.values()), {'llama3.1:latest'})
    def test_single_model_name_accepted(self):
        provider = OllamaProvider({'models': {tier: 'qwen2.5-coder:7b' for tier in TIERS}}, opener=Server())
        self.assertEqual(provider.models['fast'], 'qwen2.5-coder:7b')
    def test_missing_model_names_pull_command(self):
        with self.assertRaisesRegex(ValueError, 'ollama pull qwen2.5-coder:7b'):
            OllamaProvider({}, opener=Server(installed=()))
    def test_unreachable_server(self):
        class Offline:
            def open(self, request, timeout):
                raise urllib.error.URLError(ConnectionRefusedError())
        with self.assertRaisesRegex(ValueError, 'Cannot reach Ollama'):
            OllamaProvider({}, opener=Offline())
    def test_invalid_settings(self):
        for config in ({'host': 'ftp://localhost'}, {'host': 'http://localhost/api'}, {'num_ctx': 100},
                       {'temperature': 'hot'}, {'max_output_tokens': True}, {'models': {'fast': []}}, {'models': []}):
            with self.assertRaises(ValueError): OllamaProvider(config, opener=Server())

class ChatTests(unittest.TestCase):
    def provider(self, *chats, **config):
        self.server = Server(chats)
        self.events = []
        return OllamaProvider(config, self.events.append, self.server)
    def test_payload_replays_actions_as_native_turns(self):
        provider = self.provider(chat_text('Done'))
        transcript = [{'original_request': 'Fix a.py'}, {'action': 'read_file', 'arguments': {}, 'observation': {'path': 'a.py'}},
                      {'controller': 'Verification failed.'}]
        provider.call('balanced', 'Rules', transcript, tools=TOOLS)
        body = self.server.paths('/api/chat')[0]
        self.assertEqual(body['model'], 'qwen2.5-coder:7b')
        self.assertFalse(body['stream'])
        self.assertEqual(body['options'], {'num_ctx': 16384, 'num_predict': 2048, 'temperature': 0.2})
        self.assertEqual([m['role'] for m in body['messages']], ['system', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(body['messages'][3]['tool_name'], 'read_file')
        self.assertEqual(body['tools'][1], {'type': 'function', 'function': {k: TOOLS[1][k] for k in ('name', 'description', 'parameters')}})
    def test_tool_calls_become_function_calls(self):
        result = self.provider(chat_call('read_file', path='a.py')).call('fast', 'Rules', [], tools=TOOLS)
        self.assertEqual(result, {'status': 'completed', 'output': [
            {'type': 'function_call', 'name': 'read_file', 'arguments': '{"path": "a.py"}'}]})
        self.assertEqual(self.events[0], {'event': 'model_call', 'provider': 'ollama', 'tier': 'fast', 'model': 'qwen2.5-coder:7b',
                                          'input_tokens': 10, 'output_tokens': 5, 'seconds': 2.0})
    def test_text_tool_calls_only_for_offered_tools(self):
        call = '{"name": "read_file", "arguments": {"path": "a.py"}}'
        cases = [(call, TOOLS, ['function_call']), ('<tool_call>\n' + call + '\n</tool_call>', TOOLS, ['function_call']),
                 ('{"name": "read_file", "parameters": {"path": "a.py"}}', TOOLS, ['function_call']),
                 (call + '\n' + call, TOOLS, ['function_call', 'function_call']),
                 ('```json\n' + call + '\n```\nThis reads the file.', TOOLS, ['function_call']),
                 ('{"name": "run_check", "arguments": {}}\n{"note": 1}', TOOLS, ['function_call']),
                 ('First I will read it.\n```json\n' + call + '\n```', TOOLS, ['function_call']),
                 (call, None, ['message']), (call, TOOLS[:1], ['message']), ('No call here: {"a": 1}', TOOLS, ['message']),
                 ('{"name": "read_file", "arguments": {}, "extra": 1}', TOOLS, ['message'])]
        for content, tools, kinds in cases:
            result = self.provider(chat_text(content)).call('fast', 'Rules', [], tools=tools)
            self.assertEqual([item['type'] for item in result['output']], kinds, content)
    def test_truncated_output_is_incomplete(self):
        self.assertEqual(self.provider(chat_text('Partial', 'length')).call('fast', 'Rules', [])['status'], 'incomplete')
    def test_schema_uses_format(self):
        schema = {'type': 'object', 'properties': {'tier': {'type': 'string'}}, 'required': ['tier']}
        self.provider(chat_text('{"tier": "deep"}')).call('fast', 'Classify', [], schema=schema)
        body = self.server.paths('/api/chat')[0]
        self.assertEqual(body['format'], schema)
        self.assertNotIn('tools', body)
    def test_context_limit_blocks_request(self):
        provider = self.provider(chat_text('unused'), num_ctx=2048, max_output_tokens=1024)
        with self.assertRaisesRegex(ValueError, 'num_ctx'):
            provider.call('fast', 'Rules', [{'initial_file': {'content': 'x' * 5000}}])
        self.assertEqual(self.server.paths('/api/chat'), [])
    def test_http_error_shows_local_detail_without_retry(self):
        provider = self.provider()
        self.server.error = urllib.error.HTTPError('http://127.0.0.1:11434/api/chat', 500, 'error', {},
                                                   io.BytesIO(b'{"error": "model requires more system memory"}'))
        with self.assertRaisesRegex(ValueError, 'requires more system memory'):
            provider.call('fast', 'Rules', [])
        self.assertEqual(len(self.server.paths('/api/chat')), 1)

class AgentWorkflowTests(unittest.TestCase):
    def test_edit_and_check_through_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sum.py').write_text('def add(a, b): return a - b\n')
            ws = Workspace(tmp, True, [sys.executable, '-c', 'from sum import add; assert add(2, 3) == 5'])
            server = Server([chat_call('read_file', path='sum.py'),
                             chat_text(json.dumps({'name': 'edit_file', 'arguments': {'path': 'sum.py', 'content': 'def add(a, b): return a + b\n',
                                                                                      'expected_sha256': ws.read('sum.py')['sha256']}})),
                             chat_call('run_check'), chat_text('Fixed addition.')])
            result = Agent(OllamaProvider({}, opener=server), ws, route('Fix addition')).run('Fix addition')
            self.assertEqual(result['status'], 'checks_passed')
            self.assertEqual((Path(tmp) / 'sum.py').read_text(), 'def add(a, b): return a + b\n')
    def test_several_text_calls_execute_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            call = json.dumps({'name': 'edit_file', 'arguments': {'path': 'a.py', 'content': 'x=1', 'expected_sha256': 'NEW'}})
            server = Server([chat_text(call + '\n' + call)])
            result = Agent(OllamaProvider({}, opener=server), Workspace(tmp, True), route('Add a.py')).run('Add a.py')
            self.assertEqual(result['status'], 'incomplete')
            self.assertFalse((Path(tmp) / 'a.py').exists())

class LocalRoutingTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(route('Fix this spelling mistake').tier, 'fast')
        self.assertEqual(route('Add search to this page').tier, 'balanced')
        self.assertEqual(route('Find why login occasionally fails').tier, 'deep')
        self.assertEqual(route('Fix the flaky upload test').tier, 'deep')
    def test_vague_requests_need_clarification(self):
        for prompt in ('Fix it', 'fix this!', 'Make it work', 'It’s broken, please fix', 'Fix the bug'):
            self.assertTrue(needs_clarification(prompt), prompt)
        for prompt in ('Fix issue 42', 'Fix main.py', 'Add search to this page', 'Fix this spelling mistake'):
            self.assertFalse(needs_clarification(prompt), prompt)
        self.assertFalse(needs_clarification('Fix it', file_count=1))
    def test_edit_error_explains_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'a.py').write_text('x = 1\n')
            with self.assertRaisesRegex(ValueError, 'not NEW'):
                Workspace(tmp, True).edit('a.py', 'x = 2\n', 'NEW')

class StubOllama(FakeProvider):
    """Fake OllamaProvider with fixed model names, for command-line tests."""
    host = 'http://127.0.0.1:11434'
    models = {'fast': 'small:latest', 'balanced': 'coder:7b', 'deep': 'coder:14b'}

class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / 'state'
    def main(self, *args, provider=None):
        argv = ['ultimate', *args, '--config', str(self.root / 'config.json'), '--workspace', str(self.root),
                '--state-dir', str(self.state)]
        with patch('sys.argv', argv), patch('ultimate.__main__.OllamaProvider', return_value=provider) as ollama, \
                patch('ultimate.__main__.OpenAIProvider') as openai, patch('ultimate.__main__.JevRouter') as jev, \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = main()
        self.ollama, self.openai, self.jev, self.err = ollama, openai, jev, err.getvalue()
        return code, json.loads(out.getvalue()) if out.getvalue() else None
    def test_vague_request_asks_before_any_model(self):
        for command in ('route', 'run'):
            code, output = self.main(command, 'Fix it', '--provider', 'ollama')
            self.assertEqual((code, output['status']), (2, 'needs_clarification'))
            self.ollama.assert_not_called()
            self.assertFalse(self.state.exists())
    def test_route_preview_shows_local_model(self):
        code, output = self.main('route', 'Find why login occasionally fails', '--provider', 'ollama', provider=StubOllama([]))
        self.assertEqual((code, output['tier'], output['model']), (0, 'deep', 'coder:14b'))
    def test_jev_is_refused_with_ollama(self):
        code, _ = self.main('route', 'Fix typo', '--provider', 'ollama', '--router', 'jev')
        self.assertEqual(code, 2)
        self.assertIn('TypeSafe', self.err)
        self.jev.assert_not_called()
    def test_config_selects_ollama_for_run(self):
        (self.root / 'config.json').write_text('{"provider": "ollama"}')
        code, output = self.main('run', 'Implement a feature', provider=StubOllama([answer('Explained.')]))
        self.assertEqual((code, output['status'], output['provider'], output['model']), (0, 'answered', 'ollama', 'coder:7b'))
        self.openai.assert_not_called()
    def test_init_for_ollama(self):
        config = self.root / 'new.json'
        with patch('sys.argv', ['ultimate', 'init', '--provider', 'ollama', '--config', str(config)]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        written = json.loads(config.read_text())
        self.assertEqual(written['provider'], 'ollama')
        self.assertEqual(written['ollama']['models']['balanced'][0], 'qwen2.5-coder:7b')

if __name__ == '__main__':
    unittest.main()
