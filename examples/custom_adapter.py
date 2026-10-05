"""Template for connecting a model that is not behind an OpenAI-compatible API.

Copy this file, replace ask_model() with a call to your model, then set in ultimate.config.json:

    "provider": "custom",
    "custom": {"adapter": "custom_adapter:MyModel", "options": {"model": "qwen2.5-coder:7b"}}

Run Ultimate from the folder that contains this file, or add that folder to PYTHONPATH:

    PYTHONPATH=examples python3 -m ultimate fit --provider custom

As written, it forwards requests to a local Ollama server so you can try it unchanged.
"""
import json
import urllib.request


class MyModel:
    def __init__(self, options):
        # "options" is the custom.options object from your config.
        self.model = options.get('model', 'qwen2.5-coder:7b')
        self.url = options.get('url', 'http://127.0.0.1:11434/v1/chat/completions')
        # Optional: the model name shown in results for each tier.
        self.models = {tier: self.model for tier in ('fast', 'balanced', 'deep')}

    def complete(self, tier, messages, tools, schema=None):
        """Called once per agent step.

        tier: "fast", "balanced" or "deep"; use it to pick one of your models.
        messages, tools: OpenAI chat format. Earlier tool calls and results are included.
        schema: a JSON schema when a structured answer is needed (only with --router llm), else None.

        Return {"text": "..."} for a final answer, or
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]} to use one tool.
        Add "truncated": True if your model stopped because of an output limit.
        """
        reply = ask_model(self.url, self.model, messages, tools)
        calls = reply.get('tool_calls') or []
        if calls:
            return {'tool_calls': [{'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments'] or '{}')}
                                   for c in calls]}
        return {'text': reply.get('content') or ''}


def ask_model(url, model, messages, tools):
    """Replace this with your model's SDK or HTTP call."""
    payload = {'model': model, 'messages': messages}
    if tools:
        payload['tools'] = tools
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)['choices'][0]['message']
