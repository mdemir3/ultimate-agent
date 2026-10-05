"""Helpers shared by the chat-format adapters (Ollama, OpenAI-compatible APIs, Python adapters)."""
import json
import math

from .safety import SECRET

def text_tool_calls(content, names):
    """Some models write tool calls as JSON in prose or code fences instead of tool_calls.
    Extract calls to offered tools; a reply without one stays a text answer. This grants nothing
    the model could not request natively, and the agent validates every call."""
    decoder, calls, i = json.JSONDecoder(), [], content.find('{')
    while i != -1:
        end = i + 1
        try:
            call, parsed_end = decoder.raw_decode(content, i)
        except ValueError:
            call = None
        arguments = call.get('arguments', call.get('parameters')) if isinstance(call, dict) and len(call) == 2 else None
        if arguments is not None and isinstance(call.get('name'), str) and call['name'] in names:
            calls.append({'type': 'function_call', 'name': call['name'],
                          'arguments': arguments if isinstance(arguments, str) else json.dumps(arguments)})
            end = parsed_end
        i = content.find('{', end)
    return calls or None

def chat_messages(instructions, transcript, style='openai'):
    """Replay agent actions as native tool turns; models follow these better than one JSON record.
    Ollama takes arguments as objects and names the tool; OpenAI-style APIs pair calls by id."""
    messages, pending = [{'role': 'system', 'content': instructions}], []
    for index, entry in enumerate(transcript):
        if isinstance(entry, dict) and 'action' in entry:
            if pending:
                messages.append({'role': 'user', 'content': json.dumps(pending, ensure_ascii=False)})
                pending = []
            name, arguments = entry['action'], entry.get('arguments', {})
            observation = json.dumps(entry['observation'], ensure_ascii=False)
            if style == 'ollama':
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]})
                messages.append({'role': 'tool', 'tool_name': name, 'content': observation})
            else:
                call_id = 'call_%d' % index
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]})
                messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': observation})
        else:
            pending.append(entry)
    if pending:
        messages.append({'role': 'user', 'content': json.dumps(pending, ensure_ascii=False)})
    return messages

def chat_tools(tools):
    return [{'type': 'function', 'function': {k: t[k] for k in ('name', 'description', 'parameters')}} for t in tools]

def text_output(text):
    return {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}

def reply_output(tool_calls, content, tools):
    """Convert a chat reply to the Responses-style output that the agent loop already validates."""
    output = []
    for item in tool_calls if isinstance(tool_calls, list) else []:
        function = item.get('function') if isinstance(item, dict) else None
        function = function if isinstance(function, dict) else {}
        arguments = function.get('arguments', {})
        output.append({'type': 'function_call', 'name': function.get('name'),
                       'arguments': arguments if isinstance(arguments, str) else json.dumps(arguments)})
    if not output and isinstance(content, str) and content.strip():
        # Several calls in one reply are passed on; the agent then refuses to run any of them.
        output.extend(text_tool_calls(content, {t['name'] for t in tools or ()}) or [text_output(content)])
    return output

def setting(config, name, low, high, kinds=(int,), section='ollama'):
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, kinds) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Configure %s.%s between %s and %s.' % (section, name, low, high))
    return value

def error_detail(exc):
    """A short server error message, withheld if it might contain a credential."""
    try:
        body = json.load(exc)
    except (OSError, ValueError, AttributeError):
        return ''
    error = body.get('error') if isinstance(body, dict) else None
    detail = str((error.get('message') if isinstance(error, dict) else error) or '')[:300]
    return ': ' + detail if detail and not SECRET.search(detail) else ''
