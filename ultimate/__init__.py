"""Ultimate Agent: bounded automatic routing for coding tasks.

How the files fit together, in the order a request flows through them:

    __main__.py    command line: reads the prompt and options, then builds everything below
    policy.py      picks a tier (fast, balanced or deep) from the wording of the prompt
    jev.py         optional paid classifier that may raise the tier or ask for clarification
    agent.py       the step loop: ask the model, run the one tool it chose, repeat
    safety.py      the tools themselves: list, read and edit files, run the check command
    provider.py    the OpenAI adapter and the spending budget
    ollama.py      local models through Ollama
    compatible.py  any OpenAI-compatible API, or your own Python adapter class
    chat.py        helpers shared by the chat-format adapters
    fit.py         the fit test, which scores a model setup on built-in tasks
"""
__version__ = '0.4.0'
