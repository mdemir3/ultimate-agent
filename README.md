# Ultimate Agent 0.4 — automatic model routing for coding tasks, with your own model

One prompt, automatic model selection, bounded escalation, and evidence from real checks.

You give one coding request. Ultimate sorts it into a tier (fast, balanced or deep), sends it to the model you assigned to that tier, lets the model read and edit files through guarded tools, escalates once if the work proves harder, and reports whether your tests passed. It is a standalone command-line agent for macOS/Linux using Python 3.9+ and only the standard library. It does not change the selected model inside Codex, Cursor or Claude Code.

It works with OpenAI, local [Ollama](#run-locally-with-ollama--no-api-key-no-cloud) models, any [OpenAI-compatible API](#any-openai-compatible-api), or [your own adapter](#your-own-adapter). Before relying on a model, run the fit test to see which tiers it can handle.

## Is it a fit for your model? Run the fit test

```sh
python3 -m ultimate fit --provider ollama --model qwen2.5-coder:7b
```

The fit test gives your model six small coding projects, two per tier. Each is scored by an acceptance check that runs after the agent stops; the model never sees it, so the score does not depend on what the model claims. The projects are built-in samples, so none of your code is sent anywhere.

| Tier | Tasks |
|---|---|
| fast | fix a spelling mistake; rename a function and its call |
| balanced | fix a bug covered by tests; add a small function |
| deep | find an intermittent login bug; find a stale-cache bug |

Real output on an Apple M1 Pro with 16 GB (2026-10-01):

```text
Ultimate fit test — provider ollama; fast=qwen2.5-coder:7b, balanced=qwen2.5-coder:7b, deep=qwen2.5-coder:7b

Task                                   Tier      Model             Result  Agent status   Time
Fix a spelling mistake (fast)          fast      qwen2.5-coder:7b  PASS    unverified     16s
Rename a function and its call (fast)  fast      qwen2.5-coder:7b  PASS    checks_passed  13s
Fix a bug covered by tests (balanced)  balanced  qwen2.5-coder:7b  PASS    checks_passed  12s
Add a small feature (balanced)         balanced  qwen2.5-coder:7b  PASS    checks_passed  12s
Find an intermittent login bug (deep)  deep      qwen2.5-coder:7b  FAIL    checks_failed  24s
Find a stale-cache bug (deep)          deep      qwen2.5-coder:7b  FAIL    answered       9s

By tier:
  fast      2/2  good fit
  balanced  2/2  good fit
  deep      0/2  not a fit yet

Use this setup for fast and balanced tasks; route deep tasks to a stronger model.
```

- **Verdicts:** at least 80% passed is a good fit, 50–79% partial, below 50% not a fit yet. Six tasks is a small sample: add `--runs 3` for steadier numbers.
- **Options:** `--tasks deep` or `--tasks typo,login` runs a subset. `--json results.json` saves every run. `--model` puts one model on every tier (Ollama and OpenAI-compatible providers); without it, the tiers in your config are tested as configured.
- **Cost:** paid providers reserve cost before each request, and `--budget` (default $2) caps the whole fit test. Exit code 0 means the test completed, whatever the score; 2 means it stopped early or the setup is invalid.

## Bring your own model

| Provider | Use it for | Setup |
|---|---|---|
| `ollama` | Local models; no key, nothing leaves your machine | [Run locally with Ollama](#run-locally-with-ollama--no-api-key-no-cloud) |
| `compatible` | Any API that speaks OpenAI Chat Completions with tool calling, hosted or local | [below](#any-openai-compatible-api) |
| `custom` | Anything else, through a small Python class | [below](#your-own-adapter) |
| `openai` | OpenAI models through the Responses API | [Connect your models once](#connect-your-models-once) |

`python3 -m ultimate init --provider <name>` writes a config with every section; `provider` in it selects the default, [`tier_providers`](#mix-providers-by-tier) sends chosen tiers elsewhere, and `--provider` puts every tier on one provider for a single command. The model needs tool (function) calling: the agent works by calling tools such as `read_file` and `edit_file`.

### Mix providers by tier

Keep easy work on a free local model and send only hard work to a stronger hosted one:

```json
"provider": "ollama",
"tier_providers": {"deep": "compatible"},
"compatible": {
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY",
  "models": {"deep": "your-strong-model-id"},
  "prices": {"your-strong-model-id": {"input_usd_per_million": 3, "output_usd_per_million": 15}}
}
```

Fast and balanced tasks then run on Ollama, while deep tasks, and tasks that escalate to deep partway through, go to the hosted model. An escalated task keeps its history: the deep model sees the original request and every earlier action and result. Each provider needs settings only for the tiers it serves. Results show `provider` and `model` for the tier that finished the task, plus a `providers` map. `fit` tests the mix as configured.

Deep tasks send the request and selected files to that API. `--router jev` is refused while any tier runs on Ollama, to keep the local tiers local.

### Any OpenAI-compatible API

```json
"provider": "compatible",
"compatible": {
  "base_url": "https://api.groq.com/openai/v1",
  "api_key_env": "GROQ_API_KEY",
  "models": {"fast": "small-model-id", "balanced": "medium-model-id", "deep": "large-model-id"},
  "prices": {
    "small-model-id": {"input_usd_per_million": 0.05, "output_usd_per_million": 0.08}
  },
  "context_tokens": 32768,
  "max_output_tokens": 2048,
  "temperature": null,
  "timeout_seconds": 120
}
```

- `base_url` is the API root; Ultimate appends `/chat/completions`. Remote URLs must use https. Redirects are refused so the key never reaches another host.
- `api_key_env` names the environment variable that holds your key; the key itself never goes in the config. Use `""` for local servers without keys.
- `models` maps each tier to a model ID. The same ID may serve several tiers.
- `prices` is required for every model on a remote host, so budgets apply; use 0 for free models. Local hosts (`localhost`, `127.0.0.1`) need no prices.
- `context_tokens` should not exceed the model's context window. Requests whose estimated size would not fit are refused, because some servers silently drop the start of an over-long prompt.

Tested here against Ollama's compatible endpoint (`http://127.0.0.1:11434/v1`). Services that document this API include Groq, OpenRouter, Together, Mistral, DeepSeek, Google Gemini and Anthropic, and local servers such as LM Studio, vLLM and llama.cpp; they are untested here. Check your service's base URL, model IDs, tool-calling support and current prices. Azure OpenAI (`api-key` header) and OpenAI reasoning models (which need `max_completion_tokens`) are not supported through this provider; use `openai` for OpenAI models.

**Data flow:** task + explicitly selected files → local rules → your API → verification. Your provider's retention policy applies.

### Your own adapter

When your model has its own API, write a class with one method and point the config at it:

```json
"provider": "custom",
"custom": {"adapter": "custom_adapter:MyModel", "options": {"model": "my-model"}}
```

```python
class MyModel:
    def __init__(self, options):          # custom.options from the config
        self.models = {'fast': 'my-model', 'balanced': 'my-model', 'deep': 'my-model'}  # optional labels

    def complete(self, tier, messages, tools, schema=None):
        # messages and tools are in OpenAI chat format; tier is fast, balanced or deep.
        return {'text': 'final answer'}
        # or: return {'tool_calls': [{'name': 'read_file', 'arguments': {'path': 'app.py'}}]}
```

[examples/custom_adapter.py](examples/custom_adapter.py) is a complete template that runs as-is against a local Ollama server: `PYTHONPATH=examples python3 -m ultimate fit --provider custom` with the config above. The module must be importable from the current folder or `PYTHONPATH`. The agent validates every tool call your adapter returns, exactly as for the built-in providers.

## Try it immediately — no key or installation needed

Open a terminal in this folder:

```sh
python3 demo.py --router jev
python3 -m ultimate route "Fix a typo in README"
python3 -m ultimate route "Fix intermittent login failures"
```

The demo uses **scripted coding responses and Jev assessments**, with real edits and a real verification command inside a temporary project. It demonstrates handoff mechanics, not model intelligence. The default `route` command uses deterministic rules and never contacts a provider. Adding `--router jev` or `--router llm` makes a paid request.

## Run locally with Ollama — no API key, no cloud

Requires [Ollama](https://ollama.com) running on this machine and a model that supports tool calling:

```sh
ollama pull qwen2.5-coder:7b
python3 -m ultimate route "Find why login occasionally fails" --provider ollama
python3 -m ultimate run "Fix the addition bug and verify it" \
  --provider ollama \
  --workspace /absolute/path/to/your/project \
  --file calculator.py \
  --allow-write \
  --check "python3 -m unittest discover -s tests"
```

To make Ollama the default, run `python3 -m ultimate init --provider ollama` and drop `--provider`. Without a config file, `--provider ollama` uses the defaults below.

**Data flow:** task + explicitly selected files → local rules → Ollama at `ollama.host` (default `http://127.0.0.1:11434`; system proxies are bypassed) → verification. Nothing goes to a cloud service unless you point `ollama.host` at another machine. `--router jev` is refused with Ollama because it sends the task to TypeSafe; `--router llm` runs the classification on the local fast-tier model.

**How a model is picked.** The rules choose a tier. Each tier then uses the first model in its `ollama.models` list that is installed and reports tool support. If none is, the run stops and names the `ollama pull` command.

| Tier | Default candidates |
|---|---|
| fast | `qwen2.5-coder:7b` |
| balanced | `qwen2.5-coder:7b` |
| deep | `qwen2.5-coder:14b`, then `qwen2.5-coder:7b` |

Edit the lists to try other models, for example a smaller fast model once it passes your tasks. The result JSON shows `model` (serving the final tier) and `models` (the full mapping). Each request logs a `model_call` event with token counts and seconds.

| Prompt | Decision |
|---|---|
| "Fix this spelling mistake" | fast |
| "Add search to this page" | balanced |
| "Find why login occasionally fails" | deep, high risk |
| "Fix it" | `needs_clarification` before any model call; add `--file` to let the agent inspect instead |

The vague-request stop applies to every provider. It fires only when the prompt is made of generic words ("fix", "it", "the bug", …) and no `--file` is given.

**Measured results** (2026-10-01, Apple M1 Pro with 16 GB, Ollama 0.33.2, real CLI runs on small scratch projects):

| Task | `qwen2.5-coder:7b` | `llama3.1` (8B) |
|---|---|---|
| Fix a README typo (fast) | 5/5 fixed | 0/3 |
| Fix an addition bug, unit tests as the check (balanced) | 5/5 `checks_passed` | 0/3 |
| Find and fix an intermittent login bug: tokens from whole seconds collide (deep) | 0/5 | 0/3 |

In development runs, `llama3.2` (3B) and `qwen2.5-coder:3b` fixed none of the typo tasks either. On a 16 GB machine, the deep tier remains a 7B model unless `qwen2.5-coder:14b` (about 9 GB) is installed, and the 7B model did not solve the investigation task. This is a small sample, not a benchmark; measure your own tasks before relying on a tier.

Default Ollama configuration:

```json
"ollama": {
  "host": "http://127.0.0.1:11434",
  "models": {
    "fast": ["qwen2.5-coder:7b"],
    "balanced": ["qwen2.5-coder:7b"],
    "deep": ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]
  },
  "num_ctx": 16384,
  "max_output_tokens": 2048,
  "temperature": 0.2,
  "timeout_seconds": 300
}
```

Differences from the OpenAI path:

- **No cost.** Budgets are not consumed and `estimated_cost_usd` is 0. `--max-steps` and `ollama.timeout_seconds` bound a run.
- **Context.** Ollama truncates prompts longer than `num_ctx` instead of returning an error. The adapter refuses a request whose estimated size (about 2 UTF-8 bytes per token) would not fit. This is an estimate, not a guarantee. Raising `num_ctx` allows larger files and uses more memory.
- **Transcript.** Prior actions are replayed as native chat tool turns rather than one JSON record; in testing, local models completed more tasks this way.
- **Text tool calls.** Local models sometimes write a tool call as JSON text. The adapter extracts calls to the tools offered in that request; the agent validates them exactly as before and refuses several calls in one reply.
- **`answered`.** Exit 0 with `answered` can mean a local model only explained instead of editing. Check `changed_files`.

## Enable Jev

Create your configuration with `python3 -m ultimate init`, then set `TYPESAFE_API_KEY` in your terminal environment. Get your key through the [TypeSafe console](https://console.typesafe.ai); do not paste it into chat or configuration files. Existing configs remain usable: the new `jev` section is optional and defaults are merged in memory. `init` never overwrites an existing file.

Preview a real Jev decision without executing coding tools or requiring an OpenAI key:

```sh
python3 -m ultimate route "Fix intermittent login failures" --router jev --budget 0.02
```

To include selected project context, add `--workspace /absolute/path/to/project --file src/login.py`. Those selected files are sent to TypeSafe for assessment. The coding agent can inspect other files later, but Jev is called only once at the start of each task in this version.

For a live coding task, configure your OpenAI models and key as described below, then select Jev:

```sh
python3 -m ultimate run "Fix the addition bug and verify it" \
  --router jev \
  --workspace /absolute/path/to/your/project \
  --file calculator.py \
  --allow-write \
  --check "python3 -m unittest discover -s tests" \
  --budget 2
```

**Data flow:** task + explicitly selected files → Jev assessment → controller policy → coding LLM → verification. Enabling Jev sends the initial task record to TypeSafe in addition to the coding provider. Provider retention policies differ; the OpenAI `store: false` setting does not apply to TypeSafe.

Jev answers three separate questions: complexity (Choice), consequential impact (Noul), and essential missing requirements (Noul). The controller validates the returned schema, distributions, finite probabilities, confidence, model version, and usage. It then:

- Requires both confidence ≥ 0.7 and selected-option probability ≥ 0.8 to use the complexity selection; otherwise keeps at least balanced.
- Raises the floor to deep when consequential-impact probability ≥ 0.5.
- Returns `needs_clarification` before any coding LLM call or write when missing-requirements probability ≥ 0.7. It asks for intended behavior and acceptance criteria; it does not invent the missing requirements.
- Never lowers the existing rules-based floor, changes permissions, or overrides spending limits. A confident fast assessment therefore cannot downgrade an initially balanced/deep task in this conservative first integration.
- Skips paid assessment when a manual tier lock is selected. Existing bounded escalation remains controlled by the coding loop; Jev does not re-assess each action yet.

These thresholds are **initial engineering defaults, not empirically calibrated success probabilities**. Jev confidence describes the classification distribution, not a coding model's probability of completing the task. No training or account-specific fine-tuning was performed. Benchmark these settings before relying on cost savings.

Jev failures stop by default. To explicitly permit a logged rules fallback with at least balanced quality, use `--router-fallback rules`. Budget exhaustion, detected credentials, missing API credentials, and invalid configuration still stop; they cannot trigger this fallback. Network failures retain the cost reservation and are not retried automatically. No fallback sends data to an additional routing provider.

Default Jev configuration:

```json
"jev": {
  "model": "jev-1.13.0",
  "input_usd_per_million": 0.042,
  "max_input_bytes": 28000,
  "timeout_seconds": 15,
  "min_confidence": 0.7,
  "min_choice_probability": 0.8,
  "risk_threshold": 0.5,
  "ambiguity_threshold": 0.7
}
```

This is a field within the root configuration object. Model/version and input price were checked against [TypeSafe's model documentation](https://docs.typesafe.ai/models) on September 30, 2026; output tokens are currently free. Confirm pricing before use. The pinned version makes evaluation reproducible. Aliases such as `jev-latest` work too, but may change behavior. Each assessment logs the resolved model, question version, probabilities, confidence, and thresholds without source text.

The adapter directly calls the documented [System One HTTP endpoint](https://docs.typesafe.ai/api), so no additional package is required. [Confidence semantics](https://docs.typesafe.ai/confidence) are preserved: Noul probabilities do not carry a separate confidence value.

## Connect your models once

```sh
python3 -m ultimate init
```

Edit `ultimate.config.json`. Set an API model ID and current input/output USD prices per million tokens for each tier: `fast`, `balanced`, and `deep`. Choose models available in your account with function calling support. If using `--judge`, the fast model must also support structured outputs. Set `reasoning_effort` to `null` for a model that does not accept reasoning settings, or a supported effort value. Set `input_token_limit` conservatively below its context window, reserving room for 2,048 output tokens.

Model IDs and rates deliberately start empty: the agent will not invent availability or prices. This is one-time operator configuration; ordinary prompts are routed automatically. Current references: [API model catalog](https://developers.openai.com/api/docs/models), [pricing](https://developers.openai.com/api/docs/pricing), and [function calling](https://developers.openai.com/api/docs/guides/function-calling).

Set `OPENAI_API_KEY` in your terminal environment using your normal secret-management process. Do not put it in the configuration, source files, chat, or screenshots. A ChatGPT subscription does not configure this API client.

Start with read-only analysis:

```sh
python3 -m ultimate run "Explain the likely cause of the login bug" \
  --workspace /absolute/path/to/your/project \
  --file src/login.py \
  --budget 2
```

Allow edits and authorize one exact verification command for a trusted project:

```sh
python3 -m ultimate run "Fix the addition bug and verify it" \
  --workspace /absolute/path/to/your/project \
  --file calculator.py \
  --allow-write \
  --check "python3 -m unittest discover -s tests" \
  --budget 2 --daily-budget 10
```

The command must be appropriate for your project. Shell syntax such as pipes, variable expansion, and `&&` is not interpreted. Use an absolute interpreter path when you need a specific virtual environment. API credentials are not inherited by the verification subprocess.

Optional: `--router llm` (or the existing `--judge` alias) adds a paid OpenAI AI assessment before execution. Choose only one routing backend. Its classification may raise the tier, but cannot lower the policy floor or change permissions. `--lock deep` keeps the tier fixed. Locks below the initial policy floor are rejected. `--max-steps` defaults to 12 and is capped at 30.

## How Ultimate mode works

1. Classify the original request. Clear mechanical edits use fast; unknown or ordinary coding uses balanced; risky domains or complex investigation use deep. A request made only of generic words, with no `--file`, stops with `needs_clarification` before any model call.
2. If selected, run Jev or the LLM judge while enforcing the initial policy floor. Include only explicitly supplied files and bounded source files requested through tools.
3. Execute one tool action at a time. Available tools are listing, reading, authorized editing, the fixed verification command, and escalation.
4. Escalate at most once to deep, either on an explicit complexity request or after two failed checks. Known missing-dependency errors stop without escalation. This detection is heuristic, not an exhaustive diagnosis.
5. Carry the original task and observed tool outcomes into each request. No hidden reasoning is transferred. The controller keeps permission and spending policy unchanged.
6. After edits, require a fresh verification result when a command is configured. Otherwise label the result `unverified`.

The initial rules are transparent heuristics, not a trained quality predictor. Broad repository understanding depends on files the agent inspects; automatic learning from outcomes is not implemented. Tier selection is visible in terminal events.

## Guardrails and practical limits

- **Read-only default.** Writes require `--allow-write`; no model-provided shell commands are executed.
- **Source-file boundary.** Relative source paths only; traversal, hidden paths, symlinks, hard links, large files, credential-like names, and `ultimate.config.json` are excluded. Common dependency/build folders are skipped.
- **Conflict detection.** Existing files require their current SHA-256; concurrent changes are rejected. New files require `NEW`. File operations are not hardened against a malicious local process racing filesystem changes.
- **Recovery.** Originals are saved locally before replacement under `.ultimate/<run-id>/originals`, with a `recovery.json` mapping. New files map to `null`. Changes remain after failures for review; there is no automatic destructive rollback. Use Git or the originals to recover selectively.
- **Secrets.** Common credential patterns are blocked in prompts and file contents; matching test output is withheld. Pattern checks cannot detect every secret. Only use projects you are allowed to send to the configured provider.
- **Budget.** Per-task and UTC-day limits use persistent SQLite reservations. Every request, including Jev or the optional LLM judge, reserves conservative text-input and maximum-output cost before sending. Usage settles the estimate. Network failures retain the reservation and are not retried automatically. Costs depend on configured prices and provider accounting; this is not a provider-enforced billing ceiling. Set provider-side limits too if required. Daily accounting is shared only by runs using the same `--state-dir`.
- **Context and retries.** Input byte estimates, output tokens, tool steps, request timeout, and escalation count are bounded. No silent truncation of conversation history or automatic retries of uncertain API requests.
- **Privacy.** OpenAI requests use `store: false`; this is not a promise of zero provider retention. Events store routing/tool/status metadata, not prompts or source contents. Recovery files contain original source locally. Delete local recovery data according to your retention requirements.
- **Checks execute project code.** The process runs with your local user permissions and may access the network or other files. This is **not an OS sandbox**. Only authorize checks in trusted workspaces, or run the whole agent in a container. Changing the model never grants new controller permissions.
- **Evidence.** `checks_passed` means the chosen command returned successfully after the latest edit. It does not prove that all requirements are satisfied; generated edits may also affect tests. Review the diff for consequential work.

The agent never logs an API key. The OpenAI adapter never shows API error bodies; the Ollama and OpenAI-compatible adapters show the server's short error message (such as an unknown model or a memory error) and withhold it if it looks like a credential. Its model catalog and policies are ordinary local code/configuration that an operator controls, not model-editable runtime policy.

## Results and exit codes

The final JSON contains the status, selected tier, routing assessment metadata, escalation flag, changed files, verification result, answer, and estimated API cost. Treat the controller's status as authoritative if the model's prose disagrees.

- Exit 0: `answered` or `checks_passed`.
- Exit 2: `unverified`, `checks_failed`, `environment_blocked`, `needs_clarification`, `incomplete`, configuration errors, or budget stops.
- `fit`: exit 0 when the fit test completes, whatever the score; exit 2 when it stops early or the setup is invalid.

`answered` means the agent returned an answer without editing; it is not independently verified. Interrupted runs retain edits and pre-write recovery originals. Each run is one task; persistent chat/resume is not implemented.

## Validation

```sh
python3 -m unittest discover -s tests -v
```

119 automated tests cover routing floors, escalation, permission boundaries, invalid tool calls, credential detection, file conflicts, durable recovery, budget reservations, provider payloads, and a full edit/check workflow. Provider calls are mocked; real local verification commands execute. Jev-specific tests cover typed payloads, probability/confidence gates, clarification, explicit fallback, and shared budget enforcement. Ollama tests cover model selection by installed tool support, payloads and native tool turns, text tool-call extraction, context refusal, CLI provider selection, and the vague-request stop. Bring-your-own-model tests cover OpenAI-compatible payloads, keys, prices, budgets and error handling, the custom adapter contract, and the fit test itself: every task routes to its tier, every hidden check fails on the starting project and passes on a reference solution, and scripted perfect and idle models score as expected. Tier-routing tests cover mixed providers, per-tier validation, the `--provider` override, and escalation handing the transcript to the deep provider. Live provider access and model quality require validation with your configured account and representative tasks.

## Project layout

- `ultimate/policy.py`: routing rules and quality floors.
- `ultimate/agent.py`: execution, handoff, and verification loop.
- `ultimate/safety.py`: workspace tools and verification command.
- `ultimate/jev.py`: TypeSafe transport, typed assessment validation, and confidence-aware routing.
- `ultimate/provider.py`: Responses API transport and cost ledger.
- `ultimate/ollama.py`: local Ollama transport and per-tier model selection.
- `ultimate/compatible.py`: OpenAI-compatible Chat Completions transport and the custom adapter hook.
- `ultimate/chat.py`: chat-format helpers shared by those adapters, including tool-call conversion.
- `ultimate/fit.py`: fit-test tasks, hidden acceptance checks, and the scorecard.
- `ultimate/__main__.py`: command-line interface.
- `demo.py`: no-key demonstration.
- `examples/custom_adapter.py`: template for connecting your own model.
- `tests/`: controller, provider, and workflow tests.

Next extensions should be driven by evaluation: stronger task classification, independent acceptance checks, an OS sandbox, persistent conversations, and adapters for your preferred coding tool or provider.
