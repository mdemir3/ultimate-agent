"""Deterministic routing policy: picks a tier from the wording of the prompt.

The rules are plain keyword checks, so the same prompt always gets the same tier.
Model recommendations (Jev or the LLM judge) may raise this tier, never lower it.
"""
import re
from dataclasses import asdict, dataclass

# Ordered from quickest to strongest; code compares tiers by their position here.
TIERS = ('fast', 'balanced', 'deep')
# Work where a mistake is costly (security, money, data): always deep, marked high risk.
RISK = r'\b(auth(?:entication|orization)?|log[- ]?in|sign[- ]?in|passwords?|payments?|billing|permissions?|secrets?|cryptograph\w*|migration|production|delete|drop table)\b'
# Problems that need investigation, such as bugs that only happen sometimes: deep.
HARD = r'\b(race condition|deadlock|concurren\w*|intermittent(?:ly)?|occasional(?:ly)?|sometimes|sporadic(?:ally)?|flak(?:y|iness)|randomly|distributed|architecture|redesign|cross[- ]service|memory leak)\b'
# Small mechanical edits: fast.
SIMPLE = r'\b(typo|spelling|rename|format|label|comment|docstring)\b'
# Requests made only of these words name no target that inspection could find.
VAGUE = set('''fix fixed make work works working do it it's its this that these those please help the a an now
again broken issue issues problem problems bug bugs error errors stuff thing things me up can could you is
isn't not doesn't solve sort out handle just all everything something still my our code look at check'''.split())

@dataclass
class Decision:
    """The routing result: which tier to use, how much reasoning effort to ask for, and why."""
    tier: str
    effort: str
    risk: str
    reasons: list
    policy_version: str = '1.0'

    def to_dict(self):
        return asdict(self)

def route(prompt, file_count=0, previous_failures=0, lock=None):
    """Choose a tier for a prompt.

    Starts at balanced, drops to fast for a small mechanical edit, and rises to deep for
    risky or hard work (risk wins over simplicity). A --lock may raise the tier, but it
    cannot go below what these rules require.
    """
    if not prompt.strip():
        raise ValueError('Please provide a nonempty prompt.')
    risk = bool(re.search(RISK, prompt, re.I))
    hard = bool(re.search(HARD, prompt, re.I))
    reasons = []
    # Start in the middle; the checks below move the tier down or up.
    tier = 'balanced'
    if re.search(SIMPLE, prompt, re.I) and len(prompt) < 600 and file_count <= 3:
        tier = 'fast'
        reasons.append('A scoped mechanical edit is indicated.')
    if file_count > 3:
        reasons.append('Several files are explicitly in scope.')
    if risk:
        tier = 'deep'
        reasons.append('Consequential domain: stronger review is warranted.')
    if hard or previous_failures:
        tier = 'deep'
        reasons.append('Complex investigation or a previous failed attempt.')
    if not reasons:
        reasons.append('Scope or reasoning needs are uncertain; use the balanced default.')
    # --lock fixes the tier, but never below the floor the rules just set.
    if lock:
        if lock not in TIERS:
            raise ValueError('Unknown model tier.')
        if TIERS.index(lock) < TIERS.index(tier):
            raise ValueError('The requested lock is below the policy quality floor.')
        tier = lock
        reasons.append('User model lock applied.')
    # Effort is the reasoning setting passed to models that support one.
    return Decision(tier, 'low' if tier == 'fast' else 'medium' if tier == 'balanced' else 'high',
                    'high' if risk else 'normal', reasons)

def needs_clarification(prompt, file_count=0):
    """Ask before any model call when the request has no target and no files supply context.

    "Fix it" -> True. "Fix it" with --file login.py -> False. "Fix issue 42" -> False.
    """
    words = re.findall(r"[a-z0-9']+", prompt.lower().replace('’', "'"))
    return file_count == 0 and len(words) <= 6 and all(w in VAGUE for w in words)

def judge_upgrade(decision, recommendation):
    """Raise the tier if a model judge recommends a stronger one; never lower it.
    Judge output is untrusted and cannot select a provider or change permissions."""
    tier = recommendation.get('tier')
    if tier not in TIERS:
        raise ValueError('Invalid judge tier.')
    if TIERS.index(tier) > TIERS.index(decision.tier):
        return Decision(tier, 'high' if tier == 'deep' else 'medium', decision.risk,
                        decision.reasons + ['AI assessment raised the quality tier.'])
    return decision
