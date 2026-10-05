"""Deterministic policy. Model recommendations may raise, never lower, its floor."""
import re
from dataclasses import asdict, dataclass

TIERS = ('fast', 'balanced', 'deep')
RISK = r'\b(auth(?:entication|orization)?|log[- ]?in|sign[- ]?in|passwords?|payments?|billing|permissions?|secrets?|cryptograph\w*|migration|production|delete|drop table)\b'
HARD = r'\b(race condition|deadlock|concurren\w*|intermittent(?:ly)?|occasional(?:ly)?|sometimes|sporadic(?:ally)?|flak(?:y|iness)|randomly|distributed|architecture|redesign|cross[- ]service|memory leak)\b'
SIMPLE = r'\b(typo|spelling|rename|format|label|comment|docstring)\b'
# Requests made only of these words name no target that inspection could find.
VAGUE = set('''fix fixed make work works working do it it's its this that these those please help the a an now
again broken issue issues problem problems bug bugs error errors stuff thing things me up can could you is
isn't not doesn't solve sort out handle just all everything something still my our code look at check'''.split())

@dataclass
class Decision:
    tier: str
    effort: str
    risk: str
    reasons: list
    policy_version: str = '1.0'

    def to_dict(self):
        return asdict(self)

def route(prompt, file_count=0, previous_failures=0, lock=None):
    if not prompt.strip():
        raise ValueError('Please provide a nonempty prompt.')
    risk = bool(re.search(RISK, prompt, re.I))
    hard = bool(re.search(HARD, prompt, re.I))
    reasons = []
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
    if lock:
        if lock not in TIERS:
            raise ValueError('Unknown model tier.')
        if TIERS.index(lock) < TIERS.index(tier):
            raise ValueError('The requested lock is below the policy quality floor.')
        tier = lock
        reasons.append('User model lock applied.')
    return Decision(tier, 'low' if tier == 'fast' else 'medium' if tier == 'balanced' else 'high',
                    'high' if risk else 'normal', reasons)

def needs_clarification(prompt, file_count=0):
    """Ask before any model call when the request has no target and no files supply context."""
    words = re.findall(r"[a-z0-9']+", prompt.lower().replace('’', "'"))
    return file_count == 0 and len(words) <= 6 and all(w in VAGUE for w in words)

def judge_upgrade(decision, recommendation):
    """Judge output is untrusted and cannot select a provider or change permissions."""
    tier = recommendation.get('tier')
    if tier not in TIERS:
        raise ValueError('Invalid judge tier.')
    if TIERS.index(tier) > TIERS.index(decision.tier):
        return Decision(tier, 'high' if tier == 'deep' else 'medium', decision.risk,
                        decision.reasons + ['AI assessment raised the quality tier.'])
    return decision
