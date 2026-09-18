"""Trusted search space and promotion rules. Model proposals cannot change these."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ResearchPolicy:
    rounds: int = 2
    candidates: int = 3
    train_steps: int = 100
    micro_batch: int = 1
    accumulate: int = 8
    max_items: int = 2000
    eval_items: int = 128
    min_family_items: int = 16
    max_parameters: int = 150_000_000
    max_wall_seconds: int = 14400
    job_seconds: int = 3600
    min_relative_gain: float = 0.01
    max_family_regression: float = 0.01
    max_text_accuracy_drop: float = 0.0
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    seed: int = 1729

    def __post_init__(self):
        bounds = {'rounds': (1, 10), 'candidates': (1, 8), 'train_steps': (1, 10000),
                  'micro_batch': (1, 16), 'accumulate': (1, 128), 'max_items': (16, 100000),
                  'eval_items': (16, 10000), 'min_family_items': (2, 10000),
                  'max_parameters': (1, 10_000_000_000), 'max_wall_seconds': (1, 86400),
                  'job_seconds': (1, 86400), 'bootstrap_samples': (100, 10000)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'invalid policy {key}')
        for key in ('min_relative_gain', 'max_family_regression', 'max_text_accuracy_drop'):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid policy {key}')
        if not .5 < self.confidence < 1:
            raise ValueError('confidence must lie in (.5,1)')

    def to_dict(self):
        return asdict(self)


def validate_proposal(value, families, max_loops):
    if not isinstance(value, dict):
        raise ValueError('proposal must be an object')
    permitted = {'hypothesis', 'lr', 'weight_decay', 'loss_weights', 'mixture',
                 'qk_norm', 'gated_bank', 'loop_identity', 'n_loops', 'code_patch'}
    if set(value) - permitted:
        raise ValueError(f'unknown proposal fields: {sorted(set(value)-permitted)}')
    out = dict(value)
    for key, default, low, high in [('lr', 1e-4, 1e-6, 1e-3), ('weight_decay', .01, 0, .2)]:
        number = out.get(key, default)
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f'{key} out of approved range')
        out[key] = float(number)
    for key in ('qk_norm', 'gated_bank', 'loop_identity'):
        if key in out and type(out[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if 'n_loops' in out and (type(out['n_loops']) is not int or not 1 <= out['n_loops'] <= min(3, max_loops)):
        raise ValueError('invalid loop budget')
    weights = out.get('loss_weights', {})
    allowed_losses = {'text', 'image', 'audio', 'video', 'field', 'geometry',
                      'action_op', 'action_scalar', 'slot_type', 'quantity'}
    if not isinstance(weights, dict) or set(weights) - allowed_losses:
        raise ValueError('unknown loss weight; regularizer definitions are protected')
    mixture = out.get('mixture', {})
    if not isinstance(mixture, dict) or set(mixture) - set(families):
        raise ValueError('mixture may only name approved training families')
    for table in (weights, mixture):
        if any(type(v) not in (int, float) or not math.isfinite(v) or not .1 <= v <= 4 for v in table.values()):
            raise ValueError('loss/mixture weights must lie in [.1,4]')
    for key in ('hypothesis', 'code_patch'):
        if key in out and (not isinstance(out[key], str) or len(out[key]) > 16000):
            raise ValueError(f'invalid {key}')
    return out


def judge(parent, candidate, policy):
    """Paired bootstrap lower bound, per-family regression gates, text accuracy.

    Lower loss is better. Normalize per item by incumbent loss to prevent an
    inherently large numerical modality loss from dominating the objective.
    Bonferroni accounts for the predeclared maximum candidate/round budget.
    """
    required_checks = {'finite_hidden', 'parameter_accounting', 'causal_prefix', 'cached_decoding'}
    for report in (parent, candidate):
        if any(report.get('checks', {}).get(name) is not True for name in required_checks):
            return {'passed': False, 'reason': 'missing/failed fixed architecture checks'}
    old, new = parent['examples'], candidate['examples']
    if len(old) != len(new) or not old:
        return {'passed': False, 'reason': 'missing evaluation coverage'}
    gains, families = [], {}
    for a, b in zip(old, new):
        if (a['id'], a['family'], sorted(a['losses'])) != (b['id'], b['family'], sorted(b['losses'])):
            return {'passed': False, 'reason': 'evaluation example/modalities changed'}
        terms = []
        for key, baseline in a['losses'].items():
            loss = b['losses'][key]
            if not math.isfinite(loss) or not math.isfinite(baseline) or min(loss, baseline) < 0:
                return {'passed': False, 'reason': 'invalid metric'}
            gain = (baseline - loss) / max(baseline, 1e-6)
            terms.append(gain)
            families.setdefault(a['family'] + ':' + key, []).append(gain)
        gains.append(float(np.mean(terms)))
    for family, values in families.items():
        if len(values) < policy.min_family_items:
            return {'passed': False, 'reason': f'insufficient held-out coverage for {family}'}
        if float(np.mean(values)) < -policy.max_family_regression:
            return {'passed': False, 'reason': f'regression in {family}'}
    if candidate['text_count'] != parent['text_count']:
        return {'passed': False, 'reason': 'text generation coverage changed'}
    if candidate['text_accuracy'] < parent['text_accuracy'] - policy.max_text_accuracy_drop:
        return {'passed': False, 'reason': 'text generation regression'}
    rng = np.random.default_rng(policy.seed)
    gains = np.asarray(gains)
    means = [float(rng.choice(gains, len(gains), replace=True).mean())
             for _ in range(policy.bootstrap_samples)]
    alpha = (1 - policy.confidence) / (policy.rounds * policy.candidates)
    lower = float(np.quantile(means, alpha))
    return {'passed': lower >= policy.min_relative_gain,
            'reason': 'passed' if lower >= policy.min_relative_gain else 'insufficient paired improvement',
            'relative_gain': float(gains.mean()), 'lower_bound': lower,
            'family_gains': {key: float(np.mean(values)) for key, values in families.items()}}
