"""What a run can and cannot produce, computed before it starts.

This module exists because of a specific, expensive failure mode: a training
run completes, the loss curve looks reasonable, the checkpoint loads, and the
model babbles — and the conclusion drawn is that the *architecture* is wrong.
Nearly always it is not. It is that the run was never given enough data, or
enough of the right *kind* of data, for the result to have been anything else,
and nothing in the loop said so.

Two questions are worth answering up front, because both have arithmetic
answers and neither is visible in a loss curve.

**How much data does this parameter count need?**  The Chinchilla result
(Hoffmann et al., 2022) put the compute-optimal ratio near 20 tokens per
parameter. That is the *floor*, and it is a floor for a specific question —
"given a fixed compute budget, how should I split it between size and data" —
which is not the question anyone asks about a small model. When the model must
be small for deployment reasons, training far past Chinchilla is not wasteful,
it is the entire technique: the small open models that are actually coherent
sit between roughly 1,000 and 12,000 tokens per parameter, which is two to
three orders of magnitude past compute-optimal. Both reference points are
reported below, because a run that misses the Chinchilla floor by 100x is not
"slightly undertrained", it is a different kind of object.

**Does the mixture contain the thing being asked for?**  A model trained
entirely on synthetic structured families will do those families and produce
nothing resembling prose, no matter how large it is or how long it runs. This
is obvious when stated and very easy to ship: the mixture lives in one module,
the complaint arrives from another, and the connection between them is a number
nobody computed. :func:`audit` computes it.

Nothing here trains, loads, or allocates anything. It is arithmetic over a
config and a mixture, so it can run before a job is submitted — which is the
only time the answer is still useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

__all__ = ["BudgetReport", "audit", "tokens_for", "REFERENCE_RATIOS"]


#: Tokens per parameter, for context. ``chinchilla`` is the compute-optimal
#: ratio; the others are what small models that are actually usable were
#: trained at, and are the relevant comparison when the size is fixed by
#: deployment rather than chosen to be optimal. These are approximate published
#: figures for orientation, not measurements made here.
REFERENCE_RATIOS: dict[str, float] = {
    "chinchilla_optimal": 20.0,
    "small_model_practice_low": 1_000.0,
    "small_model_practice_high": 12_000.0,
}

#: Families in ``training.datasets.DEFAULT_MIXTURE`` and elsewhere that consist
#: of natural language a language model could learn prose from. Everything not
#: listed is a synthetic structured family: valuable for what it teaches, and
#: no substitute for text when the complaint is "it does not talk properly".
NATURAL_LANGUAGE_FAMILIES = frozenset({"text_lm", "chat"})


@dataclass(frozen=True)
class BudgetReport:
    rung: str
    parameters: int
    #: Distinct tokens the corpus holds, before any repetition.
    corpus_tokens: int
    #: Tokens the optimizer will consume: steps x batch x sequence length.
    consumed_tokens: int
    epochs: float
    tokens_per_parameter: float
    natural_language_fraction: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shortfall(self) -> dict[str, float]:
        """How many times more data each reference ratio would want."""
        return {
            name: (self.parameters * ratio) / max(self.corpus_tokens, 1)
            for name, ratio in REFERENCE_RATIOS.items()
        }

    def describe(self) -> str:
        lines = [
            f"{self.rung}: {self.parameters:,} parameters",
            f"  corpus            {self.corpus_tokens:,} distinct tokens",
            f"  consumed          {self.consumed_tokens:,} tokens "
            f"({self.epochs:.1f} epochs over the corpus)",
            f"  tokens/parameter  {self.tokens_per_parameter:.3f}",
            f"  natural language  {self.natural_language_fraction:.1%} of the mixture",
            "  data shortfall against published ratios:",
        ]
        for name, factor in self.shortfall.items():
            lines.append(f"    {name:<26} {factor:>12,.0f}x more data wanted")
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def tokens_for(n_items: int, window: int) -> int:
    """Distinct tokens in a corpus of ``n_items`` windows of ``window`` tokens.

    An upper bound, and deliberately so: it assumes every window is full and
    that no two windows overlap. Unpacked random-window sampling violates the
    second assumption routinely — two windows drawn from the same document can
    overlap or repeat — so the real figure is at or below this one. A bound that
    errs optimistic is the right shape here, because the conclusion being drawn
    is "even at best, this is not enough".
    """
    if n_items < 0 or window < 1:
        raise ValueError("item count must be nonnegative and window positive")
    return n_items * window


def audit(
    cfg,
    n_items: int,
    steps: int,
    batch_size: int,
    window: int = 256,
    accumulate: int = 1,
    mixture: Optional[Mapping[str, float]] = None,
    sequence_length: Optional[int] = None,
) -> BudgetReport:
    """Cost a planned run against its parameter count and its mixture.

    ``cfg`` is an :class:`~iridium.config.IridiumConfig`. ``mixture`` is the
    family-weight mapping the corpus will be built from; pass the one actually
    being used, not the default, since the default is exactly what this is for.

    ``sequence_length`` defaults to ``window``: the number of tokens each item
    contributes to a training step. It is separate from ``window`` because a
    packed corpus can slice longer training sequences out of the same bytes.
    """
    if min(n_items, steps, batch_size, accumulate) < 1:
        raise ValueError("items, steps, batch size and accumulation must be positive")
    length = sequence_length or window
    corpus = tokens_for(n_items, window)
    consumed = steps * batch_size * accumulate * length
    parameters = int(cfg.n_params)

    weights = dict(mixture) if mixture else {}
    total_weight = sum(v for v in weights.values() if v > 0)
    natural = (
        sum(v for k, v in weights.items() if k in NATURAL_LANGUAGE_FAMILIES and v > 0)
        / total_weight
        if total_weight > 0
        else 0.0
    )

    warnings: list[str] = []
    if weights and natural == 0.0:
        warnings.append(
            "the mixture contains no natural-language family (text_lm, chat), so "
            "this run cannot produce a model that writes prose at any size or "
            "duration — add text_lm/chat weight before concluding anything about "
            "the architecture from the model's writing"
        )
    elif 0.0 < natural < 0.2:
        warnings.append(
            f"natural language is {natural:.1%} of the mixture; prose quality is "
            "bounded by that share, not by the parameter count"
        )

    floor = REFERENCE_RATIOS["chinchilla_optimal"]
    ratio = corpus / max(parameters, 1)
    epochs = consumed / max(corpus, 1)
    if ratio < floor:
        # Which failure it is depends on repetition. Seen once, too little data
        # leaves a model undertrained; seen many times, it gets memorised.
        consequence = (
            "and with repeated passes it will memorise rather than generalise"
            if epochs > 1.5 else
            "so it will be undertrained -- more tokens, not more passes, is the fix"
        )
        warnings.append(
            f"{ratio:.3f} tokens per parameter against a compute-optimal floor of "
            f"{floor:.0f}; {consequence}"
        )

    if epochs > 4:
        warnings.append(
            f"{epochs:.1f} passes over the same tokens; past roughly 4 epochs "
            "repeated data stops adding signal (Muennighoff et al., 2023) and the "
            "run is buying memorisation with compute"
        )
    if consumed < corpus:
        warnings.append(
            f"only {consumed / corpus:.1%} of the corpus will be seen; the extra "
            "items are being streamed and paid for without being used"
        )

    return BudgetReport(
        rung=cfg.name,
        parameters=parameters,
        corpus_tokens=corpus,
        consumed_tokens=consumed,
        epochs=epochs,
        tokens_per_parameter=ratio,
        natural_language_fraction=natural,
        warnings=tuple(warnings),
    )


if __name__ == "__main__":                              # pragma: no cover
    import argparse

    from ..config import get_config
    from .datasets import DEFAULT_MIXTURE

    ap = argparse.ArgumentParser(description="Cost a planned training run.")
    ap.add_argument("--rung", default="nano")
    ap.add_argument("--items", type=int, default=12_000)
    ap.add_argument("--steps", type=int, default=1_800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--accumulate", type=int, default=1)
    args = ap.parse_args()

    print(audit(
        get_config(args.rung), args.items, args.steps, args.batch_size,
        window=args.window, accumulate=args.accumulate, mixture=DEFAULT_MIXTURE,
    ).describe())
