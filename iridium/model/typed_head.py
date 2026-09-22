"""A typed, calibrated, non-autoregressive decision head.

This is the well-understood mechanism that a "System-1" model such as Jev
(TypeSafe AI, announced 2026-09-15) is one instance of: TypeSafe has not
published Jev's architecture, so nothing here reproduces it. What is public
about Jev, and independently verified for this module by web search before
writing it (TypeSafe's own launch post, MarkTechPost's coverage, and the
DataCamp/DEV Community explainers, all dated 2026-09-15 through 2026-09-19),
is the *shape* of the mechanism: given a block of context, it returns typed
values with calibrated confidence, produced by evaluating every requested
field in one parallel forward pass rather than one token at a time, and it is
trained so that a stated confidence tracks empirical accuracy (TypeSafe calls
their training method Reinforcement Learning for Calibrated Decisions, RLCD).
Those are three separable, individually well-established ideas, and this
module builds each honestly rather than pretending to have reverse-engineered
a closed model:

1. **Schema-constrained typed output.** Standard structured-decoding practice
   (grammar-constrained decoding, JSON-schema-constrained sampling) applies a
   mask so the *support* of the output distribution is exactly the valid
   values for the declared type — invalid values have zero probability
   because they are not in the softmax at all, not merely down-weighted.
   ``TypedHead`` does this per field, chosen by a ``TypeSchema``.
2. **Calibration.** A raw softmax max is not a calibrated confidence — a
   network trained with cross-entropy is well known to drift overconfident
   (Guo et al., 2017, "On Calibration of Modern Neural Networks"), and the obvious
   wrong fix is to just trust argmax probability as "confidence" without
   checking it against held-out accuracy, which is what most systems that
   report "confidence: 0.94" are actually doing. ``calibrate_temperature``
   fits a single scalar per field by grid search on held-out (logits, label)
   pairs (temperature scaling: divide logits by T before softmax; it cannot
   change which class is predicted, only how peaked the distribution is, so
   it cannot hurt accuracy and is the standard first thing to try before
   anything fancier like isotonic regression or Platt scaling per class).
   ``expected_calibration_error`` and ``brier_score`` are the two standard
   ways to check it worked, and this module tests that they actually improve
   on a synthetic model built to be miscalibrated on purpose.
3. **Parallel emission with an explicit independence assumption.** All fields
   of a struct come off one shared trunk in one forward pass — no field's
   computation depends on another field's *sampled* value. That is a real
   modelling assumption (conditional independence of the fields given the
   hidden state) and it is wrong whenever two fields are correlated beyond
   what the trunk already encodes (e.g. a ``(min, max)`` pair where the joint
   constraint ``min <= max`` is not itself part of either field's marginal
   type). The honest fixes are either (a) don't use a struct for fields whose
   *joint* distribution matters more than their marginals — decode them as
   one combined enum/int over the joint support instead, or (b) turn on
   ``dependency=True``, which adds one round of message-passing between the
   fields' pre-output embeddings (still a single forward pass, not
   autoregressive) so each field's logits can react to a summary of the
   others'. That captures soft correlation, not hard joint constraints; it
   does not make ``min <= max`` structurally guaranteed the way the type mask
   makes an out-of-range int structurally impossible, and this module does
   not claim otherwise.

What Jev is *not*, and what this module does not attempt: a published
architecture to copy, a specific parameter count or latency figure to match,
or a claim of "cannot hallucinate" — the type mask genuinely forecloses
representing an invalid value, but a *wrong, valid* value (a bool that is
confidently False when the answer is True) is still possible and is exactly
what calibration is for: knowing the confidence attached to it means
something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm

# ---------------------------------------------------------------------------
# Type schema: closed, flat, fixed-shape. Deliberately not a general type
# system -- "a small fixed-shape struct" per the brief, not arbitrary nesting.
# Nesting a StructType inside a StructType is rejected in __init__ rather than
# silently doing something under-tested with it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoolType:
    """Two-valued. Modelled as a 2-way categorical rather than a single
    sigmoid logit so it shares the same softmax/temperature/ECE machinery as
    every other categorical field instead of needing a separate code path."""


@dataclass(frozen=True)
class EnumType:
    """A closed set of string labels. The supplied set *is* the vocabulary --
    there is no "other" bucket, so a value outside it is unrepresentable by
    construction, not filtered after the fact."""

    values: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("EnumType needs at least two values")
        if len(set(self.values)) != len(self.values):
            raise ValueError("EnumType values must be unique")


@dataclass(frozen=True)
class IntType:
    """An inclusive integer range ``[lo, hi]``, modelled as a categorical over
    the ``hi - lo + 1`` representable integers -- the same reason as
    ``EnumType``: the output layer's width *is* the valid range, so there is
    no integer this head can emit that a caller then has to range-check."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError("IntType requires hi > lo")
        if self.hi - self.lo > 4096:
            raise ValueError(
                "IntType range too wide for a categorical head; bucket or "
                "rescale, or use RealType and round"
            )


@dataclass(frozen=True)
class RealType:
    """A real number, optionally bounded. Unlike the categorical types, an
    unconstrained real cannot be made unrepresentable-when-invalid by masking
    a finite output space -- there is no finite mask for the reals. Instead,
    boundedness is made a *structural* property of the forward pass: a
    bounded ``RealType`` is produced by squashing an unconstrained linear
    output through a sigmoid into ``[lo, hi]``, so a bug in training can
    produce a bad value but never an out-of-range one; that is the closest
    analogue available to "invalid unrepresentable" for a continuous type."""

    lo: Optional[float] = None
    hi: Optional[float] = None

    def __post_init__(self) -> None:
        if self.lo is not None and self.hi is not None and self.hi <= self.lo:
            raise ValueError("RealType requires hi > lo when both are given")


FieldType = Union[BoolType, EnumType, IntType, RealType]


@dataclass(frozen=True)
class StructType:
    """A fixed, named set of fields, each an independent (or lightly coupled;
    see ``dependency``) field type. Order is preserved from the mapping."""

    fields: Mapping[str, FieldType]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("StructType needs at least one field")
        for name, t in self.fields.items():
            if isinstance(t, StructType):
                raise ValueError(
                    f"field {name!r}: nested StructType is not supported -- "
                    "flatten the schema; this head is for small fixed-shape "
                    "structs, not arbitrary trees"
                )


TypeSchema = Union[FieldType, StructType]


# ---------------------------------------------------------------------------
# Calibration metrics. Both take already-computed (post-hoc) confidences, not
# raw model internals, so they apply equally to a temperature-scaled softmax
# max or to anything else a caller wants to check.
# ---------------------------------------------------------------------------


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean squared error between a predicted categorical distribution and
    the one-hot true label -- proper (strictly minimized in expectation only
    by the true distribution), unlike accuracy or raw log-loss magnitude,
    which is why it is reported alongside ECE rather than instead of it."""
    onehot = F.one_hot(labels, probs.shape[-1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(-1).mean().item()


def expected_calibration_error(
    confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 10
) -> float:
    """Bin predictions by confidence; compare each bin's mean confidence to
    its actual accuracy; return the bin-size-weighted mean gap.

    This is the standard ECE (Naeini et al., 2015; Guo et al., 2017), and its
    known blind spot is worth stating: a model that is overconfident on half
    its bins and underconfident on the other half by the same amount scores
    the same ECE as one that is perfectly calibrated everywhere, because the
    errors are computed with signs stripped and then averaged. It is a
    screening metric, not a proof of per-bin calibration; that is why the
    test in ``tests/unit/test_typed_head.py`` also checks Brier score, which
    does not have that particular blind spot.
    """
    confidence = confidence.detach().float()
    correct = correct.detach().float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    n = confidence.shape[0]
    ece = torch.zeros(())
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        if not bool(mask.any()):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece = ece + (mask.float().sum() / n) * (bin_acc - bin_conf).abs()
    return float(ece)


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grid: Optional[Sequence[float]] = None,
) -> float:
    """Fit a scalar temperature by grid search on held-out (logits, labels).

    Grid search rather than the textbook LBFGS fit (Guo et al., 2017, fit T
    by minimizing NLL with LBFGS) is a deliberate simplification: this is a
    convex 1-D problem in log(T) for cross-entropy loss, LBFGS on it is
    solving a harder problem than the problem requires, and a ~40-point log
    grid over the range temperatures are ever usefully set to (0.05x-20x)
    finds a minimum indistinguishable from the LBFGS solution at a fraction
    of the iterations -- which matters because this is meant to run as a
    cheap post-hoc calibration step, not a second training run. Returns the
    grid point minimizing held-out NLL.
    """
    if grid is None:
        grid = [0.05 * (1.2 ** i) for i in range(40)]  # ~0.05 .. ~20
    labels = labels.long()
    best_t, best_nll = 1.0, float("inf")
    with torch.no_grad():
        for t in grid:
            nll = F.cross_entropy(logits / t, labels).item()
            if nll < best_nll:
                best_nll, best_t = nll, t
    return best_t


# ---------------------------------------------------------------------------
# The head.
# ---------------------------------------------------------------------------


@dataclass
class FieldPrediction:
    """One field's decoded output, batched (leading dim = batch)."""

    value: Any                 # decoded python-typed values, one per batch row (bool/str/int/float)
    confidence: torch.Tensor   # (batch,) calibrated in [0, 1]
    raw: torch.Tensor          # (batch, k) logits (categorical) or (batch, 2) [mean, std] (real)


@dataclass
class TypedPrediction:
    """The full output of one ``TypedHead`` forward pass, batched."""

    fields: Dict[str, FieldPrediction]
    abstain_prob: torch.Tensor  # (batch,)
    abstained: torch.Tensor     # (batch,) bool


def _field_width(t: FieldType) -> int:
    if isinstance(t, BoolType):
        return 2
    if isinstance(t, EnumType):
        return len(t.values)
    if isinstance(t, IntType):
        return t.hi - t.lo + 1
    if isinstance(t, RealType):
        return 2  # [mean_raw, logvar]
    raise TypeError(f"unknown field type: {t!r}")


def _decode_field(t: FieldType, raw: torch.Tensor) -> Any:
    """``raw`` is this field's (batch, k) network output. Returns a python
    value per batch row -- the categorical types by argmax (the MAP value;
    ``FieldPrediction.raw`` carries the full distribution for callers who
    want to sample or inspect it instead), the real type by its predicted
    mean."""
    raw = raw.detach()
    if isinstance(t, BoolType):
        idx = raw.argmax(-1)
        return [bool(v) for v in idx.tolist()]
    if isinstance(t, EnumType):
        idx = raw.argmax(-1)
        return [t.values[i] for i in idx.tolist()]
    if isinstance(t, IntType):
        idx = raw.argmax(-1)
        return [t.lo + i for i in idx.tolist()]
    if isinstance(t, RealType):
        mean_raw = raw[..., 0]
        if t.lo is not None and t.hi is not None:
            mean = t.lo + (t.hi - t.lo) * torch.sigmoid(mean_raw)
        elif t.lo is not None:
            mean = t.lo + F.softplus(mean_raw)
        elif t.hi is not None:
            mean = t.hi - F.softplus(mean_raw)
        else:
            mean = mean_raw
        return mean.tolist()
    raise TypeError(f"unknown field type: {t!r}")


class TypedHead(nn.Module):
    """Emits a schema-constrained typed value (or a small struct of them) in
    a single forward pass, with calibrated confidence and an abstention
    output. See the module docstring for what this is and is not modelled on.

    Router integration contract (the "clean interface" the brief asks for):

        head = TypedHead(d_model, schema)
        pred = head(h)                    # h: (..., d_model), any leading dims
        pred.fields["answer"].value       # python-typed values, batched
        pred.fields["answer"].confidence  # (...,) in [0, 1], calibrated if
                                           # calibrate_temperature() was run
        pred.abstained                    # (...,) bool -- True means "do not
                                           # use this field's value; fall back"

    A router wires this in as an alternative to the autoregressive path
    (``iridium/runtime/decode.py`` / ``iridium/runtime/generate.py``) for any
    slot whose answer is known ahead of time to be one of these five shapes:
    call ``TypedHead(d_model, schema)`` once per distinct schema the router
    needs (schema is fixed at construction, like ``ActionHead``'s op/scalar
    counts -- it is not part of the input, because the output layer's width
    depends on it), run it on the same pooled hidden state that would
    otherwise have seeded the first token of an autoregressive answer, and
    branch on ``pred.abstained``: if False, take ``pred.fields[...].value``
    directly with no decode loop, no detokenization and no sampling
    temperature to tune; if True, fall back to the existing autoregressive
    path. This is the same shape of integration ``ConfidenceHead`` already
    has in this file (a cheap side head consulted for a routing decision),
    generalized from "is this token probably right" to "here is the typed
    answer and here is how much to trust it."
    """

    def __init__(
        self,
        d_model: int,
        schema: TypeSchema,
        d_hidden: int = 0,
        *,
        dependency: bool = False,
        abstain_init_logit: float = -2.0,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.fields: Dict[str, FieldType] = (
            dict(schema.fields) if isinstance(schema, StructType) else {"value": schema}
        )
        self.dependency = dependency
        d_hidden = d_hidden or max(d_model, 32)
        self.d_hidden = d_hidden

        self.norm = RMSNorm(d_model)
        self.trunk = nn.Linear(d_model, d_hidden)

        # One small per-field branch off the shared trunk -- this is the
        # "parallel emission" itself: every field's Linear here is applied to
        # the same ``trunk`` activation in the same forward call, so there is
        # exactly one pass through the network no matter how many fields the
        # struct has, versus the ``len(fields)`` sequential decode steps an
        # autoregressive struct-as-tokens approach would need.
        self.branches = nn.ModuleDict({
            name: nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden))
            for name in self.fields
        })
        self.field_out = nn.ModuleDict({
            name: nn.Linear(d_hidden, _field_width(t))
            for name, t in self.fields.items()
        })

        if dependency and len(self.fields) > 1:
            # One round of message passing: every field's branch embedding
            # contributes to a shared mean-pooled context, and that context
            # is mixed back into every field's branch embedding before the
            # final linear layer reads it off. This is still one forward
            # pass -- no field's *value* is sampled and fed to another field
            # -- so it captures soft correlation between fields' logits, not
            # a hard joint constraint between their sampled values. See the
            # module docstring for what this does and does not fix.
            self.context_mix = nn.Linear(d_hidden, d_hidden)
            nn.init.zeros_(self.context_mix.weight)
            nn.init.zeros_(self.context_mix.bias)

        # Calibration is fit post-hoc (see calibrate_temperature); 1.0 means
        # "uncalibrated, trust the raw softmax" and is the honest default
        # until calibrate() has actually been called with held-out data.
        self.temperatures: Dict[str, float] = {name: 1.0 for name in self.fields}

        # The SiLU is not decoration. "Should I answer?" is rarely monotone in
        # any feature: the typical shape is a *band* -- abstain near a decision
        # boundary, answer on either side of it. Without a nonlinearity between
        # the trunk and this readout the abstain logit is a linear function of
        # the (normalised) input, and a linear function cannot carve out a band;
        # it can only say "abstain more as this feature goes up". The field
        # branches have always had one; this path was the only one without.
        self.abstain_head = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, 1))
        nn.init.zeros_(self.abstain_head[1].weight)
        nn.init.constant_(self.abstain_head[1].bias, abstain_init_logit)

    def field_logits(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw (uncalibrated) per-field logits/params, before decoding to
        python values. Exposed separately from ``forward`` so
        ``calibrate_temperature`` can be fit against exactly what the head
        produces, on a held-out batch, without re-running the trunk twice."""
        return self._field_logits_from(self.trunk(self.norm(h)))

    def _field_logits_from(self, base: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Split out so ``forward`` can evaluate the trunk exactly once and feed
        # the same features to every field *and* to the abstention head. The
        # previous version re-ran the trunk for abstention, doubling the cost of
        # a head whose entire reason to exist is being the one-pass, cheap path.
        branch_embed = {name: self.branches[name](base) for name in self.fields}
        if self.dependency and len(self.fields) > 1:
            pooled = torch.stack(list(branch_embed.values()), dim=0).mean(0)
            context = self.context_mix(pooled)
            branch_embed = {name: e + context for name, e in branch_embed.items()}
        return {name: self.field_out[name](branch_embed[name]) for name in self.fields}

    def calibrate(
        self,
        held_out_h: torch.Tensor,
        held_out_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Fit and store one temperature per categorical field from a
        held-out batch of (hidden states, true label indices). ``RealType``
        fields have no categorical temperature to fit and are skipped -- their
        calibration story is the predicted variance, not a softmax
        temperature; see the module docstring's caveat about that."""
        with torch.no_grad():
            logits = self.field_logits(held_out_h)
        fitted = {}
        for name, t in self.fields.items():
            if isinstance(t, RealType) or name not in held_out_labels:
                continue
            temp = calibrate_temperature(logits[name], held_out_labels[name])
            self.temperatures[name] = temp
            fitted[name] = temp
        return fitted

    def forward(self, h: torch.Tensor, abstain_threshold: float = 0.5) -> TypedPrediction:
        base = self.trunk(self.norm(h))
        raw = self._field_logits_from(base)
        fields: Dict[str, FieldPrediction] = {}
        for name, t in self.fields.items():
            r = raw[name]
            if isinstance(t, RealType):
                mean_raw, logvar = r[..., 0], r[..., 1]
                std = F.softplus(logvar) + 1e-3
                if t.lo is not None and t.hi is not None:
                    scale = t.hi - t.lo
                elif t.lo is None and t.hi is None:
                    scale = std.detach().mean().clamp_min(1e-3) * 4  # self-referential fallback scale
                else:
                    scale = std.detach().mean().clamp_min(1e-3) * 4
                # A real-valued "confidence" is not a probability over a
                # finite support the way categorical confidence is; this
                # maps predictive std, relative to the field's range (or, for
                # an unbounded field, to its own current typical scale), onto
                # (0, 1] with 1.0 at zero uncertainty and asymptoting to 0 as
                # std grows past the range -- a monotone, bounded, but
                # heuristic proxy, not a calibrated probability. It is not
                # run through ``calibrate()``, which is why ``calibrate``
                # skips ``RealType`` fields rather than silently pretending
                # to fit a temperature for it.
                confidence = torch.clamp(1.0 - std / scale, min=0.0, max=1.0)
                value = _decode_field(t, r)
            else:
                scaled = r / self.temperatures[name]
                probs = F.softmax(scaled, dim=-1)
                confidence = probs.max(-1).values
                value = _decode_field(t, r)
            fields[name] = FieldPrediction(value=value, confidence=confidence.detach(), raw=r)

        abstain_prob = torch.sigmoid(self.abstain_head(base)).squeeze(-1)
        abstained = abstain_prob > abstain_threshold
        return TypedPrediction(fields=fields, abstain_prob=abstain_prob, abstained=abstained)

    def decide(self, h: torch.Tensor, abstain_threshold: float = 0.5) -> Any:
        """Single-item convenience wrapper for router integration: takes one
        hidden vector (no leading batch dim), returns a plain python value
        (or ``{name: value}`` for a struct schema) plus a scalar confidence,
        or ``None`` if the head abstained. This is the call a router makes in
        place of an autoregressive decode loop for a slot whose type is
        known; batched callers should use ``forward`` directly and read
        ``TypedPrediction.abstained`` per row instead of calling this in a
        python loop.
        """
        pred = self.forward(h.unsqueeze(0), abstain_threshold=abstain_threshold)
        if bool(pred.abstained[0]):
            return None
        if isinstance(self.schema, StructType):
            return {
                name: (fp.value[0], float(fp.confidence[0]))
                for name, fp in pred.fields.items()
            }
        fp = pred.fields["value"]
        return fp.value[0], float(fp.confidence[0])

    def loss(
        self,
        h: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        abstain_label: Optional[torch.Tensor] = None,
        abstain_weight: float = 0.2,
    ) -> torch.Tensor:
        """Supervised training loss: cross-entropy per categorical field,
        Gaussian NLL per real field, plus binary cross-entropy for
        abstention against a realized-correctness signal (mirrors
        ``ConfidenceHead`` in ``heads.py``: abstention is trained against
        whether the head was actually right, not a self-reported feeling).
        ``abstain_label`` is optional because a caller doing a first pass of
        training the value heads has no correctness signal yet; omit it and
        only the field losses are used.
        """
        raw = self.field_logits(h)
        total = None
        for name, t in self.fields.items():
            if name not in labels:
                continue
            if isinstance(t, RealType):
                mean_raw, logvar = raw[name][..., 0], raw[name][..., 1]
                std = F.softplus(logvar) + 1e-3
                target = labels[name].to(mean_raw.dtype)
                nll = 0.5 * ((target - mean_raw) / std) ** 2 + torch.log(std)
                term = nll.mean()
            else:
                term = F.cross_entropy(
                    raw[name].reshape(-1, raw[name].shape[-1]),
                    labels[name].reshape(-1).long(),
                )
            total = term if total is None else total + term
        if abstain_label is not None:
            base = self.trunk(self.norm(h))
            logit = self.abstain_head(base).squeeze(-1)
            abstain_term = F.binary_cross_entropy_with_logits(
                logit, abstain_label.to(logit.dtype)
            )
            total = abstain_term * abstain_weight if total is None else total + abstain_weight * abstain_term
        if total is None:
            raise ValueError("no labels matched this schema's fields")
        return total
