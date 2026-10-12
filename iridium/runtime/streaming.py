"""Realtime token streaming by cutting the control core back mid-decode.

Two modes, and the honest line between them is the whole point of this
module:

* **``mode="lossless"``** -- self-speculative decoding (LayerSkip, Elhoushi et
  al. 2024, arXiv:2404.16710). A handful of tokens are drafted cheaply using
  only the control core's first ``min_layers`` layers and an
  :class:`~iridium.model.exits.ExitHead`, skipping the router, every
  superstack, and the rest of the core. All drafted tokens are then checked
  in a *single* full-depth forward (the "verify" pass), and the longest
  prefix on which the full-depth model's own greedy choice agrees with the
  draft is kept; the first disagreement is replaced by the full-depth model's
  actual choice. Under ``temperature=0`` this produces the identical token
  sequence full-depth greedy decoding would produce -- not approximately: see
  the proof sketch below and ``tests/unit/test_streaming.py::test_lossless_matches_full_depth_greedy``,
  which checks exact ``ids`` equality against
  ``iridium.runtime.generate.generate(..., temperature=0)`` over several
  prompts and seeds on the ``tiny`` config.

* **``mode="budget"``** -- latency-budget streaming (CALM, Schuster et al.
  2022, arXiv:2207.07061). Each token exits the control core at the
  shallowest layer whose confidence clears ``confidence``, or when a
  wall-clock deadline derived from ``target_tokens_per_s`` would otherwise be
  missed, whichever comes first. This is genuinely lossy: layers past the
  exit never see this token's real activation, only the copied exit state
  (CALM's "state propagation": a skipped layer's K/V for this position are
  computed by feeding the *exit* hidden state through that layer's own K/V
  projections, standing in for the K/V a full pass would have produced), so
  every later token that attends back to a shallow-exited position attends
  to an approximation of it, not the true full-depth key and value. There is
  no exactness claim for this mode; ``docs/streaming.md`` quantifies what the
  approximation costs.

Why self-speculative decoding is exact
---------------------------------------
Write the accepted context as ``c_1..c_L`` (already in the KV cache) and let
``h_L`` be the full-depth hidden state produced by the last accepted step
(the model already computed this; nothing is redone). Greedy full-depth
decoding from here is, by definition, "repeatedly take ``argmax`` of the
full-depth head applied to the current hidden state, append it, run the
model on it, repeat". Self-speculative decoding drafts ``d_1..d_k`` cheaply
and then, in a *single* forward call over exactly those tokens against the
*real* cache, computes the full-depth hidden state the model would have at
every one of those positions had they actually been decoded -- because a
transformer's hidden state at position ``i`` is a pure function of the
tokens at positions ``<= i`` and the cache holding real full-depth K/V for
everything before position 1, which is exactly what
``tests/integration/test_kv_parity.py`` exists to guarantee equals
teacher-forced, uncached computation. Comparing ``argmax`` of that state to
``d_{i+1}`` is therefore not a heuristic approval check, it is *computing the
same greedy decision full-depth decoding would have made*, and every token
this module ever emits (accepted or corrected) *is* that decision, never the
draft's own guess. The draft only decides how many tokens get bundled into
one verify call; it never decides what gets emitted. Acceptance rate changes
how much compute is saved; it cannot change the output.

Cache correctness (the part that is easy to get subtly wrong)
---------------------------------------------------------------
The verify call appends real, full-depth cache entries for all ``k + 1``
tokens in its chunk (the already-committed token plus ``k`` drafts), exactly
like that many sequential decode steps would. When the accepted prefix is
shorter than that, the trailing entries must be undone before the corrected
token is decoded, or the cache would hold state for tokens that were never
actually emitted -- a stream state no real decoding path could reach, which
every following token would then attend to or read.

Two designs were considered for undoing it:

1. **Per-key positional truncation.** Core-layer and bridge cache entries
   are plain ``[b, ..., seq, ...]`` tensors, sliceable back along the
   sequence axis. But two cache families are *not* sequence-indexed at all:
   superstack cache keys are keyed on a *packed*, member-only sequence
   (``iridium/model/router.py:pack_routed`` gives a slot only to tokens
   actually routed to that stack), and the router's prefix-mean statistic
   (``MacroRouter.prefix_summary``) is a running sum and count, not a
   per-position list -- "drop the last N" does not even parse for either
   one without re-deriving, per stack or from scratch, exactly which packed
   slots or how much of the running sum belongs to the rejected tail.
2. **Snapshot and replay.** Copy the cache dict (a shallow copy is exact:
   every write anywhere in this model is a key *reassignment*,
   ``cache[key] = new_tensor``, never an in-place mutation of a tensor
   already in the cache -- true for core/bridge/superstack/router entries
   alike, confirmed by reading each write site) before the verify call.  On
   a rejection, restore that snapshot and redo *one* ordinary chunked
   forward call over just the accepted prefix. That call is indistinguishable
   from a normal prefill of those tokens -- the same code path
   ``tests/integration/test_kv_parity.py`` already certifies -- so it is
   correct for every cache family at once, including the two that defeated
   option 1, with no per-key knowledge at all.

This module uses (2). The cost is one wasted full-depth pass over the
rejected suffix (inherent to speculative decoding generally: verifying a
draft always costs something even when the draft is wrong) plus, only on a
rejection, redoing the accepted prefix's stage-one/router/bank/stage-two
work a second time -- avoidable in principle with per-key surgery, but not
worth the fragility here given how easy option 1 is to get subtly wrong for
exactly the two families that matter. ``tests/unit/test_streaming.py``
checks the resulting cache against plain decoding's, not just the emitted
token ids, because a wrong cache would not show up in the very next token --
only in one further downstream, which a token-ids-only test would miss.

This restricts the current implementation (see the class docstring) to
``controller_mode=False``, ``n_loops=1``, and batch size 1 -- the
configurations the ``tiny``/``nano`` rungs and
``tests/integration/test_kv_parity.py`` already exercise. Extending it is
listed in the final report as work that needs the model owner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, Sample, collate
from ..model.exits import ExitHead
from .decode import atomic_chunks, slice_batch
from .device import device_of, generator_for
from .generate import _single_token_batch


@dataclass
class StreamEvent:
    """One decided token, emitted the moment it is decided."""

    token: int
    text: str
    modality: str
    exit_layer: int          # how many control-core layers this token's own
                              # forward actually ran (informational; for
                              # lossless mode this is always n_layers, since
                              # every *emitted* token is a full-depth decision
                              # -- see module docstring)
    latency_ms: float
    mode: str                 # "lossless" | "budget"
    accepted: Optional[bool]  # lossless: True (draft matched), False
                               # (corrected), None (bonus token / prefill);
                               # budget: always None
    position: int


def _tokens_batch(
    template: TensorBatch, modality: int, token_ids: Sequence[int], start_position: int,
) -> TensorBatch:
    """A contiguous batch of ``len(token_ids)`` new tokens, batch size 1.

    Generalizes ``generate._single_token_batch`` to more than one token, for
    the verify pass's single chunked forward call. Kept here rather than
    changed in ``generate.py`` (owned by the lead) even though the two
    functions overlap; see the final report.
    """
    device = template.modality.device
    dtype = template.scalars.dtype
    n = len(token_ids)
    view = object.__new__(TensorBatch)
    view.modality = torch.full((1, n), modality, dtype=torch.long, device=device)
    view.discrete = torch.tensor([token_ids], dtype=torch.long, device=device)
    view.positions = torch.arange(start_position, start_position + n, device=device).view(1, n)
    view.rope_positions = view.positions.unsqueeze(-1).expand(1, n, 3).contiguous()
    view.valid = torch.ones(1, n, dtype=torch.bool, device=device)
    view.supervised = torch.ones(1, n, dtype=torch.bool, device=device)
    view.span_id = torch.full((1, n), -1, dtype=torch.long, device=device)
    view.scalars = torch.zeros(1, n, template.scalars.shape[-1], dtype=dtype, device=device)
    view.continuous = {
        k: torch.zeros(1, n, v.shape[-1], dtype=dtype, device=device)
        for k, v in template.continuous.items()
    }
    view.grids = []
    view.meta = template.meta
    return view


def _snapshot_cache(cache: dict) -> dict:
    """A correct full snapshot, cheaply: every write anywhere in this model
    is ``cache[key] = new_tensor`` (reassignment), never an in-place mutation
    of a tensor already sitting in the cache, so a shallow ``dict`` copy
    captures every entry's exact value at this instant -- see the module
    docstring's "cache correctness" section for why this, rather than
    per-key positional truncation, is what undoes a rejected draft suffix.
    """
    return dict(cache)


def _restore_cache(cache: dict, snapshot: dict) -> None:
    cache.clear()
    cache.update(snapshot)


class StreamingSession:
    """Realtime decode session: draft-and-verify (lossless) or budgeted early
    exit (lossy), one token event at a time.

    Scope of the current implementation (see module docstring for why):
    ``model.cfg.controller_mode`` must be ``False``, ``n_loops`` is fixed at
    ``1``, batch size is fixed at 1 (one realtime stream per session, which
    matches the use case), and ``model.cfg.memory_slots`` must be 0. These
    match ``iridium.config.get_config("tiny")`` and ``"nano"``.

    Parameters
    ----------
    model:
        An ``Iridium1`` instance.
    target_tokens_per_s:
        ``mode="budget"`` only. Soft wall-clock deadline per token; if the
        confidence threshold has not been cleared by the time this deadline
        would be missed, the core exits at whatever layer it has reached.
        ``None`` disables the deadline (confidence alone decides).
    confidence:
        ``mode="budget"`` only. Exit as soon as the :class:`ExitHead`'s
        confidence at some layer ``>= min_layers`` clears this threshold.
    min_layers:
        Draft depth (lossless) / earliest allowed exit (budget). Must be
        ``<= model.core.split`` (stage-one only -- see ``exits.py``'s
        docstring on why an exit cannot live inside stage two without a
        model-owner change). Defaults to ``max(1, split // 2)``.
    mode:
        ``"lossless"`` or ``"budget"``.
    draft_len:
        ``mode="lossless"`` only. Tokens drafted per verify round.
    exit_head:
        Optional pre-trained :class:`ExitHead`. If omitted, one is created
        with random initialization -- lossless mode's exactness does not
        depend on this being trained (see module docstring); budget mode's
        usefulness does (see docs/streaming.md).
    """

    def __init__(
        self,
        model,
        target_tokens_per_s: Optional[float] = None,
        confidence: float = 0.9,
        min_layers: Optional[int] = None,
        mode: str = "lossless",
        draft_len: int = 4,
        exit_head: Optional[ExitHead] = None,
        max_new_tokens: int = 64,
        stop_ids: Sequence[int] = (2,),
        text_offset: int = 16,
        seed: int = 0,
    ) -> None:
        if mode not in ("lossless", "budget"):
            raise ValueError("mode must be 'lossless' or 'budget'")
        if model.cfg.controller_mode:
            raise NotImplementedError(
                "StreamingSession does not yet support controller_mode; see "
                "the final report for what iridium1.py would need to expose."
            )
        if model.cfg.memory_slots:
            raise NotImplementedError(
                "StreamingSession does not yet support context_memory's "
                "recurrent state under lossless rollback; see the module docstring."
            )
        if mode == "budget":
            core = model.cfg.core
            unsupported = [
                name for name, bad in (
                    ("layer_pattern other than global", set(core.layer_kinds()) != {"global"}),
                    ("block_kind='parallel'", core.block_kind != "sequential"),
                    ("hyper_streams > 1", core.hyper_streams > 1),
                    ("ngram embeddings", bool(model.cfg.codecs.ngram_table_size)),
                    ("per-layer embeddings", bool(model.cfg.ple_dim)),
                ) if bad
            ]
            if unsupported:
                # Budget mode steps the core's layers by hand and propagates
                # K/V through GQA's own projections for the layers it skips.
                # Under these options that hand-stepping is a *different
                # function* from the model -- it would bypass the residual
                # streams, call projections MLA and DeltaNet do not have, or
                # embed without the n-gram history -- and it would do so
                # silently. Lossless mode has no such restriction: every token
                # it emits is decided by a real full-depth forward.
                raise NotImplementedError(
                    "budget mode does not support " + ", ".join(unsupported)
                    + "; use mode='lossless', which is exact under every option"
                )
        self.model = model
        self.mode = mode
        self.target_tokens_per_s = target_tokens_per_s
        self.confidence = confidence
        split = model.core.split
        self.min_layers = min_layers if min_layers is not None else max(1, split // 2)
        if not (1 <= self.min_layers <= split):
            raise ValueError(f"min_layers must be in [1, {split}]")
        self.draft_len = draft_len
        self.exit_head = exit_head if exit_head is not None else ExitHead(model.cfg.core.d_model)
        self.max_new_tokens = max_new_tokens
        self.stop_ids = tuple(stop_ids)
        self.text_offset = text_offset
        self.seed = seed
        self._interrupted = False
        self._pending_inputs: list[Sample] = []
        self.last_cache: Optional[dict] = None
        # Diagnostics, readable after (or during, between yields) stream().
        self.stats = {"drafted": 0, "accepted": 0, "verify_rounds": 0, "full_depth_forwards": 0}

    # -- realtime controls --------------------------------------------------

    def interrupt(self) -> None:
        """Stop ``stream()`` after the token currently in flight."""
        self._interrupted = True

    def push_input(self, sample: Sample) -> None:
        """Queue new user input to be spliced in between decode rounds.

        The cache stays valid because the queued sample is processed exactly
        like the original prompt was -- an ordinary prefill forward call that
        *extends* the same cache -- never by editing or replaying anything
        already decoded.
        """
        self._pending_inputs.append(sample)

    # -- main loop ------------------------------------------------------

    def stream(self, sample: Sample) -> Iterator[StreamEvent]:
        self._interrupted = False
        model = self.model
        model.eval()
        dims = continuous_dims(model.cfg.codecs)
        device = device_of(model)
        batch = TensorBatch(collate([sample], dims), device=device)
        cache: dict = {}
        self.last_cache = cache  # exposed for tests/inspection; not part of the event stream
        rng = generator_for(device, self.seed)

        hidden = None
        with torch.no_grad():
            for start, end in atomic_chunks(batch, int(batch.modality.shape[1])):
                out = model(slice_batch(batch, start, end), n_loops=1, cache=cache)
                hidden = out.hidden
        assert hidden is not None
        position = int(batch.positions[0, -1]) + 1
        template = batch

        n_new = 0
        while n_new < self.max_new_tokens and not self._interrupted:
            if self._pending_inputs:
                extra = self._pending_inputs.pop(0)
                extra_batch = TensorBatch(collate([extra], dims), device=device)
                # collate() positions a fresh sample from 0; shift it to
                # continue the stream's actual position so RoPE and the
                # cache's history bookkeeping stay contiguous.
                extra_batch.positions = extra_batch.positions + position
                with torch.no_grad():
                    for start, end in atomic_chunks(extra_batch, int(extra_batch.modality.shape[1])):
                        out = model(slice_batch(extra_batch, start, end), n_loops=1, cache=cache)
                        hidden = out.hidden
                position = int(extra_batch.positions[0, -1]) + 1
                continue

            if self.mode == "lossless":
                remaining = self.max_new_tokens - n_new
                if remaining == 1:
                    # A draft-and-verify round always commits at least two
                    # tokens' worth of cache state (the already-decided
                    # token, then a corrected/bonus token to seed the next
                    # round) -- one more than a budget of exactly 1 allows.
                    # Emit just the already-decided token, the same single
                    # forward step plain decoding would take for the last
                    # token it is asked for.
                    events, hidden, position = self._final_token_round(
                        template, hidden, position, cache,
                    )
                else:
                    # Cap this round's draft length so it cannot commit more
                    # tokens to the cache than the remaining budget allows.
                    k = min(self.draft_len, max(0, remaining - 2))
                    events, hidden, position = self._lossless_round(
                        template, hidden, position, cache, rng, k=k,
                    )
            else:
                events, hidden, position = self._budget_round(
                    template, hidden, position, cache, rng,
                )
            for ev in events:
                n_new += 1
                yield ev
                if n_new >= self.max_new_tokens or (
                    ev.modality in ("text", "control") and ev.token in self.stop_ids
                ):
                    return
                if self._interrupted:
                    return

    def _final_token_round(self, template, hidden, position, cache):
        """The already-decided next token, forwarded once. Used only when
        exactly one more token is wanted -- see the call site."""
        model = self.model
        h = hidden[:, -1:]
        token = int(model.codecs.text_head(h)[0, -1].float().argmax(-1))
        step_batch = _single_token_batch(template, MODALITY_INDEX["text"], token, position)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(step_batch, n_loops=1, cache=cache)
        self.stats["full_depth_forwards"] += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ev = self._make_event(token, position, model.cfg.core.n_layers, elapsed_ms, "lossless", accepted=None)
        return [ev], out.hidden, position + 1

    # -- lossless: self-speculative draft + verify -----------------------

    def _lossless_round(self, template, hidden, position, cache, rng, k: Optional[int] = None):
        """Draft, verify, accept/correct -- one round, ``k + 2`` tokens
        emitted at most (the already-committed token, up to ``k`` accepted
        drafts, and one corrected/bonus token).

        ``predicted0`` is the full-depth model's already-decided next token
        (its logits came from the *previous* round's real full-depth hidden
        state); it has not been run through the model yet -- nothing has
        computed *its* cache entry -- so it opens the verify chunk rather
        than being treated as separately "drafted". Only the tokens after it
        are genuinely speculative.
        """
        model = self.model
        if k is None:
            k = self.draft_len
        draft_cache: dict = {}
        for i in range(self.min_layers):
            key = ("core", 0, i)
            if key in cache:
                draft_cache[key] = cache[key]

        h = hidden[:, -1:]
        pre_logits = model.codecs.text_head(h)[0, -1].float()
        predicted0 = int(pre_logits.argmax(-1))

        draft_ids: list[int] = []
        with torch.no_grad():
            cur_token = predicted0
            for j in range(k):
                step_batch = _single_token_batch(template, MODALITY_INDEX["text"], cur_token, position + j)
                dh = model.codecs.embed(step_batch)
                keep = model._stream_keep(step_batch, history=position + j)
                pos_t = step_batch.positions
                for i in range(self.min_layers):
                    dh = model.core.layers[i](dh, pos_t, keep, draft_cache, ("core", 0, i))
                dlogits, _ = self.exit_head(dh, model.codecs.text_head)
                cur_token = int(dlogits[0, -1].float().argmax(-1))
                draft_ids.append(cur_token)
        self.stats["drafted"] += k

        chunk_ids = [predicted0] + draft_ids
        chunk = _tokens_batch(template, MODALITY_INDEX["text"], chunk_ids, position)
        pre_verify_cache = _snapshot_cache(cache)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(chunk, n_loops=1, cache=cache)
        self.stats["full_depth_forwards"] += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        verify_logits = model.codecs.text_head(out.hidden)[0].float()  # [k+1, V]
        predicted = [int(verify_logits[i].argmax(-1)) for i in range(k + 1)]  # predicted[i] follows chunk_ids[i]

        a = 0
        while a < k and predicted[a] == draft_ids[a]:
            a += 1
        self.stats["accepted"] += a
        self.stats["verify_rounds"] += 1

        events: list[StreamEvent] = [
            self._make_event(predicted0, position, model.cfg.core.n_layers, elapsed_ms / (k + 1),
                              "lossless", accepted=None),
        ]
        for i in range(a):
            events.append(self._make_event(
                draft_ids[i], position + 1 + i, self.min_layers, elapsed_ms / (k + 1),
                "lossless", accepted=True,
            ))

        if a < k:
            # Verify's cache now holds state for tokens that were never
            # actually emitted (the rejected suffix). Undo it by restoring
            # the pre-verify snapshot and replaying only the accepted
            # prefix as one ordinary chunked forward -- see the module
            # docstring's "cache correctness" section for why this, not a
            # per-key truncation, is what is done here.
            _restore_cache(cache, pre_verify_cache)
            accepted_ids = chunk_ids[: a + 1]
            replay_batch = _tokens_batch(template, MODALITY_INDEX["text"], accepted_ids, position)
            with torch.no_grad():
                model(replay_batch, n_loops=1, cache=cache)
            self.stats["full_depth_forwards"] += 1
            corrected = predicted[a]
        else:
            corrected = predicted[k]

        next_position = position + 1 + a
        step_batch = _single_token_batch(template, MODALITY_INDEX["text"], corrected, next_position)
        t1 = time.perf_counter()
        with torch.no_grad():
            step_out = model(step_batch, n_loops=1, cache=cache)
        self.stats["full_depth_forwards"] += 1
        elapsed1_ms = (time.perf_counter() - t1) * 1000.0
        events.append(self._make_event(
            corrected, next_position, model.cfg.core.n_layers, elapsed1_ms,
            "lossless", accepted=(False if a < k else None),
        ))
        return events, step_out.hidden, next_position + 1

    # -- budget: CALM-style confidence/deadline exit -----------------------

    def _budget_round(self, template, hidden, position, cache, rng):
        """Decode one token, exiting stage one early per CALM's rule.

        This deliberately does *not* call ``model.forward`` for the
        continuation: ``Iridium1.forward`` always runs its own full
        ``stage_one`` internally and has no hook to hand it a
        precomputed, truncated-depth stage-one output instead (that hook
        is exactly the "optional exit hook... keeping existing behaviour
        bit-identical when off" the ground rules describe for
        ``control_core.py`` -- see the final report for the precise ask).
        Absent that hook, the only way to honor an early exit *and* still
        feed the router/bank/stage-two the exit state as their input is to
        replicate ``Iridium1.forward``'s ``controller_mode=False,
        n_loops=1`` path here, using only the model's public
        ``router``/``bank``/``core`` components. Keep this in sync with
        that path if it changes; ``tests/unit/test_streaming.py`` cross-
        checks the emitted token against a full-depth ``min_layers=n_layers``
        session (which forces no early exit and must reduce to plain
        decoding) as a regression net for drift between the two.
        """
        model = self.model
        deadline_s = (1.0 / self.target_tokens_per_s) if self.target_tokens_per_s else None
        t0 = time.perf_counter()
        h = hidden[:, -1:]
        pre_logits = model.codecs.text_head(h)[0, -1].float()
        token = int(pre_logits.argmax(-1))

        step_batch = _single_token_batch(template, MODALITY_INDEX["text"], token, position)
        with torch.no_grad():
            dh = model.codecs.embed(step_batch)
            keep = model._stream_keep(step_batch, history=position)
            pos_t = step_batch.positions
            exit_layer = model.core.split
            exit_state = dh
            for i in range(model.core.split):
                dh = model.core.layers[i](dh, pos_t, keep, cache, ("core", 0, i))
                exit_state = dh
                if i + 1 < self.min_layers:
                    continue
                _, conf = self.exit_head(dh, model.codecs.text_head)
                over_budget = (
                    deadline_s is not None
                    and (time.perf_counter() - t0) > deadline_s
                )
                if float(conf[0, -1]) >= self.confidence or over_budget:
                    exit_layer = i + 1
                    break
            # CALM state propagation: any stage-one layer this token skipped
            # still needs a KV entry so later tokens can attend to it. Feed
            # the exit state through those layers' own K/V projections
            # (never through the residual/FFN path) -- an approximation, not
            # a computation of what those layers would have produced; see
            # docs/streaming.md for what this costs.
            for i in range(exit_layer, model.core.split):
                layer = model.core.layers[i]
                k, v = layer.attn.project_kv(layer.norm_attn(exit_state), pos_t)
                key = ("core", 0, i)
                prev = cache.get(key)
                if prev is not None:
                    k = torch.cat([prev[0], k], dim=2)
                    v = torch.cat([prev[1], v], dim=2)
                cache[key] = (k, v)
            h1 = exit_state

            core_states_cache = cache.get(("bridge", "states"))
            core_pos_cache = cache.get(("bridge", "positions"))
            bridge_states = h1 if core_states_cache is None else torch.cat([core_states_cache, h1], dim=1)
            bridge_positions = pos_t if core_pos_cache is None else torch.cat([core_pos_cache, pos_t], dim=1)
            cache[("bridge", "states")] = bridge_states
            cache[("bridge", "positions")] = bridge_positions

            summary = model.router.prefix_summary(h1, 0, cache)
            decision = model.router(
                h1, loop_index=0, span_id=None, summary=summary, valid=step_batch.valid,
            )
            stack_out, _ = model.bank(
                h1, pos_t, decision, bridge_states, bridge_positions, step_batch.grids,
                cache, (0,), False, 0.9, False,
            )
            h2 = model.core.stage_two(h1 + stack_out, pos_t, keep, 0, cache)
            hidden_final = model.core.finalize(h2)
            cache[("stream", "n")] = cache.get(("stream", "n"), 0) + 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ev = self._make_event(token, position, exit_layer, elapsed_ms, "budget", accepted=None)
        return [ev], hidden_final, position + 1

    def _make_event(self, token, position, exit_layer, latency_ms, mode, accepted) -> StreamEvent:
        modality = "text"
        text = ""
        v = token - self.text_offset
        if 0 <= v < 256:
            text = bytes([v]).decode("utf-8", errors="replace")
        return StreamEvent(
            token=token, text=text, modality=modality, exit_layer=exit_layer,
            latency_ms=latency_ms, mode=mode, accepted=accepted, position=position,
        )
