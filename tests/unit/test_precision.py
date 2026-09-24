"""An fp64 model must stay in fp64.

This file exists because of a failure that was invisible on the machine it
was written on. CI ran the same commit on two runners; one passed, the other
lost every cache-parity test at once, with discrepancies of exactly 2**-22
and 2**-23 — fp32 ULPs — inside models the tests build in fp64. The fp64
physics tests held on both.

The cause was six-plus `.float()` calls in the forward path. Widening
fp16/bf16 before a reduction is correct; `.float()` spells that as "make
this fp32", which for an fp64 caller is a downcast that leaves the result
still *typed* fp64 with fp32 rounding inside it.

What makes it nasty is the failure mode. A downcast does not by itself make
cached decoding disagree with the uncached forward — it sets a ceiling on
how closely they can agree. Two fp64 values a natural 1e-15 apart usually
round to the *same* fp32 value, and the comparison then looks not merely
close but bit-identical. Occasionally they straddle a rounding boundary,
round to different fp32 values, and the difference jumps to ~1e-7. Which
way they land depends on reduction order, and so on the machine. The gate
was therefore a lottery that happened to be winning locally, and its
headline claim — bit-exactness — was an artifact of discarding the
precision that would have distinguished the two paths.

So rather than assert the *consequence* on whichever hardware CI hands us,
assert the *cause* is absent: run the model in fp64 and fail if any tensor
is narrowed anywhere under it. That is machine-independent, it names the
file and line, and it fails on the commit that introduces the downcast
rather than on a later one that happens to draw an unlucky runner.
"""
from __future__ import annotations

import collections
import dataclasses
import traceback

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.model.layers import use_attention_backend
from iridium.runtime.decode import run_chunked

#: Calls that can narrow a tensor's floating-point type.
NARROWING = {"float", "half", "bfloat16", "to", "type", "type_as"}


class CatchFp64Downcast(TorchFunctionMode):
    """Record every fp64 tensor narrowed to a smaller float, with its source.

    Intercepting at the torch-function level rather than hooking modules is
    what makes this complete: the sites that actually caused the failure were
    mid-expression (`self.gate(hn).float()`, `positions.to(torch.float32)`),
    never a module's own output, so an output-dtype hook walks straight past
    them — as one did, reporting a clean model, while the downcasts were
    there all along.
    """

    def __init__(self) -> None:
        self.hits: collections.Counter[str] = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if getattr(func, "__name__", "") in NARROWING and args and torch.is_tensor(args[0]):
            source = args[0]
            if (source.dtype == torch.float64 and torch.is_tensor(out)
                    and out.is_floating_point() and out.dtype != torch.float64):
                blame = self._blame()
                if blame is not None:
                    self.hits[blame] += 1
        return out

    @staticmethod
    def _blame() -> str | None:
        """The innermost frame inside ``iridium/``, or None.

        Narrowing that happens entirely within torch is not this project's to
        fix and is not actionable, so it is not recorded. Only code this repo
        owns can be blamed for it.
        """
        for frame in reversed(traceback.extract_stack()[:-2]):
            if "/iridium/" in frame.filename:
                where = "iridium/" + frame.filename.split("/iridium/", 1)[1]
                return f"{where}:{frame.lineno}  {(frame.line or '').strip()}"
        return None


def _batch(cfg, seed: int = 6, n: int = 2) -> TensorBatch:
    rng = np.random.default_rng(seed)
    samples = [
        Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 11)))])
        for _ in range(n)
    ]
    return TensorBatch(collate(samples, continuous_dims(cfg.codecs)), dtype=torch.float64)


# The variants that failed on the unlucky runner, which between them cover
# the core layers, the router, the rotary table and the routed stacks.
CONFIGS = {
    "default": {},
    "local_global": dict(layer_pattern=("local", "global"), local_window=3),
    "mla": dict(layer_pattern=("mla",), mla_kv_rank=24, mla_rope_dim=8),
    "deltanet_hybrid": dict(layer_pattern=("deltanet", "deltanet", "deltanet", "global")),
    "dyt_parallel_dynamic": dict(norm_kind="dyt", block_kind="parallel", ffn_dynamic_rank=4),
    "everything": dict(layer_pattern=("deltanet", "mla", "local", "global"),
                       local_window=3, mla_kv_rank=24, mla_rope_dim=8,
                       norm_kind="dyt", block_kind="parallel", ffn_dynamic_rank=4,
                       hyper_streams=4),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_an_fp64_model_never_narrows_a_tensor(name):
    tiny = get_config("tiny")
    cfg = dataclasses.replace(tiny, core=dataclasses.replace(tiny.core, **CONFIGS[name]))
    torch.manual_seed(0)
    model = Iridium1(cfg).to(torch.float64).eval()
    batch = _batch(cfg)

    guard = CatchFp64Downcast()
    # Both paths, and two ponder loops, so the router and the rotary table
    # are exercised the way the parity gate exercises them.
    with torch.no_grad(), use_attention_backend("manual"), guard:
        model(batch, n_loops=2)
        run_chunked(model, batch, chunk=1, n_loops=2)

    assert not guard.hits, (
        "fp64 was narrowed inside an fp64 model — every site below caps how "
        "closely cached decoding can match the uncached forward, and makes "
        "whether it does machine-dependent:\n  "
        + "\n  ".join(f"x{n} {site}" for site, n in guard.hits.most_common())
    )


def test_the_guard_catches_the_downcast_it_was_written_for(monkeypatch):
    """A guard that cannot fail proves nothing, so put the real bug back.

    Not a synthetic narrowing in this file — `at_least_fp32` *is* the fix, so
    reverting it to the `.float()` it replaced reproduces the original defect
    through the real model and shows the guard names the site.
    """
    import iridium.model.layers as layers

    monkeypatch.setattr(layers, "at_least_fp32", lambda x: x.float())

    tiny = get_config("tiny")
    torch.manual_seed(0)
    model = Iridium1(tiny).to(torch.float64).eval()
    batch = _batch(tiny)

    guard = CatchFp64Downcast()
    with torch.no_grad(), use_attention_backend("manual"), guard:
        model(batch, n_loops=1)

    assert guard.hits, "the guard failed to see a downcast that is definitely there"
    assert any("layers.py" in site for site in guard.hits), guard.hits


def test_narrowing_inside_torch_is_not_blamed_on_this_repo():
    """Only code this repo owns is actionable, so only it is reported."""
    guard = CatchFp64Downcast()
    with guard:
        torch.zeros(4, dtype=torch.float64).float()
    assert not guard.hits, guard.hits


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_at_least_fp32_widens_only_the_narrow_types(dtype):
    """The fix must be a no-op in every dtype anyone actually trains in.

    `.float()` did two jobs at once: widen fp16/bf16 so a reduction does not
    lose the sum, and — unintentionally — narrow fp64. `at_least_fp32` keeps
    the first and drops the second, so fp32 and bf16 models compute exactly
    what they computed before. Verified directly at the time of the change:
    an fp32 and a bf16 forward through the tiny model hashed identically
    before and after it, bit for bit.
    """
    from iridium.model.layers import at_least_fp32

    x = torch.zeros(4, dtype=dtype)
    out = at_least_fp32(x)
    if dtype in (torch.float32, torch.float64):
        assert out.dtype == dtype, "must not touch a type that is already wide enough"
        assert out is x, "and should not copy it either"
    else:
        assert out.dtype == torch.float32, "half types must widen for reductions"
