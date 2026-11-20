"""Fixtures shared across the suite."""
from __future__ import annotations

import pytest

from iridium.model.layers import use_attention_backend


@pytest.fixture
def exact_attention():
    """Pin the manual attention backend for tests that assert *exactness*.

    The cache-parity gate compares a T-token forward against T one-token
    cached steps and demands they agree to 1e-10 in float64. That is a claim
    about the model's algebra — does the cache reproduce the function that
    was trained? — and it is only a test of that claim if the arithmetic underneath
    is the same arithmetic both times.

    ``F.scaled_dot_product_attention`` does not promise that. It dispatches
    to whichever fused kernel the build and the CPU make eligible, and on
    some hosts that kernel computes float64 inputs at float32 internally.
    The two sides of the gate then differ by float32 ULPs — 2**-22 and
    2**-23 turn up verbatim in the failure output — for a reason that has
    nothing to do with the cache. Observed live: every cache-parity test in
    the suite failed on one GitHub runner and passed on another, same
    commit, same tree, while the float64 physics tests (Taylor-Green to
    6e-15) held on both. A gate that is red on half a runner fleet is not
    telling anyone about the cache in either direction.

    So exactness claims run on the manual path, whose arithmetic is written
    out here and is the same for any shape. SDPA is not thereby untested:
    ``tests/unit/test_attention.py`` compares the two backends directly, to
    the float32-scale tolerance that comparison actually warrants, which is
    where a broken fused kernel gets caught.
    """
    with use_attention_backend("manual"):
        yield "manual"
