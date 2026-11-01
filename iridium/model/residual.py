"""Multi-stream residual connections: Hyper-Connections and mHC.

A plain pre-norm residual stack has exactly one stream: ``x_{l+1} = x_l +
f_l(x_l)``. Hyper-Connections (Zhu et al., ByteDance, "Hyper-Connections",
arXiv:2409.19606, ICLR 2025) widen that to ``n`` parallel streams and replace
the fixed ``+`` with three *learned, per-layer* maps: a stream-mixing matrix
that lets streams exchange information laterally, a "pre" map that decides
what mixture of streams feeds the sublayer, and a "post" map that decides how
the sublayer's output is scattered back across streams. The paper reports
this fixes the "seesaw" between gradient vanishing (residual too weak) and
representation collapse (residual too strong) that a single stream cannot
escape, because different streams can specialise to different amplification
regimes.

The catch, which is why this file does not stop at plain Hyper-Connections:
an unconstrained stream-mixing matrix is a free real ``n x n`` matrix applied
once per layer, so its spectral radius compounds multiplicatively with depth
exactly the way an un-normalised RNN's transition matrix does. DeepSeek's
mHC paper ("mHC: Manifold-Constrained Hyper-Connections", arXiv:2512.24880,
adopted in DeepSeek-V4 with 4 streams) measured this directly: at 27B
parameters, unconstrained Hyper-Connections reached a signal amplification
of ~3000x by depth and diverged around step 12,000. mHC's fix is to project
the mixing matrix onto the Birkhoff polytope -- the set of doubly-stochastic
matrices (every row and column sums to 1) -- via Sinkhorn-Knopp iteration
before applying it. A doubly-stochastic matrix is a convex combination of
permutation matrices (Birkhoff-von Neumann), so it can reshuffle and blend
streams but cannot scale the total signal: mHC reports the same setup then
bounded at ~1.6x. ``constrained=True`` on :class:`HyperConnections` is that
projection; ``constrained=False`` is plain, unconstrained Hyper-Connections,
kept so the difference is something this file can measure rather than assert
(see ``tests/unit/test_residual.py::test_signal_gain_table``).

Memory cost, stated once here rather than at every call site: ``n`` streams
means the residual stream itself is held ``n`` times over for the duration of
:class:`HyperStack`'s forward (expand to collapse). That is ``n`` times the
activation memory of *only* the residual tensor, not of the sublayers' own
internal activations (attention scores, FFN hidden state, ...), which are
computed once per layer exactly as before. DeepSeek-V4 uses ``n=4``.

Integration contract
---------------------
Nothing outside this file, and nothing the caller passes in, ever sees a
stream tensor. :func:`expand_streams` turns an ordinary ``[B, T, d]`` tensor
into ``[B, T, n, d]``; :class:`StreamCollapse` turns it back. A layer wrapped
in :class:`HyperConnections` still looks, from outside, like a function
``[B, T, d] -> [B, T, d]`` (via :meth:`HyperConnections.forward`, or as a
whole run through :class:`HyperStack`). That is what lets
``ControlCore.run_layers`` adopt this by wrapping its layer loop, without the
router, the bridge, the KV cache or the superstacks needing to know streams
exist -- they are a detail of what happens *between* the layers a core layer
range already iterates, not a change to the tensor that crosses that
boundary. See the module docstring's end for the exact wiring recipe.

Wrapping an existing residual block (e.g. ``layers.TransformerBlock``, which
adds its own two residuals internally) rather than a bare delta function:
pass ``delta=False`` to :meth:`HyperConnections.forward`. The wrapper then
computes ``sublayer(x) - x`` as the quantity Hyper-Connections scatters back
across streams, so at ``n_streams=1`` with identity maps the composition is
*exactly* ``sublayer(x)`` -- the block's own two internal residuals are left
alone, and only the block's net effect is what Hyper-Connections replaces
the *outer* residual with. This needs no change to ``layers.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinkhorn_knopp(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Project ``logits`` (``[..., n, n]``) onto the Birkhoff polytope.

    Alternately row- and column-normalises ``exp(logits)`` in log space
    (``logsumexp`` rather than ``exp`` then divide) so this stays finite for
    the large negative logits the identity initialisation below uses --
    ``exp(-40)`` would round to a denormal in the naive version and does not
    need to, since the log-domain update never actually forms it until the
    final ``.exp()``. ``iters`` rounds is a fixed, differentiable computation
    (no data-dependent stopping rule), which is what makes this usable inside
    a forward pass rather than only as a one-off preprocessing step.

    Convergence is geometric in ``iters`` for a strictly positive matrix
    (Sinkhorn 1964); it is not exact after finitely many rounds, only close,
    which is why the exactness tests below use a tolerance rather than 0.0
    except at ``n=1``, where every row/column normalisation of a 1x1 matrix
    is exact after a single iteration regardless of the input.
    """
    log_alpha = logits
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


def expand_streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """``[B, T, d] -> [B, T, n, d]`` by replication.

    Parameter-free and identical across all ``n`` streams at every call
    (not just at init): the papers describe the expansion as itself
    learnable, but a replicated expansion is already exact at ``n=1`` and,
    at ``n>1``, gives every stream the same starting point, so which stream
    becomes "the fast one" and which becomes "the slow one" is left entirely
    to training rather than pre-decided by an expansion prior this codebase
    has no evidence for. A learned expansion is a straightforward later
    addition (an ``[n, 1]`` map like :class:`StreamCollapse`'s, transposed)
    if training shows streams need to start differentiated.
    """
    return x.unsqueeze(2).expand(-1, -1, n_streams, -1)


class _DynamicVector(nn.Module):
    """Shared machinery for a per-layer ``[n]``-shaped map (pre-alpha, post-beta).

    ``static`` is the learned base value, ``dynamic`` adds a per-token,
    per-layer correction from a cheap linear read of the current stream
    state (mean over streams -- the same summary :class:`HyperConnections`
    already computes, so this adds no extra communication pattern). The
    dynamic projection is zero-initialised, which is what makes every
    ``HyperConnections`` map -- not just the static part -- start at the
    plain-residual value: training grows the dynamic term from exactly zero
    rather than from an arbitrary random correction stacked on top of a
    correct static init.

    ``zero_mean`` subtracts the dynamic delta's own mean before adding it,
    which makes ``sum(vector) == sum(static)`` an *exact* invariant at every
    step, not just at init -- used for the pre-map ``alpha``, where
    ``sum(alpha) == 1`` is what makes "the sublayer's input is some affine
    combination of the streams" a stable property of the layer rather than
    one that only holds before the first gradient step. The post-map
    ``beta`` has no such constraint in the literature (it is a broadcast
    weight, not expected to describe a combination), so it is left
    unconstrained beyond the zero-init.
    """

    def __init__(self, d: int, n: int, static_init: torch.Tensor,
                 dynamic: bool, zero_mean: bool) -> None:
        super().__init__()
        self.n = n
        self.zero_mean = zero_mean
        self.static = nn.Parameter(static_init.clone())
        self.dynamic = dynamic
        if dynamic:
            self.proj = nn.Linear(d, n)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        b, t, _ = summary.shape
        v = self.static.view(1, 1, self.n).expand(b, t, self.n)
        if self.dynamic:
            delta = self.proj(summary)
            if self.zero_mean:
                delta = delta - delta.mean(dim=-1, keepdim=True)
            v = v + delta
        return v

    @staticmethod
    def param_count(d: int, n: int, dynamic: bool) -> int:
        static = n
        dyn = (d * n + n) if dynamic else 0
        return static + dyn


class StreamMix(nn.Module):
    """The per-layer ``[n, n]`` stream-mixing map -- the thing mHC constrains.

    ``constrained=False`` (plain Hyper-Connections): the matrix is used as
    learned, exactly as parameterised, with no normalisation. Static part
    initialised to ``I_n`` (an exact, not approximate, doubly-stochastic
    matrix, so this and the constrained path start identically); the dynamic
    part is a zero-initialised per-token correction. This is the path whose
    spectral radius is unconstrained during training -- see the module
    docstring for the 3000x figure the mHC paper measured from exactly this
    failure mode.

    ``constrained=True`` (mHC): the same static + dynamic logits are instead
    passed through :func:`sinkhorn_knopp` before use, projecting them onto
    the Birkhoff polytope. The static logits are initialised with a large
    diagonal gap (``0`` on the diagonal, ``-40`` off it) so that, before any
    dynamic correction, Sinkhorn-Knopp converges to (numerically
    indistinguishable from) the identity matrix -- exactly the same starting
    matrix as the unconstrained path, just reached by projection instead of
    direct parameterisation.
    """

    def __init__(self, d: int, n: int, constrained: bool,
                 sinkhorn_iters: int = 20, dynamic: bool = True) -> None:
        super().__init__()
        self.n = n
        self.constrained = constrained
        self.sinkhorn_iters = sinkhorn_iters
        self.dynamic = dynamic
        if constrained:
            init = torch.full((n, n), -40.0)
            init.fill_diagonal_(0.0)
        else:
            init = torch.eye(n)
        self.logits = nn.Parameter(init)
        if dynamic:
            self.proj = nn.Linear(d, n * n)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        b, t, _ = summary.shape
        m = self.logits.view(1, 1, self.n, self.n).expand(b, t, self.n, self.n)
        if self.dynamic:
            m = m + self.proj(summary).view(b, t, self.n, self.n)
        if self.constrained:
            m = sinkhorn_knopp(m, self.sinkhorn_iters)
        return m

    @staticmethod
    def param_count(d: int, n: int, dynamic: bool) -> int:
        static = n * n
        dyn = (d * n * n + n * n) if dynamic else 0
        return static + dyn


class StreamCollapse(nn.Module):
    """The run-ending ``[n] -> [B, T, d]`` reduction, once per :class:`HyperStack`.

    Same static + zero-mean-dynamic construction as the pre-map (see
    :class:`_DynamicVector`), with the static part initialised to
    ``1/n`` each so collapsing ``n`` identical streams (as
    :func:`expand_streams` produces and, at init, every layer preserves)
    reproduces the input exactly, matching a single collapse of ``n``
    copies of ``x`` back to ``x``.
    """

    def __init__(self, d: int, n: int, dynamic: bool = True) -> None:
        super().__init__()
        self.n = n
        self.vector = _DynamicVector(
            d, n, torch.full((n,), 1.0 / n), dynamic, zero_mean=True
        )

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        summary = streams.mean(dim=2)
        w = self.vector(summary)
        return torch.einsum("btn,btnd->btd", w, streams)

    @staticmethod
    def param_count(d: int, n: int, dynamic: bool) -> int:
        return _DynamicVector.param_count(d, n, dynamic)


class HyperConnections(nn.Module):
    """One layer's worth of Hyper-Connections / mHC, wrapping one sublayer call.

    ``forward(streams, sublayer, *args, delta=True, **kwargs)`` runs, per
    the papers:

    1. mix streams laterally: ``h = mix(summary) @ streams``
    2. read the sublayer's input as a combination of the mixed streams:
       ``x_in = alpha(summary) . h``
    3. run the sublayer: ``y = sublayer(x_in, *args, **kwargs)`` (or, if
       ``delta=False``, ``y = sublayer(x_in, ...) - x_in`` -- for wrapping a
       sublayer that already adds its own residual internally, such as
       ``layers.TransformerBlock``)
    4. scatter the result back across streams:
       ``streams' = h + beta(summary) (outer) y``

    where ``summary = streams.mean(dim=2)`` is the same cheap per-token read
    all three maps condition on. At ``n_streams=1`` every one of ``mix``,
    ``alpha`` and ``beta`` is a 1x1 or length-1 map that is exactly the
    identity/one regardless of training (see :class:`StreamMix` and
    :class:`_DynamicVector`), so step 4 reduces to ``streams' = x + y`` --
    the plain residual update -- exactly, not approximately. See
    ``test_reduces_to_plain_residual_at_n1`` for the 0.0-tolerance check.
    """

    def __init__(self, d: int, n_streams: int, constrained: bool,
                 sinkhorn_iters: int = 20, dynamic: bool = True) -> None:
        super().__init__()
        self.n = n_streams
        self.mix = StreamMix(d, n_streams, constrained, sinkhorn_iters, dynamic)
        self.pre = _DynamicVector(
            d, n_streams, torch.full((n_streams,), 1.0 / n_streams),
            dynamic, zero_mean=True,
        )
        self.post = _DynamicVector(
            d, n_streams, torch.ones(n_streams), dynamic, zero_mean=False,
        )

    def forward(self, streams: torch.Tensor, sublayer, *args,
                delta: bool = True, **kwargs) -> torch.Tensor:
        summary = streams.mean(dim=2)
        mix = self.mix(summary)
        h = torch.einsum("btij,btjd->btid", mix, streams)
        alpha = self.pre(summary)
        x_in = torch.einsum("btn,btnd->btd", alpha, h)
        y = sublayer(x_in, *args, **kwargs)
        if not delta:
            y = y - x_in
        beta = self.post(summary)
        return h + torch.einsum("btn,btd->btnd", beta, y)

    @staticmethod
    def param_count(d: int, n_streams: int, dynamic: bool = True) -> int:
        return (
            StreamMix.param_count(d, n_streams, dynamic)
            + 2 * _DynamicVector.param_count(d, n_streams, dynamic)
        )


class HyperStack(nn.Module):
    """Expand -> (per-layer HyperConnections around each sublayer) -> collapse.

    This is the ``[B, T, d] -> [B, T, d]`` object ``ControlCore.run_layers``
    can wrap its layer loop with. ``sublayers`` is any list of callables
    accepting ``(x, *args, **kwargs) -> tensor``; each gets its own
    :class:`HyperConnections` instance (independent learned maps per layer,
    as in the papers -- streams are shared state, but how a layer reads and
    writes them is layer-specific). Pass ``delta=False`` at call time (see
    :class:`HyperConnections`) if the sublayers are full residual blocks
    rather than bare deltas.
    """

    def __init__(self, sublayers, d: int, n_streams: int, constrained: bool = True,
                 sinkhorn_iters: int = 20, dynamic: bool = True) -> None:
        super().__init__()
        self.n = n_streams
        self.sublayers = nn.ModuleList(sublayers)
        self.connections = nn.ModuleList(
            HyperConnections(d, n_streams, constrained, sinkhorn_iters, dynamic)
            for _ in sublayers
        )
        self.collapse = StreamCollapse(d, n_streams, dynamic)

    def forward(self, x: torch.Tensor, *args, delta: bool = True, **kwargs) -> torch.Tensor:
        streams = expand_streams(x, self.n)
        for hc, sub in zip(self.connections, self.sublayers):
            streams = hc(streams, sub, *args, delta=delta, **kwargs)
        return self.collapse(streams)

    @staticmethod
    def param_count(d: int, n_streams: int, n_layers: int, dynamic: bool = True) -> int:
        return (
            n_layers * HyperConnections.param_count(d, n_streams, dynamic)
            + StreamCollapse.param_count(d, n_streams, dynamic)
        )
