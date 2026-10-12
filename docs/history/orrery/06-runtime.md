# 06 — Runtime: One Resident Instance

> Weights that never unload, streams that never restart, and a firewall down the middle of the
> attention matrix.

## 6.1 What "one always-running instance" already means

Half of constraint C5 is a solved engineering problem that can simply be adopted. Frontier serving
already keeps weights resident and multiplexes many requests through one instance:

- **Continuous batching** admits and retires requests at each decode step rather than at batch
  boundaries — the single largest throughput unlock in modern serving.
- **Paged KV** eliminates fragmentation so concurrent streams share the memory pool without
  reservation waste.
- **Chunked prefill** lets a long incoming context interleave with ongoing decodes instead of
  stalling them.
- **Ragged batching** means one batch simultaneously contains prefills and decodes at different
  positions.

Nothing here needs inventing. "One instance instead of spinning up a model per request" is what
production inference does. The interesting part of C5 is not the residency; it is what the model is
allowed to do *with* residency.

## 6.2 The part that must not be built

C5 also implies attention flowing between concurrent chats — one focus moving across all live
requests.

**This is a cross-tenant data leak, not a capability.** If stream A's query can attend to stream B's
KV cache, stream A can read stream B's conversation. There is no version of this that is safe and
also does what it sounds like it does.

The enforcement is structural, per [`02`](02-trunk.md) §2.5: a **block-diagonal stream mask** with
`−∞` on every off-diagonal block. Streams share the batch, the kernel, and every parameter; they
cannot see each other because the mask makes those entries unreachable, not because a policy asks
them not to look. Cross-stream isolation should be a property of the arithmetic.

### What legitimately is global

| Shared | Not shared |
|---|---|
| The weights | Any stream's KV cache |
| Attention sinks | Any stream's context |
| The scheduler and the depth-budget pool | Any stream's intermediate states |
| Operator-owned resident memory (see §6.4) | Anything a user did not authorize |

That list preserves everything the "one mind" intuition actually wants — shared knowledge, shared
capacity, continuity across time — and drops only the part that was a vulnerability.

## 6.3 The compute broker: how focus is actually allocated

This is where C3 becomes an operational system rather than an architectural claim.

```
model  →  emits per-stream difficulty estimate and requested depth budget
scheduler  →  reconciles all requests against available capacity, returns an allocation
trunk  →  spends recursion steps within the allocation, per-token (see [02])
```

**The model proposes; the scheduler disposes.** This split is not bureaucratic caution — without it,
a single stream that judges itself important starves every other one, and the model has no visibility
into global load. The scheduler owns fairness; the model owns difficulty estimation. Each is
qualified for exactly one of those jobs.

Allocation inputs: requested depth, stream priority class, tokens already spent, latency target, and
KV residency pressure. A dual-heap arrangement — one ordering compute admission, one ordering cache
retention — prevents the failure where a high-priority stream is admitted promptly and then evicted
from cache, which is a priority inversion dressed as a scheduling success.

**Deep pondering must not stall shallow streams.** This is the hardest scheduling problem in the
design and it should not be waved away. A stream recursing 30× through a pipeline-parallel stage
occupies that stage; a chat stream behind it waits. Mitigations: bound per-stream recursion within a
scheduling quantum and requeue rather than running to completion; interleave micro-batches from
different streams within each recursion pass; and admit deep-ponder work against a reserved capacity
fraction so it cannot consume the interactive pool. Variable depth plus pipeline parallelism is a
genuinely unsolved scheduling problem at scale, and [`10`](10-open-problems.md) carries it as such.

## 6.4 Persistence and duplex

**Streaming stability.** Non-evicting attention sinks anchor the softmax for an instance that never
restarts ([`02`](02-trunk.md) §2.5). Beyond the active window, older context is compressed into
summary latents rather than dropped, so continuity degrades gracefully instead of falling off a
cliff.

**Two-tier memory.** Latency-critical interaction runs in the foreground; consolidation, indexing,
and historical recall run asynchronously in the background and hand results forward. This is the
pattern streaming-memory systems converge on, and it is what lets a resident instance accumulate
useful state without paying for it in response time.

**Full duplex.** Input and output share one temporal axis: the model perceives while generating.
Consequences that matter more than they sound:

- It can be interrupted mid-sentence and incorporate the interruption, because the interruption
  arrives as latents on the same timeline.
- It can act unprompted — a monitored simulation diverging, a long solve completing, a watched file
  changing — because "nothing has been said recently" is not the same as "nothing is happening."
- Audio and video generation stay aligned by construction rather than by post-hoc synchronization.

**Long solves are asynchronous.** A Tier B run ([`03`](03-physics.md)) takes minutes to hours. It
executes outside the forward pass; the stream stays live, the model reports progress, and the solved
field arrives as new latents when it is ready. The instance is never "busy" — only differently
weighted.

## 6.5 Failure and operational posture

- **A stream is not a session.** Streams are cheap and disposable; the resident memory is what
  persists, and it is owned by an operator or user, not by the connection.
- **Backpressure is explicit.** Under load the scheduler reduces depth allocations before it drops
  streams, and the model is told its budget was cut. A model that silently thinks less should still
  say it thought less — a shallow answer to a hard question is a reportable event.
- **Weight updates are gated.** Weights change at deliberate, evaluated deployment boundaries with
  rollback, never as a live side effect of a self-improvement loop. See
  [`09`](09-self-improvement.md).
- **Every stream is isolated by default and shares only by explicit grant.** The default for anything
  not on the "global" list in §6.2 is not-shared.
