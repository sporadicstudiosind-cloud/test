# What was built, and how to run it

A working implementation of the routed control-core architecture, at a scale
that trains on a laptop CPU, using the *same* modules a 9-trillion-parameter
instantiation would use. Scale is a config file, not a rewrite.

## The shape of it

```
            text  image  video  audio  fields  geometry  actions
                            |
                    codecs  ->  one d_model space
                            |
        +-------------------v--------------------+
        |  CONTROL CORE stage I  (layers 0..L/2)  |   owns the causal KV cache
        +-------------------+--------------------+
                            |  h
                    +-------v--------+
                    | MACRO ROUTER   |  top-k stacks, focus, ponder halt
                    +-------+--------+   (all decisions causal)
          +-----------------+------------------+
          v                 v                  v
    +-----------+     +-----------+      +-----------+
    | SUPERSTACK|     | SUPERSTACK| ...  | SUPERSTACK|   N deep domain banks
    |  sparse   |     |  + FNO    |      |           |   - stack-local sparse KV
    |  local KV |     |  blocks   |      |           |   - bridge -> core KV
    |  + bridge |     |           |      |           |   - PonderNet depth
    +-----+-----+     +-----+-----+      +-----+-----+
          +-----------------+------------------+
                            |  weighted residual
        +-------------------v--------------------+
        |  CONTROL CORE stage II (layers L/2..L)  |
        +-------------------+--------------------+
                            |            ^
                    ponder loop ---------+  (chunk-uniform, cache-indexed)
                            |
       heads: text | flow-matching continuous | action | slot-type | confidence
                            |
              dual-system physics verifier -> exact solver on failure
```

## The ladder

| rung | params | active/token | GFLOP/token | MXFP4 weights |
|---|---|---|---|---|
| tiny | 1.5 M | 0.40 M | 0.003 | 0.81 MB |
| nano | 34.0 M | 18.7 M | 0.07 | 18.1 MB |
| micro | 995.8 M | 324.5 M | 1.95 | 529 MB |
| small | 25.8 B | 4.2 B | 25.3 | 13.7 GB |
| **base** | **9.04 T** | **692 B** | **4154** | **4.80 TB** |
| extreme | 39.5 T | 1.63 T | 9774 | 21.0 TB |

`python -m iridium ladder` prints it. `python -m iridium report nano --verify`
instantiates the model and shows the formula and the modules agree exactly.

## Run it

```bash
python -m iridium ladder                       # the scaling ladder
python -m iridium report nano --verify         # parameter accounting, checked
python -m iridium plan base --gpus 1024        # 4-D parallelism + cost model
python -m iridium plan base --training --accelerator h200   # returns infeasible
python -m iridium quant base                   # MXFP4 memory and measured SQNR
python -m iridium fluid                        # Taylor-Green validation
python -m iridium waterfall --q 3 --factor 2   # the originating question
python -m iridium serve --prompt "hello" --prompt "second stream"
pytest -q                                      # 375 tests
```

Training phases, each runnable:

```bash
python -m iridium.training.phase1_pretrain --rung nano --steps 1800
python -m iridium.training.phase2_specialize --init runs/phase1/phase1-final.pt
python -m iridium.training.phase3_rlvr      --init runs/phase1/phase1-final.pt
python -m iridium.training.phase4_agentic   --init runs/phase1/phase1-final.pt
```

## What is established, and what is not

Established, each by a test that fails if it stops being true:

- **Cache parity.** Cached incremental decoding reproduces the uncached forward
  pass — bit-exact at one ponder loop, within 4 ULP of float64 at two. This is
  the hardest property in the design: the sparse stack-local KV, the bridge and
  the loop index could each break it silently.
- **Parameter accounting.** The formulae that cost a 9 T configuration match
  the real torch modules exactly at the small rungs.
- **Physics.** Taylor-Green reproduced to 6e-15 relative; divergence held at
  1e-14; open-channel flow converges to the analytic normal depth from both
  sides and independently recovers Manning's law.
- **Isolation.** A stream's output is bit-identical whether or not other
  streams exist; no tensor storage is reachable from two streams.
- **Span coherence.** Per-token routing keeps 0% of field grids intact;
  span-coherent routing keeps 100%.

Not established, and not claimed:

- No capability claim at any scale above `nano`. The large rungs are *costed*,
  not built, and the cost model is a model.
- `nano` is 34 M parameters trained on a synthetic corpus. It demonstrates that
  the architecture trains and that the mechanisms work; it demonstrates nothing
  about what a 9 T version would know.
- Per-token ponder granularity, copy-through loop history, and a real
  distributed dispatcher are **specified, not implemented**.
- The sandbox has no seccomp filter and no filesystem namespace. It says so, in
  the result object, every time.

See [`verification.md`](verification.md) for the twenty findings against the
source plan, and [`capability-register.md`](capability-register.md) for
per-capability status.
