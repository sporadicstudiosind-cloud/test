# Small Iridium chat: run, train, and interpret

The bundled `serve/weights/nano-phase1-fp16.pt` is a real Iridium checkpoint,
roughly 34 million parameters. It was trained for 800 steps on synthetic phase-1
tasks. It was **not trained on chat turns**, and its recorded task results are
mostly at or below a prompt-ignoring baseline. The commands below give you a
working chat interface and a path to train a chat checkpoint; they do not turn
the bundled weights into a reliable assistant.

## Run the bundled checkpoint

From the repository root, install the dependencies in your Python environment:

```bash
python -m pip install numpy torch pyyaml
python -m iridium.runtime.device
python -m iridium chat --device auto
```

The device check exercises matrix multiplication, FFT, and a device-side random
generator. On an AMD ROCm build of PyTorch, `--device rocm` selects the GPU
through PyTorch's `cuda` API. On NVIDIA, use `--device cuda`. A CPU-only PyTorch
install cannot validate or run the GPU path. See [GPU setup](gpu.md).

Useful chat options:

```bash
python -m iridium chat --device auto --prompt "Hello" --max-new-tokens 64
python -m iridium chat --checkpoint runs/chat-nano/chat-inference.pt --device auto
```

Omit `--prompt` for an interactive session. `/reset` starts a new conversation;
`/exit` quits. The command loads the exact checkpoint architecture from its
manifest and rejects unexplained missing or unexpected weights. The bundled
older checkpoint leaves ten compatible parameters at initialization, including
an untrained quantity codec. The loader reports that gap when it starts; those
weights do not establish quantity or chat ability.

For a local web interface, run `python serve/server.py` and open
`http://localhost:8080`. It keeps prior turns in the browser and sends them to
the same conversation format used by training. Set `IRIDIUM_CHECKPOINT` to a
chat inference checkpoint before starting the server to use your own weights.
The server fails if an explicit checkpoint path does not exist.

## Fine-tune for chat on a GPU

Install the `datasets` package and use a PyTorch build matched to your GPU.
From the repository root:

```bash
python -m pip install datasets
python -m iridium.runtime.device
python -m iridium.training.chat_finetune --device auto --steps 200
python -m iridium chat --checkpoint runs/chat-nano/chat-inference.pt --device auto
```

The fine-tuning command starts from the bundled nano checkpoint. Its defaults
sample human-written Dolly and OASST1 conversations, supervise assistant turns
only, hold conversations out by a content hash, and save assistant-token loss
before and after training. It also saves generated examples for manual review
in `runs/chat-nano/chat-evaluation.json`, plus a resumable training checkpoint
and the `chat-inference.pt` file used above. The corpus licences and source
details are recorded in the training manifest. Dataset download and training
require connectivity and suitable compute; this repository does not include a
fine-tuned chat checkpoint or measured chat-quality improvement.

## How the small model works

1. A byte-level codec turns UTF-8 text and role markers into tokens. The same
   format is used for training and inference.
2. Every token passes through the shared control core. A router chooses up to
   two of four internal superstacks for additional computation, then returns
   their states to the core. These are parts of one checkpoint, although the
   internal computation is routed rather than fully dense.
3. A stopping head assigns probabilities to computation cycles. The last cycle
   absorbs any remaining stopping probability; a text head predicts the next
   byte or the end-of-turn marker.
4. Generation uses a KV cache, carries complete prior turns within the 1,024
   token context window, and restricts chat output to text bytes and stop tokens.

The nano configuration has width 256, eight control-core layers, four
eight-layer superstacks, and a maximum of two core cycles. Its code-level
correctness tests do not establish conversational usefulness. Evaluate a new
checkpoint using held-out loss and human review before treating its answers as
reliable.
