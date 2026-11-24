# Kaggle runbook: train `iridium-50m` (for a small, low-effort agent)

You drive one Kaggle notebook across several sessions. Follow the steps in order.
Do not edit model code or hyperparameters. If a step fails twice, stop and report
the exact error text to the human.

## What you need from the human (once)

1. A Kaggle account with phone verification (needed for GPU and internet).
2. Kaggle API credentials at `~/.kaggle/kaggle.json`. Never print this file.
3. A Kaggle **secret** named `GITHUB_TOKEN` (read-only, for the private repo). Secrets
   can only be added in the notebook editor (Add-ons -> Secrets), so the human must add
   and attach it. Never print or log the token.
4. (Optional) A Kaggle **secret** named `HF_TOKEN`, a free Hugging Face read token. It
   lifts the anonymous rate limit that slows data preparation. The notebook uses it only
   if it is present, and never prints it.

## The notebook

`notebooks/iridium_studio_kaggle.ipynb`, branch `claude/gallant-faraday-lhycva` of
`sporadicstudiosind-cloud/test`. It needs no edits: it works out where the run is
from what is on disk and in its inputs.

`kernel-metadata.json`:

```json
{
  "id": "<kaggle-username>/iridium-50m",
  "title": "iridium-50m",
  "code_file": "iridium_studio_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "kernel_sources": []
}
```

CLI field names change between versions, so trust `kaggle kernels push --help` over this file.

## Steps

1. **Data session.** Push with `enable_gpu: false` (Accelerator None). This spends no GPU
   quota and writes about 7.5 GB of shards to its output under `iridium-runs/data/`. The
   notebook prints a warning if a GPU is attached in this session. If it hits the time
   limit before finishing, report that to the human.
2. **First training session.** Set `enable_gpu: true`, and add the data version to
   `kernel_sources` (in the editor: Add Input, then Notebook Output Files). The notebook
   finds `iridium-50m/prepared.json` under `/kaggle/input` by itself, so no path is needed.
   The accelerator must be **GPU T4 x2**: training uses both cards and is about twice as
   fast as on one. If your CLI version cannot choose the accelerator, ask the human to set
   it once in the notebook editor; it is remembered.
   Push. It stops itself after about 11 hours. The log should say
   `training on 2 GPUs at once`; if it does not, report that.
3. **Later sessions.** Add the newest training version to `kernel_sources` too. Push again.
   The notebook copies the newest checkpoint from the attached inputs and resumes from it.
   Repeat until the last cell reports the run finished (`iridium-runs/iridium-50m/preset.json`
   then has `"trained": true`).
4. Poll with `kaggle kernels status` every 15-30 minutes, not more often.
5. When finished, download `iridium-runs/iridium-50m-inference.pt`, `battery.json` and
   `specialists.json` from the output.
6. **Next run, only when the human asks for it.** Set `PRESET = 'iridium-50m-next'` in the
   notebook (the one value to change), then repeat steps 2-5 with the same data version
   attached. The next run reads iridium-50m's shards and starts fresh: it does not resume
   iridium-50m's checkpoints. Its run folder is `iridium-runs/iridium-50m-next`.

## Report to the human

- Status, step reached, and the latest `val_*_bpb` lines (lower is better).
- The `tok/s` figure from the training log lines. The measured rate on two T4s was about
  9.8k tokens/s for both together, which implies about 140 GPU-hours for 5B tokens.
  Compare this run's `tok/s` with that.
- The contents of `battery.json`, if it exists.
- Any error text, verbatim, without guessing at the cause.

## Limits to respect

- Kaggle gives about 30 GPU-hours a week and 12 hours a session at the time of writing.
  Check the current numbers. Do not run two sessions at once.
- A full `iridium-50m` run is about 140 GPU-hours at the measured rate, roughly 4-5 weeks
  of the weekly quota.
- To stop a multi-GPU run, use **Stop session** (the power button) in the Kaggle editor.
  On Linux, ranks also exit when their launcher dies, so a cell that was killed does not
  leave its ranks training.
- Never print secrets, commit `kaggle.json`, or make the notebook public.
