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

1. **Data session.** Push with `enable_gpu: false`. This spends no GPU quota and writes
   about 7.5 GB of shards to its output. If it hits the time limit before finishing,
   report that to the human.
2. **First training session.** Set `enable_gpu: true`, and add the data version to
   `kernel_sources`. Push. It stops itself after about 11 hours.
3. **Later sessions.** Add the newest training version to `kernel_sources` too. Push again.
   Repeat until the log says the run finished.
4. Poll with `kaggle kernels status` every 15-30 minutes, not more often.
5. When finished, download `iridium-runs/iridium-50m-inference.pt` and `battery.json`
   from the output.

## Report to the human

- Status, step reached, and the latest `val_*_bpb` lines (lower is better).
- Tokens per second from the first training session. The ~20 GPU-hour figure is an
  estimate, and this is the number that tests it.
- The contents of `battery.json`, if it exists.
- Any error text, verbatim, without guessing at the cause.

## Limits to respect

- Kaggle gives about 30 GPU-hours a week and 12 hours a session at the time of writing.
  Check the current numbers. Do not run two sessions at once.
- Never print secrets, commit `kaggle.json`, or make the notebook public.
