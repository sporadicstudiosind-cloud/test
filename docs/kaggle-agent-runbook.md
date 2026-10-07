# Kaggle runbook: train Iridium `50m` (for a small, low-effort agent)

You drive one Kaggle notebook that trains the `50m` preset. Follow the steps in
order. Do not edit model code. Do not change hyperparameters. If a step fails
twice, stop and report the exact error text to the human.

## What you need from the human (once)

1. A Kaggle account with phone verification (needed for GPU and internet).
2. Kaggle API credentials at `~/.kaggle/kaggle.json` (Kaggle -> Settings -> API -> Create New Token).
   Never print this file.
3. A Kaggle **secret** named `GITHUB_TOKEN` (a read-only GitHub token for the
   private repo). Secrets can only be added in the notebook editor UI
   (Add-ons -> Secrets), so the human must add it and attach it to this notebook.
   You cannot do this through the API. Never print or log the token.

## Steps

1. `pip install kaggle` and check `kaggle --version` works.
2. Get the notebook: `notebooks/iridium_studio_kaggle.ipynb` from branch
   `claude/gallant-faraday-lhycva` of `sporadicstudiosind-cloud/test`.
3. Make exactly two edits in cell 5 / cell 11 of the notebook:
   - `PRESET = 'chat-100m'` -> `PRESET = '50m'`
   - leave `RESUME_FROM = 'auto'` as it is (it resumes from the newest round).
4. Make a folder with the notebook and a `kernel-metadata.json`:
   ```json
   {
     "id": "<kaggle-username>/iridium-50m",
     "title": "iridium-50m",
     "code_file": "iridium_studio_kaggle.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": true,
     "enable_gpu": true,
     "enable_internet": true
   }
   ```
   Check `kaggle kernels push --help` for a GPU-type option (e.g. T4 vs P100);
   field names change between CLI versions, so trust `--help` over this file.
5. Push it: `kaggle kernels push -p <folder>`. This runs the whole notebook as a
   committed version, so `/kaggle/working` (checkpoints) is kept as output.
6. Poll politely: `kaggle kernels status <user>/iridium-50m` every 15-30 minutes,
   not more often. States: `running`, `complete`, `error`.
7. When it is `complete` or `error`, download output:
   `kaggle kernels output <user>/iridium-50m -p ./out`.
   Expect `out/iridium-runs/50m/50m-round0.pt`, `-round1.pt`, ... and
   `50m-final.pt` when it finished. `latest.json` names the newest checkpoint.
8. If the run stopped early (time limit, `error` after some rounds): push again
   (step 5) with the previous output attached as a dataset input, or copy the
   checkpoints into `/kaggle/working/iridium-runs/50m/` first. `RESUME_FROM = 'auto'`
   continues from the newest round. Do not start over from scratch.

## Report to the human

- Final status, how many rounds finished, and the last `val_*_bpb` line from the log
  (lower is better; a falling number means it is learning).
- The path to the newest checkpoint.
- Any error text, verbatim, without guessing at the cause.

## Limits to respect

- Kaggle gives a weekly GPU quota and caps a single session's runtime (about 30 h/week
  and 12 h/session at the time of writing; check the Kaggle docs for current numbers).
  Do not start a second run while one is running.
- Never print secrets, never commit `kaggle.json`, never make the notebook public.
