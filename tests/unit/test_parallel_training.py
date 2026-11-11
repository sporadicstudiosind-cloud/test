"""Two-process data-parallel training, prefetching and sampled ponder loops."""

import dataclasses
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

from iridium.config import get_config
from iridium.presets import get_preset
from iridium.training import parallel
from iridium.training.datasets import BatchLoader, build_corpus
from iridium.training.trainer import Prefetcher, TrainConfig, Trainer, split_batch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _corpus(n=24):
    return build_corpus(n, seed=0, split="train", mixture={"false_premise": 1.0})


def test_rank_shards_are_disjoint_and_cover_the_corpus():
    cfg = get_config("tiny")
    corpus = _corpus()
    seen = []
    for rank in range(2):
        loader = BatchLoader(corpus, cfg.codecs, 4, seed=3, shard=(rank, 2))
        seen.append([it.prompt for _, items in loader.batches() for it in items])
    whole = [it.prompt for _, items in BatchLoader(corpus, cfg.codecs, 4, seed=3).batches()
             for it in items]
    assert len(seen[0]) == len(seen[1]) == len(whole) // 2
    assert sorted(seen[0] + seen[1]) == sorted(whole)
    assert seen[0] == whole[0::2] and seen[1] == whole[1::2]


def test_split_batch_keeps_every_row_once():
    cfg = get_config("tiny")
    batch, _ = next(BatchLoader(_corpus(), cfg.codecs, 5, seed=0).batches())
    pieces = split_batch(batch, 3)
    assert [p.modality.shape[0] for p in pieces] == [2, 1, 2]
    assert torch.equal(torch.cat([p.discrete for p in pieces]), batch.discrete)
    assert sum(len(p.meta) for p in pieces) == 5
    assert sum(p.n_tokens for p in pieces) == batch.n_tokens
    assert split_batch(batch, 1) == [batch] and len(split_batch(batch, 99)) == 5


def test_prefetcher_preserves_order_and_surfaces_errors():
    assert list(Prefetcher(iter(range(50)), 4)) == list(range(50))
    assert list(Prefetcher(iter(range(5)), 0)) == list(range(5))

    def broken():
        yield 1
        raise RuntimeError("loader failed")

    stream = Prefetcher(broken(), 2)
    assert next(stream) == 1
    try:
        next(stream)
    except RuntimeError as exc:
        assert "loader failed" in str(exc)
    else:
        raise AssertionError("the loader's error was swallowed")
    endless = Prefetcher(iter(int, 1), 2)       # never ends; close must not hang
    next(endless)
    endless.close()


def test_sampled_loops_follow_the_probabilities_and_train():
    cfg = get_config("tiny")
    cfg = dataclasses.replace(cfg, router=dataclasses.replace(cfg.router, max_loops=3))
    from iridium.model.iridium1 import Iridium1
    torch.manual_seed(0)
    tcfg = TrainConfig(steps=4, batch_size=4, loop_probs=(0.5, 0.3, 0.2), log_every=1)
    trainer = Trainer(Iridium1(cfg), _corpus(), tcfg)
    draws = [trainer._loops_for(s) for s in range(2000)]
    share = np.bincount(draws, minlength=4)[1:] / len(draws)
    assert np.allclose(share, (0.5, 0.3, 0.2), atol=0.04)
    assert draws == [trainer._loops_for(s) for s in range(2000)]      # a function of the step
    history = trainer.train()
    assert len(history) == 4 and all(np.isfinite(h["total"]) for h in history)
    assert history[-1]["tokens_per_second"] > 0
    try:
        Trainer(Iridium1(cfg), _corpus(), dataclasses.replace(tcfg, loop_probs=(0.5, 0.2)))
    except ValueError:
        pass
    else:
        raise AssertionError("probabilities that do not sum to 1 were accepted")


# The worker runs from a temporary folder, so it is told where the checkout is:
# CI tests an uninstalled checkout, where `import iridium` only works from its root.
WORKER = """
import dataclasses, json, sys
sys.path.insert(0, sys.argv[2])
import torch
from iridium.config import get_config
from iridium.presets import get_preset
from iridium.training import parallel, run_preset
out = sys.argv[1]
rank, world = parallel.world_from_env()
preset = dataclasses.replace(get_preset("chat-34m"), config=get_config("tiny"),
    mixture={"false_premise": 1.0}, steps=6, batch_size=4, window=64, rounds=2,
    micro_batch=2, ema_decay=0.0, loop_probs=(0.5, 0.5))
captured = {}
orig = run_preset.Path
from iridium.training import trainer as T
keep = T.Trainer.train
def train(self, *a, **k):
    captured["model"] = self.model
    return keep(self, *a, **k)
T.Trainer.train = train
run_preset.train_preset(preset, device="cpu", out=out, rank=rank, world=world)
flat = torch.cat([p.detach().double().reshape(-1) for p in captured["model"].parameters()])
open(f"{out}/sum{rank}.json", "w").write(json.dumps([float(flat.sum()), float(flat.abs().sum())]))
"""


def test_two_processes_train_in_lockstep(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(WORKER))
    out = tmp_path / "runs"
    out.mkdir()
    code = parallel.launch([], 2, log_dir=out,
                           command=[sys.executable, str(script), str(out), str(REPO_ROOT)])
    assert code == 0, (out / "rank1.log").read_text() if (out / "rank1.log").exists() else ""
    sums = [json.loads((out / f"sum{r}.json").read_text()) for r in range(2)]
    assert sums[0] == sums[1], "ranks ended with different weights"
    run = out / "chat-34m"
    assert (run / "chat-34m-final.pt").exists() and (run / "chat-34m-round0.pt").exists()
    blob = torch.load(run / "chat-34m-final.pt", weights_only=False)
    assert blob["completed_steps"] == 6
    assert json.loads((run / "preset.json").read_text())["trained"] is True


def test_launch_stops_the_group_when_one_rank_fails(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("import os, sys, time\n"
                      "if os.environ['RANK'] == '1': sys.exit(3)\n"
                      "time.sleep(120)\n")
    assert parallel.launch([], 2, command=[sys.executable, str(script)]) == 3


def test_iridium_50m_trains_every_loop_it_can_be_asked_for():
    preset = get_preset("iridium-50m")
    assert len(preset.loop_probs) == preset.config.router.max_loops == 3
    assert abs(sum(preset.loop_probs) - 1) < 1e-9 and preset.loop_probs[0] >= 0.5
    assert preset.batch_size % 2 == 0 and preset.free_tier == "kaggle_2xt4"
