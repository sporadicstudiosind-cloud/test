"""Cloze benchmark scoring (iridium.evaluation.cloze), offline."""
import numpy as np
import torch
import torch.nn.functional as F

from iridium.config import get_config
from iridium.evaluation.cloze import ClozeItem, _sample, choice_logprobs, evaluate_items
from iridium.model.iridium1 import Iridium1


def _model():
    torch.manual_seed(0)
    return Iridium1(get_config("tiny")).eval()


def test_choice_logprob_is_the_answer_tokens_only():
    model = _model()
    lp = choice_logprobs(model, ["The sky is"], [" blue"])[0]
    # Same quantity by hand: log-probs of the answer bytes given everything before.
    from iridium.codecs.bank import TensorBatch, continuous_dims
    from iridium.codecs.spans import collate
    batch = TensorBatch(collate([_sample("The sky is", " blue", None, 4096)],
                                continuous_dims(model.cfg.codecs)))
    with torch.no_grad():
        logits = model.codecs.text_head(model(batch).hidden[:, :-1])
    logp = F.log_softmax(logits.float(), -1)
    target = batch.discrete[:, 1:]
    mask = batch.supervised[:, 1:]
    manual = float((logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask).sum())
    assert abs(lp - manual) < 1e-3
    assert int(mask.sum()) == len(" blue")


def test_long_context_is_cut_from_the_left_never_the_answer():
    s = _sample("x" * 5000, " answer", None, 64)
    assert sum(len(sp) for sp in s.spans) == 64
    assert s.spans[-1].supervised and len(s.spans[-1]) == len(" answer")


def test_evaluate_items_scores_and_normalises():
    model = _model()
    items = [ClozeItem("Question: 2 + 2?\\nAnswer:", [" 4", " five", " a long wrong answer"], 0)
             for _ in range(3)]
    result = evaluate_items(model, items, task="arc_easy")
    assert result.n == 3 and 0.0 <= result.acc_norm <= 1.0 and result.chance == 0.25
    assert set(result.as_dict()) >= {"acc", "acc_norm", "acc_uncond"}
