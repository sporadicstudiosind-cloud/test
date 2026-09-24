"""A subword preset must be read, prompted and decoded through its own tokenizer.

The failure these guard against does not raise: a model trained on BPE ids and
prompted or decoded as bytes produces fluent nonsense, and every earlier layer
of the pipeline reports success.
"""

from dataclasses import replace

import pytest
import torch

from iridium.codecs.spans import Sample, text_span
from iridium.config import TEXT_ID_OFFSET, get_config
from iridium.data.tokenizer import BytePairTokenizer
from iridium.model.iridium1 import Iridium1
from iridium.runtime.device import detect
from iridium.runtime.generate import generate
from iridium.training.tokenizer_bridge import tokenizer_from_manifest, tokenizer_manifest

TEXT = ["the cat sat on the mat. the cat ran. the mat sat still."] * 20


def _tok(size=300):
    return BytePairTokenizer().train(TEXT, size)


def test_text_span_uses_the_tokenizer_when_given():
    tok = _tok()
    span = text_span("the cat", offset=16, tokenizer=tok)
    assert list(span.payload - 16) == tok.encode("the cat")
    assert len(span.payload) < len("the cat".encode())           # merges happened
    assert list(text_span("ab", offset=16).payload) == [16 + 97, 16 + 98]


def test_tokenizer_round_trips_through_a_manifest():
    tok = _tok()
    manifest = {"tokenizer": tokenizer_manifest(tok)}
    again = tokenizer_from_manifest(manifest)
    assert again.encode("the mat sat") == tok.encode("the mat sat")
    assert again.vocab_size == tok.vocab_size


def test_subword_checkpoint_without_its_tokenizer_is_refused():
    cfg = replace(get_config("tiny"), text_vocab_size=300)
    with pytest.raises(ValueError, match="does not store its tokenizer"):
        tokenizer_from_manifest({"model_config": cfg.to_dict()})
    assert tokenizer_from_manifest({"model_config": get_config("tiny").to_dict()}) is None


def test_generate_decodes_through_the_tokenizer():
    tok = _tok()
    cfg = get_config("tiny")
    model = Iridium1(cfg).eval()
    # Force the head to emit one merged token, then EOS.
    target = tok.encode("the cat")[0]
    assert len(tok.decode([target]).encode()) > 1
    calls = {"n": 0}

    class Forced(torch.nn.Module):
        def forward(self, h):
            logits = torch.full((*h.shape[:-1], cfg.codecs.vocab_size), -1e9)
            logits[..., TEXT_ID_OFFSET + target if calls["n"] == 0 else 2] = 0.0
            calls["n"] += 1
            return logits

    model.codecs.text_head = Forced()
    out = generate(model, Sample([text_span("the", offset=16, tokenizer=tok)]),
                   max_new_tokens=4, text_only=True, tokenizer=tok)
    assert out.text == tok.decode([target])


def test_xla_is_explicit_and_explains_a_missing_torch_xla():
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="torch_xla"):
            detect("xla")
    assert detect("auto").backend != "xla"


def test_every_tokenizer_kind_round_trips_through_its_state():
    from iridium.data.tokenization import ByteTokenizer, FastBPETokenizer, fast_available, from_state

    kinds = [ByteTokenizer(), _tok()]
    if fast_available():
        kinds.append(FastBPETokenizer.train(TEXT, 300))
    for tok in kinds:
        again = from_state(tok.to_dict())
        s = "the cat sat, naïvely ✓"
        assert again.encode(s) == tok.encode(s)
        assert again.decode(again.encode(s)) == s
        assert again.encode_batch([s, "x"]) == [tok.encode(s), tok.encode("x")]


def test_check_fits_refuses_a_tokenizer_larger_than_the_table():
    from iridium.data.tokenization import check_fits

    cfg = get_config("tiny")
    check_fits(None, cfg)
    big = type("T", (), {"vocab_size": cfg.codecs.vocab_size})()
    with pytest.raises(ValueError, match="embedding rows"):
        check_fits(big, cfg)
