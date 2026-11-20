"""Teacher-written corpus: generate (mocked HTTP), shard round-trip, subject mapping."""

import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from iridium.config import TEXT_ID_OFFSET
from iridium.data import teacher_corpus
from iridium.data.domains import subject_of
from iridium.data.teacher_corpus import (
    MAX_TRIES, TEACHER_FAMILIES, TOPICS, build_prompt, generate, to_shards,
)

KEY_ENV = "TEACHER_API_KEY"
SECRET = "sk-test-secret-0123456789"
HOST = "api.example.test"
BASE = f"https://{HOST}/openai/v1"


class _Resp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeAPI:
    """Stands in for urllib.request.urlopen. Each script entry is text or an HTTP status."""

    def __init__(self, script=()):
        self.script = list(script)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        prompt = json.loads(req.data.decode("utf-8"))["messages"][0]["content"]
        item = self.script.pop(0) if self.script else f"reply {len(self.requests)}: {prompt[:24]}"
        if isinstance(item, int):
            raise urllib.error.HTTPError(req.full_url, item, "status", None, None)
        return _Resp({"choices": [{"message": {"content": item}}]})

    def prompts(self):
        return [json.loads(r.data.decode("utf-8"))["messages"][0]["content"] for r in self.requests]


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = FakeAPI()
    monkeypatch.setattr(teacher_corpus.urllib.request, "urlopen", fake)
    sleeps = []
    monkeypatch.setattr(teacher_corpus, "time", SimpleNamespace(sleep=sleeps.append))
    fake.sleeps = sleeps
    return fake


def _rows(path):
    """Complete JSON rows; a line cut short by a crash is skipped, as generate does."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


# -- subject mapping ----------------------------------------------------------

def test_subject_of_maps_teacher_families_to_their_subjects():
    assert subject_of("math_teacher") == "math"
    assert subject_of("science_teacher") == "science"
    assert subject_of("code_teacher") == "code"
    assert subject_of("chat_teacher") == "chat"
    assert subject_of("math_text") == "math"          # existing mappings unchanged
    assert subject_of("no_such_family") is None


def test_every_teacher_subject_has_enough_topics():
    for family, subject in TEACHER_FAMILIES.items():
        assert len(TOPICS[subject]) >= 30, family


# -- prompts ------------------------------------------------------------------

def test_prompts_are_deterministic_and_cover_every_topic():
    assert build_prompt("math", 7, seed=3) == build_prompt("math", 7, seed=3)
    assert build_prompt("math", 7, seed=3) != build_prompt("math", 7, seed=4)
    topics = TOPICS["math"]
    prompts = [build_prompt("math", i, seed=0) for i in range(len(topics))]
    for topic in topics:
        assert any(f" about {topic} for " in p for p in prompts), topic


# -- generate -----------------------------------------------------------------

def test_generate_writes_one_row_per_sample_and_never_the_key(tmp_path, api, monkeypatch, capsys):
    out = tmp_path / "math.jsonl"
    written = generate("math", 3, out, BASE, "model-x", api_key_env=KEY_ENV)
    assert written == 3
    rows = _rows(out)
    assert [r["seed_index"] for r in rows] == [0, 1, 2]
    assert set(rows[0]) == {"family", "subject", "text", "prompt", "model", "base_url",
                            "seed_index", "created"}
    assert rows[0]["family"] == "math_teacher" and rows[0]["subject"] == "math"
    assert rows[0]["model"] == "model-x" and rows[0]["base_url"] == HOST
    assert api.requests[0].get_header("Authorization") == f"Bearer {SECRET}"
    assert SECRET not in out.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert SECRET not in captured.out and SECRET not in captured.err


def test_missing_key_is_a_clear_error_and_no_request_is_made(tmp_path, api, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    out = tmp_path / "math.jsonl"
    with pytest.raises(RuntimeError, match=KEY_ENV):
        generate("math", 2, out, BASE, "model-x", api_key_env=KEY_ENV)
    assert api.requests == [] and not out.exists()


def test_resume_skips_seed_indices_already_in_the_file(tmp_path, api):
    out = tmp_path / "math.jsonl"
    assert generate("math", 2, out, BASE, "model-x") == 2
    # A line cut short by an earlier crash must not stop the resume.
    with out.open("a", encoding="utf-8") as fh:
        fh.write('{"family": "math_teacher", "seed_in')
    assert generate("math", 4, out, BASE, "model-x") == 2
    assert [r["seed_index"] for r in _rows(out)] == [0, 1, 2, 3]
    assert api.prompts()[2:] == [build_prompt("math", 2), build_prompt("math", 3)]
    assert generate("math", 4, out, BASE, "model-x") == 0      # nothing left to do
    assert len(api.requests) == 4


def test_exact_duplicates_and_empty_outputs_are_dropped(tmp_path, api):
    api.script = ["Same text.", "Same text.", "   ", "A different text."]
    out = tmp_path / "math.jsonl"
    assert generate("math", 4, out, BASE, "model-x") == 2
    assert [r["text"] for r in _rows(out)] == ["Same text.", "A different text."]


def test_rate_limit_backs_off_exponentially_then_stops_cleanly(tmp_path, api):
    api.script = [429] * 10
    out = tmp_path / "math.jsonl"
    assert generate("math", 3, out, BASE, "model-x") == 0
    assert len(api.requests) == MAX_TRIES
    assert api.sleeps == [2.0, 4.0, 8.0, 16.0]          # no sleep after the last try
    assert not out.exists() or _rows(out) == []


def test_rate_limit_then_success_keeps_going_and_counts_written_rows(tmp_path, api):
    api.script = ["first", 429, "second"] + [429] * MAX_TRIES
    out = tmp_path / "math.jsonl"
    # Index 0 succeeds; index 1 is rate-limited once then succeeds; index 2 is
    # rate-limited on every try, so the call stops with two rows written.
    assert generate("math", 5, out, BASE, "model-x") == 2
    assert [r["text"] for r in _rows(out)] == ["first", "second"]
    assert len(api.requests) == 1 + 2 + MAX_TRIES


def test_other_http_errors_raise_without_the_key(tmp_path, api):
    api.script = [500]
    with pytest.raises(RuntimeError, match="HTTP 500") as info:
        generate("math", 1, tmp_path / "math.jsonl", BASE, "model-x")
    assert SECRET not in str(info.value)


def test_unknown_subject_is_rejected(tmp_path, api):
    with pytest.raises(ValueError, match="unknown teacher subject"):
        generate("poetry", 1, tmp_path / "x.jsonl", BASE, "model-x")
    assert api.requests == []


def test_family_name_is_accepted_as_subject(tmp_path, api):
    out = tmp_path / "code.jsonl"
    assert generate("code_teacher", 1, out, BASE, "model-x") == 1
    assert _rows(out)[0]["family"] == "code_teacher"


# -- to_shards ----------------------------------------------------------------

class CharTokenizer:
    """A tiny subword-style tokenizer: one id per ASCII character."""

    kind = "char"
    vocab_size = 128

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _decode_docs(ids, mask, decode):
    """Split a shard item back into its documents, checking the supervision layout."""
    from iridium.training.tasks import BOS, EOS
    docs, buf = [], []
    for tid, m in zip(ids.tolist(), mask.tolist()):
        if tid == BOS:
            assert m == 0, "BOS is context, never a target"
        elif tid == EOS:
            assert m == 1, "EOS is a target"
            docs.append(decode(buf))
            buf = []
        else:
            assert tid >= TEXT_ID_OFFSET and m == 1, "text is a target"
            buf.append(tid - TEXT_ID_OFFSET)
    assert not buf, "every text run ends with EOS"
    return docs


def _ids_and_mask(sample):
    ids = np.concatenate([np.asarray(s.payload, np.int64) for s in sample.spans])
    mask = np.concatenate([np.full(len(s.payload), int(s.supervised), np.uint8)
                           for s in sample.spans])
    return ids, mask


@pytest.mark.parametrize("tokenizer,decode", [
    (None, lambda ids: bytes(ids).decode("utf-8")),
    (CharTokenizer(), CharTokenizer().decode),
])
def test_to_shards_round_trips_text_and_supervision(tmp_path, tokenizer, decode):
    from iridium.data.shards import Shard

    math_texts = ["Two plus two is four.", "A triangle has three sides.", "Zero is even."]
    chat_texts = ["User: hi\nAssistant: hello!", "User: why?\nAssistant: because."]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with a.open("w", encoding="utf-8") as fh:
        for text in math_texts:
            fh.write(json.dumps({"family": "math_teacher", "subject": "math", "text": text,
                                 "model": "model-x", "base_url": HOST}) + "\n")
        for text in chat_texts:
            fh.write(json.dumps({"family": "chat_teacher", "subject": "chat", "text": text,
                                 "model": "model-y", "base_url": HOST}) + "\n")
    with b.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"family": "math_teacher", "text": math_texts[0]}) + "\n")  # duplicate
        fh.write(json.dumps({"family": "poetry_teacher", "text": "roses"}) + "\n")
        fh.write(json.dumps({"family": "math_teacher", "text": "   "}) + "\n")
        fh.write("{not json\n")

    shard_dir = tmp_path / "shards"
    manifest = to_shards([a, b], shard_dir, tokenizer, window=256, vocab_rows=4096)

    assert manifest["rows"] == 5
    assert manifest["dropped"] == {"duplicate": 1, "empty": 1, "malformed": 1,
                                   "unknown_family": 1, "too_long": 0}
    assert manifest["models"] == ["model-x", "model-y"]
    assert manifest["hosts"] == [HOST]
    assert json.loads((shard_dir / "teacher_manifest.json").read_text())["rows"] == 5

    for family, texts in (("math_teacher", math_texts), ("chat_teacher", chat_texts)):
        shard = Shard(shard_dir / f"{family}-train")
        assert shard.family == family and shard.meta["window"] == 256
        docs = []
        for i in range(len(shard)):
            ids, mask = _ids_and_mask(shard.sample(i))
            docs += _decode_docs(ids, mask, decode)
        assert sorted(docs) == sorted(texts)


def test_to_shards_holds_out_a_stable_test_split(tmp_path):
    from iridium.data.shards import Shard

    texts = [f"Fact number {i} is simple." for i in range(40)]
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps({"family": "science_teacher", "text": t}) + "\n"
                           for t in texts), encoding="utf-8")
    first = to_shards([src], tmp_path / "one", None, 512, test_fraction=0.5, vocab_rows=4096)
    again = to_shards([src], tmp_path / "two", None, 512, test_fraction=0.5, vocab_rows=4096)
    counts = first["families"]["science_teacher"]
    assert counts["train"] + counts["test"] == 40
    assert counts["train"] > 0 and counts["test"] > 0
    assert again["families"] == first["families"]          # the split is a hash, not chance
    held = Shard(tmp_path / "one" / "science_teacher-test")
    assert len(held) > 0


def test_to_shards_rejects_test_as_the_main_split(tmp_path):
    with pytest.raises(ValueError, match="split must not be 'test'"):
        to_shards([], tmp_path, None, 64, split="test", vocab_rows=512)


# -- CLI ----------------------------------------------------------------------

def test_cli_teacher_generate_parses_and_needs_the_key(tmp_path, monkeypatch, capsys):
    from iridium.cli import main

    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    code = main(["teacher", "generate", "--subject", "math", "--n", "2",
                 "--out", str(tmp_path / "m.jsonl"), "--base-url", BASE, "--model", "model-x"])
    assert code == 2
    assert "TEACHER_API_KEY" in capsys.readouterr().err


def test_cli_teacher_shard_uses_the_prepared_tokenizer(tmp_path):
    from iridium.cli import build_parser

    args = build_parser().parse_args(["teacher", "shard", "--in", "a.jsonl", "b.jsonl",
                                      "--data", str(tmp_path), "--preset", "iridium-50m"])
    assert args.action == "shard" and args.inputs == ["a.jsonl", "b.jsonl"]
    assert args.preset == "iridium-50m"
