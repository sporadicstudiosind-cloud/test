"""Provenance, licensing and mixture arithmetic for the real-text corpus.

The offline tests here pin the things that were actually wrong: a boilerplate
stripper that must not eat the book, a shuffle buffer measured in rows when
rows differ in size by two orders of magnitude, and a mixture that is exact by
construction rather than in expectation.

The network tests are opt-in (``IRIDIUM_NETWORK_TESTS=1``). They are the only
ones that can catch a dataset whose config or split has been renamed upstream —
a whole class of failure a mocked test cannot see — so they are worth running
before trusting a notebook, and worth skipping in CI.
"""

import os

import pytest

from iridium.data import text_corpus as tc

NETWORK = os.environ.get("IRIDIUM_NETWORK_TESTS") == "1"
needs_network = pytest.mark.skipif(
    not NETWORK, reason="set IRIDIUM_NETWORK_TESTS=1 to hit the Hugging Face Hub"
)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(tc.SOURCES))
def test_every_source_states_its_terms(key):
    """A corpus with no stated licence is not "ethically sourced", it is
    undocumented. Each field here is something a derivative work may owe."""
    spec = tc.SOURCES[key]
    assert spec.key == key
    assert spec.licence.strip(), f"{key} has no licence"
    assert spec.obligation.strip(), f"{key} does not say what it obliges"
    assert spec.description.strip()
    assert spec.homepage.startswith("http"), f"{key} has no verifiable homepage"
    assert spec.text_field
    assert spec.split


def test_default_mixture_is_a_normalised_subset():
    assert set(tc.DEFAULT_MIX) <= set(tc.SOURCES)
    assert abs(sum(tc.DEFAULT_MIX.values()) - 1.0) < 1e-9
    assert all(w > 0 for w in tc.DEFAULT_MIX.values())


def test_licence_notice_names_every_source_it_covers():
    notice = tc.licence_notice(tc.DEFAULT_MIX)
    for key in tc.DEFAULT_MIX:
        spec = tc.SOURCES[key]
        assert spec.dataset in notice
        assert spec.licence in notice
    # The share-alike obligation is the one that actually binds a derivative.
    assert "CC BY-SA" in notice
    assert "share-alike" in notice.lower()


def test_licence_notice_does_not_overclaim():
    """Recording provenance is not the same as clearing the data of problems,
    and the notice has to say so or it is marketing."""
    notice = tc.licence_notice(tc.DEFAULT_MIX).lower()
    assert "bias" in notice


def test_source_spec_round_trips_into_a_manifest():
    d = tc.SOURCES["wikipedia"].as_dict()
    assert d["licence"] == "CC BY-SA 4.0"
    assert d["dataset"] == "wikimedia/wikipedia"
    import json
    json.dumps(d)   # must survive going into a run manifest


# ---------------------------------------------------------------------------
# Gutenberg boilerplate
# ---------------------------------------------------------------------------

BOOK = "CHAPTER ONE\n\nIt was a bright cold day in April.\n" * 4


def test_strips_header_and_footer():
    raw = (
        "The Project Gutenberg eBook of Something, by Someone\n"
        "This ebook is for the use of anyone anywhere at no cost...\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        + BOOK +
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "Updated editions will replace the previous one...\n"
        "START: FULL LICENSE\n"
    )
    out = tc.strip_gutenberg_boilerplate(raw)
    assert "CHAPTER ONE" in out
    assert "PROJECT GUTENBERG" not in out.upper()
    assert "FULL LICENSE" not in out.upper()


def test_strips_the_this_variant_and_is_case_insensitive():
    raw = ("*** start of this project gutenberg ebook x ***\n" + BOOK
           + "*** End of this Project Gutenberg EBook X ***\n")
    out = tc.strip_gutenberg_boilerplate(raw)
    assert "CHAPTER ONE" in out and "gutenberg" not in out.lower()


def test_keeps_the_document_when_markers_are_absent():
    """Conservative by design: a missing marker must not truncate a book to
    nothing. Silently emptying documents is worse than leaving a header in."""
    assert tc.strip_gutenberg_boilerplate(BOOK).strip() == BOOK.strip()


def test_keeps_the_document_when_markers_are_out_of_order():
    raw = "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n" + BOOK
    assert "CHAPTER ONE" in tc.strip_gutenberg_boilerplate(raw)


def test_never_returns_empty():
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
    assert tc.strip_gutenberg_boilerplate(raw).strip()


def test_strips_the_plain_text_footer_that_precedes_the_asterisks():
    """The variant that actually leaked. Two footers appear in real files and
    the plain one comes first, so matching only the asterisk form leaves a
    licence line in every single book."""
    raw = ("*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n" + BOOK +
           "End of the Project Gutenberg EBook of X, by Someone\n\n"
           "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n"
           "***** This file should be named 2685-0.txt *****\n")
    out = tc.strip_gutenberg_boilerplate(raw)
    assert "CHAPTER ONE" in out
    assert "gutenberg" not in out.lower()
    assert "2685-0.txt" not in out


def test_strips_the_possessive_footer_without_the_article():
    """"End of Project Gutenberg's X" — no "the", apostrophe instead."""
    for apostrophe in ("'", "\u2019"):
        raw = ("*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n" + BOOK +
               f"End of Project Gutenberg{apostrophe}s X, by Someone\n"
               "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n")
        out = tc.strip_gutenberg_boilerplate(raw)
        assert "CHAPTER ONE" in out, apostrophe
        assert "gutenberg" not in out.lower(), apostrophe


def test_strips_markers_with_no_surrounding_spaces():
    raw = ("***START OF THE PROJECT GUTENBERG EBOOK X***\n" + BOOK +
           "***END OF THE PROJECT GUTENBERG EBOOK X***\n")
    out = tc.strip_gutenberg_boilerplate(raw)
    assert "CHAPTER ONE" in out and "gutenberg" not in out.lower()


def test_a_footer_phrase_in_the_body_does_not_truncate_the_book():
    """The position guard. A bibliography mentioning the phrase early in a book
    must not cut it there — losing a licence line is cheap, losing nine tenths
    of a book without noticing is not."""
    raw = ("*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n"
           "End of Project Gutenberg's earlier volume, cited in the preface.\n"
           + BOOK * 20 +
           "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n")
    out = tc.strip_gutenberg_boilerplate(raw)
    assert out.count("CHAPTER ONE") == 80, "the book was truncated at a false footer"


def test_gutenberg_is_wired_to_the_cleaner():
    assert tc.CLEANERS["gutenberg"] is tc.strip_gutenberg_boilerplate


def test_gutenberg_obligation_mentions_the_header():
    """The works are public domain; the wrapper PG adds is not. Saying
    "no obligation" without that distinction is the inaccurate version."""
    assert "header" in tc.SOURCES["gutenberg"].obligation.lower()


# ---------------------------------------------------------------------------
# streaming economics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(tc.SOURCES))
def test_shuffle_buffer_is_sized_in_bytes_not_rows(key):
    """A shuffle buffer is counted in rows, but paid for in bytes. A row is a
    whole book in one corpus and a paragraph in another, so a constant row
    count downloads a gigabyte from one source and a megabyte from another."""
    spec = tc.SOURCES[key]
    held = spec.shuffle_buffer * spec.typical_doc_bytes
    assert held <= 64_000_000, (
        f"{key} would hold {held/1e6:.0f} MB before yielding its first document"
    )
    assert spec.shuffle_buffer >= 8, f"{key} barely shuffles at all"


@pytest.mark.parametrize("key", sorted(tc.SOURCES))
def test_window_cap_keeps_a_sane_download_ratio(key):
    """Eight 256-byte windows out of a one-megabyte book is 500x more download
    than data kept. The cap is per source for exactly this reason."""
    spec = tc.SOURCES[key]
    kept = spec.windows_per_doc * 256
    assert kept <= spec.typical_doc_bytes * 2, f"{key} over-samples its documents"
    if spec.typical_doc_bytes > 100_000:
        assert spec.windows_per_doc >= 32, (
            f"{key} has book-length rows but keeps only "
            f"{spec.windows_per_doc} windows of each"
        )


def test_realised_mixture_counts_sources():
    class FakeItem:
        def __init__(self, src):
            self.truth = {"source": src}
    items = [FakeItem("a")] * 3 + [FakeItem("b")]
    assert tc.realised_mixture(items) == {"a": 0.75, "b": 0.25}


def test_realised_mixture_survives_an_empty_corpus():
    assert tc.realised_mixture([]) == {}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

DOCS = [f"document {i} with enough words to be a document" for i in range(4000)]


def test_split_shares_tile_the_range_without_overlapping():
    bands = sorted(tc.SPLIT_SHARES.values())
    assert bands[0][0] == 0 and bands[-1][1] == 100
    for (_, prev_hi), (lo, _) in zip(bands, bands[1:]):
        assert lo == prev_hi, f"a gap or overlap at {prev_hi}/{lo}"


def test_every_document_lands_in_exactly_one_split():
    for doc in DOCS[:500]:
        assert sum(tc.in_split(doc, s) for s in tc.SPLIT_SHARES) == 1


def test_split_membership_is_a_property_of_the_document():
    """Not of the seed, the buffer size or the position in the stream — which
    is the whole point. The same book must land in the same split everywhere."""
    doc = DOCS[0]
    first = tc.split_bucket(doc)
    assert all(tc.split_bucket(doc) == first for _ in range(5))
    # and different documents do not all collapse into one bucket
    assert len({tc.split_bucket(d) for d in DOCS[:200]}) > 50


def test_bucket_is_stable_across_encodings_of_the_same_text():
    doc = "a document with a non-ascii character: caf\u00e9"
    assert tc.split_bucket(doc) == tc.split_bucket(str(doc))


def test_splits_are_roughly_the_size_they_claim():
    counts = {s: sum(tc.in_split(d, s) for d in DOCS) for s in tc.SPLIT_SHARES}
    for name, (lo, hi) in tc.SPLIT_SHARES.items():
        share = counts[name] / len(DOCS)
        assert abs(share - (hi - lo) / 100) < 0.03, counts


def test_unknown_split_name_falls_back_to_train_rather_than_emptying():
    """A typo in a split name must not silently produce an empty corpus."""
    assert any(tc.in_split(d, "trian") for d in DOCS[:200])


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

@needs_network
@pytest.mark.parametrize("key", sorted(tc.DEFAULT_MIX))
def test_source_is_reachable_with_the_config_it_claims(key):
    """Catches an upstream rename. The Gutenberg corpus exposes language as a
    *split*, not a config, and asking for the wrong one fails only here."""
    status = tc.probe_availability([key])[key]
    assert status == "ok", status


@needs_network
def test_realised_mixture_matches_the_requested_one():
    items = tc.text_items(300, window=256, mix=tc.DEFAULT_MIX, seed=5)
    assert len(items) == 300
    realised = tc.realised_mixture(items)
    for key, want in tc.DEFAULT_MIX.items():
        assert abs(realised.get(key, 0.0) - want) <= 0.02, realised


@needs_network
@pytest.mark.parametrize("seed", [0, 7, 13])
def test_streamed_gutenberg_has_no_licence_boilerplate(seed):
    """The claim is about the *licence wrapper*, not about the phrase. A
    transcriber's note saying an HTML edition exists is part of the text and
    should survive; the trademark notice and the 19 kB licence should not."""
    doc = next(tc.stream_documents("gutenberg", limit=1, seed=seed))
    upper = doc.upper()
    for marker in (
        "START OF THIS PROJECT GUTENBERG EBOOK",
        "START OF THE PROJECT GUTENBERG EBOOK",
        "END OF THIS PROJECT GUTENBERG EBOOK",
        "END OF THE PROJECT GUTENBERG EBOOK",
        "PROJECT GUTENBERG LICENSE",
        "START: FULL LICENSE",
        "WWW.GUTENBERG.ORG/LICENSE",
        "REDISTRIBUTING PROJECT GUTENBERG",
    ):
        assert marker not in upper, f"seed {seed} kept {marker!r}"
    assert len(doc) > 5000, "the cleaner ate the book"


@needs_network
def test_cleaning_removes_a_minority_of_each_book():
    """A sanity bound in both directions: a book that loses nothing kept its
    licence, and one that loses most of itself was truncated by a false match."""
    from datasets import load_dataset
    spec = tc.SOURCES["gutenberg"]
    ds = load_dataset(spec.dataset, spec.config, split=spec.split, streaming=True)
    ds = ds.shuffle(seed=7, buffer_size=spec.shuffle_buffer)
    for i, row in zip(range(4), ds):
        raw = row[spec.text_field]
        kept = len(tc.strip_gutenberg_boilerplate(raw)) / len(raw)
        assert 0.4 <= kept < 1.0, f"doc {i} kept {kept:.1%} of its bytes"


@needs_network
def test_the_splits_share_no_documents():
    """The regression this guards: a streaming shuffle permutes within a buffer
    while traversing the file in order, so seeding the splits differently left
    a measured 10% of the test set inside the training set — and bits per byte
    then reports memorisation as generalisation."""
    pulls = {
        s: set(tc.stream_documents("fineweb_edu", limit=40, seed=0, split=s))
        for s in tc.SPLIT_SHARES
    }
    assert all(len(v) == 40 for v in pulls.values()), {k: len(v) for k, v in pulls.items()}
    names = list(pulls)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (pulls[a] & pulls[b]), f"{a} and {b} share documents"


@needs_network
def test_the_mixture_holds_within_every_split():
    for split in ("train", "test"):
        items = tc.text_items(100, mix=tc.DEFAULT_MIX, seed=0, split=split)
        realised = tc.realised_mixture(items)
        for key, want in tc.DEFAULT_MIX.items():
            assert abs(realised.get(key, 0.0) - want) <= 0.03, (split, realised)


@needs_network
def test_limit_counts_documents_yielded_not_rows_scanned():
    """With a split filter in play, counting scanned rows would return a
    fraction of what was asked for, silently."""
    docs = list(tc.stream_documents("fineweb_edu", limit=25, seed=3, split="test"))
    assert len(docs) == 25


@needs_network
def test_items_carry_their_source_for_the_manifest():
    items = tc.text_items(40, window=256, mix={"fineweb_edu": 1.0}, seed=1)
    assert items and all(i.truth["source"] == "fineweb_edu" for i in items)
    assert all(i.family == "text_lm" for i in items)


# ---------------------------------------------------------------------------
# how text_lm is scored
# ---------------------------------------------------------------------------

def _reduction(bpb: float) -> float:
    """The scaling grade_text_lm applies, recomputed here independently."""
    return max(0.0, min(1.0, 1.0 - bpb / 8.0))


@pytest.mark.parametrize("bpb,expected", [
    (8.0, 0.0), (8.5, 0.0), (7.9, 0.0125), (4.0, 0.5), (1.0, 0.875), (0.0, 1.0),
])
def test_entropy_reduction_is_scaled_not_thresholded(bpb, expected):
    """A binary "did it beat 8 bits" puts 1.000 in the same column where 1.000
    means every answer was right, so 7.9 bits per byte — very nearly knowing
    nothing — reads as a perfect score. The scaled figure cannot be misread."""
    assert abs(_reduction(bpb) - expected) < 1e-9


def test_a_barely_better_than_uniform_model_does_not_look_perfect():
    assert _reduction(7.9) < 0.05


def test_scoring_matches_the_harness():
    from iridium.evaluation import harness
    import inspect
    src = inspect.getsource(harness.grade_text_lm)
    assert "1.0 - bpb / 8.0" in src, "the harness no longer scales the way this test assumes"
    assert "entropy_reduction" in src
