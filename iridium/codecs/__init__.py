"""Omnimodal codecs: everything becomes a token in one metric space.

A codec is an Iridium-1 component — trained in the same checkpoint, listed in
the same manifest, never a bolted-on pretrained encoder. See
docs/architecture.md §1.2 for why that boundary is drawn where it is.
"""

from .spans import (
    MODALITIES,
    MODALITY_INDEX,
    Batch,
    Sample,
    Span,
    collate,
    modality_id,
)
from .bank import CodecBank

__all__ = [
    "MODALITIES",
    "MODALITY_INDEX",
    "Batch",
    "Sample",
    "Span",
    "CodecBank",
    "collate",
    "modality_id",
]
