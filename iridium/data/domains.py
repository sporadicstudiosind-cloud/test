"""What Iridium-50m trains on, split by the subject each stack specialises in.

Each family is one shard (or one on-demand generator) and belongs to at most one
subject. A window's subject supervises the router, through the ``subject_routing``
loss already in :meth:`iridium.model.iridium1.Iridium1.losses`, towards that
subject's superstack. Only the media families have no subject.

No torch and no numpy here, so presets can import it with the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Stack order: 1 maths, 2 chat, 3 code, 4 science. "chat" is everything
#: conversational and written: stories, how-tos, small talk, saying "I don't know".
SUBJECTS = ("math", "chat", "code", "science")


@dataclass(frozen=True)
class DomainFamily:
    subject: Optional[str]
    #: Exactly one of these names the sources behind a prepared shard.
    text_mix: Optional[dict[str, float]] = None
    chat_mix: Optional[dict[str, float]] = None


DOMAIN_FAMILIES: dict[str, DomainFamily] = {
    "science_text": DomainFamily("science", text_mix={
        "wikipedia": 0.40, "fineweb_edu": 0.30, "cosmopedia_textbooks": 0.30}),
    "math_text": DomainFamily("math", text_mix={"finemath": 0.70, "openwebmath": 0.30}),
    "writing_text": DomainFamily("chat", text_mix={
        "cosmopedia_stories": 0.80, "cosmopedia_wikihow": 0.20}),
    # Short questions with short answers are what the chat-36m quiz showed missing.
    "science_qa": DomainFamily("science", chat_mix={
        "nq_open": 0.45, "openstax_requests": 0.45, "arc": 0.10}),
    "math_qa": DomainFamily("math", chat_mix={"openr1_math": 0.85, "gsm8k": 0.15}),
    "writing_qa": DomainFamily("chat", chat_mix={"story_requests": 1.0}),
    "code_qa": DomainFamily("code", chat_mix={
        "self_oss_instruct": 0.55, "tested_python": 0.22, "python_codes": 0.20, "mbpp": 0.03}),
    "chat_general": DomainFamily("chat", chat_mix={
        "smol_smoltalk": 0.70, "oasst": 0.20, "dolly": 0.10}),
}

#: On-demand generators: exactly graded, unlimited, and never stored in shards.
GENERATED_SUBJECT: dict[str, Optional[str]] = {
    "arithmetic_drill": "math",
    "code_exec": "code",
    "agent_episode": "code",
    "unknowable": "chat",
    "false_premise": "chat",
    "image_foundation": None,
    "audio_foundation": None,
}


def subject_of(family: str) -> Optional[str]:
    spec = DOMAIN_FAMILIES.get(family)
    return spec.subject if spec else GENERATED_SUBJECT.get(family)


#: Sources the tokenizer learns from. Weighted towards code and maths because a
#: vocabulary trained only on prose splits ``def``, ``==`` and JSON into bytes.
TOKENIZER_SOURCES = {"fineweb_edu": 0.30, "wikipedia": 0.15, "cosmopedia_stories": 0.15,
                     "finemath": 0.20, "python_code_text": 0.20}
