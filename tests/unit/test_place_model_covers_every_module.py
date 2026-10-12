"""``place_model`` must move *every* shared tensor, not a list of names.

It used to move a hardcoded set of submodules (codecs, core, router, ...).
When the per-layer embedding table ``ple`` was added to Iridium1 it was not
on that list, so it stayed on the CPU while everything else went to the GPU,
and the first Kaggle training step died with "Expected all tensors to be on
the same device ... index is on cuda:0, different from other tensors on cpu".

No GPU is needed to catch this class of bug: the ``meta`` device stands in for
the accelerator, and anything left behind is still on ``cpu``.
"""

import pytest
import torch

from iridium.model.iridium1 import Iridium1
from iridium.presets import PRESETS, get_preset
from iridium.runtime.placement import place_model


def _stray(model, primary, stack_device):
    stacks = {id(t) for s in model.bank.stacks for t in list(s.parameters()) + list(s.buffers())}
    bad = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        want = stack_device if id(t) in stacks else primary
        if t.device.type != want:
            bad.append(f"{name} on {t.device}")
    return bad


# Materialising a preset costs 4 bytes per parameter; the multi-billion ones
# share the same module tree as the small ones, so they add nothing here.
SMALL = sorted(n for n, p in PRESETS.items() if p.config.n_params < 200_000_000)


@pytest.mark.parametrize("name", SMALL)
def test_every_preset_lands_entirely_on_the_primary_device(name):
    cfg = get_preset(name).config
    with torch.device("meta"):
        model = Iridium1(cfg)
    # Pull it back to the CPU, then place onto the stand-in accelerator.
    model = model.to_empty(device="cpu")
    place_model(model, ("meta",))
    assert _stray(model, "meta", "meta") == []
