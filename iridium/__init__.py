"""Iridium-1: a routed omnimodal model family, built to be trained on free compute.

Nothing is imported eagerly: ``import iridium`` stays cheap and torch-free, so
config accounting, presets and the CLI's costing commands work without a GPU
stack installed.
"""

__version__ = "1.0.0"
