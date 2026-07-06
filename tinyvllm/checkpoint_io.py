"""Safe checkpoint loading — handles legacy ClipConfig pickle paths."""

from __future__ import annotations

import pickle
from typing import Any

import torch

from tinyvllm.config import ClipConfig, Config


class _CheckpointUnpickler(pickle.Unpickler):
    """Redirect pickled config classes to their current module locations."""

    def find_class(self, module: str, name: str):
        if name == "ClipConfig":
            return ClipConfig
        if name == "Config":
            return Config
        return super().find_class(module, name)


class _PickleModule:
    Unpickler = _CheckpointUnpickler
    dump = pickle.dump
    dumps = pickle.dumps
    load = pickle.load
    loads = pickle.loads


def register_clip_config_shims() -> None:
    """Register ClipConfig on modules that old checkpoints may reference."""
    import tinyvllm.inference.match_text as match_text_mod
    import tinyvllm.train_clip as train_clip_mod

    match_text_mod.ClipConfig = ClipConfig
    train_clip_mod.ClipConfig = ClipConfig


def safe_torch_load(path: str, device: torch.device) -> dict[str, Any]:
    """Load a TinyVLLM checkpoint, including legacy ClipConfig pickles."""
    register_clip_config_shims()
    return torch.load(
        path,
        map_location=device,
        weights_only=False,
        pickle_module=_PickleModule,
    )
