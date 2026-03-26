"""Shared PyTorch device selection with runtime verification."""

from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    """Return a verified CUDA device or fall back to CPU.

    torch.cuda.is_available() can return True even when the installed PyTorch
    build does not support the installed GPU's compute capability (e.g. TITAN
    Xp sm_61 vs PyTorch requiring sm_70+).  This function probes with a tiny
    tensor operation so incompatible GPUs silently fall back to CPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:  # noqa: BLE001
        return torch.device("cpu")
