"""Strict compute-device validation with no fallback."""

import os
from pathlib import Path

from climbtrack.config import Device
from climbtrack.errors import DeviceUnavailableError


def configure_ultralytics(cache_root: Path) -> None:
    """Keep Ultralytics settings inside the configured project cache."""
    config_root = cache_root / ".ultralytics-config"
    (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    matplotlib_root = cache_root / ".matplotlib"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_root)


def require_torch_device(device: Device) -> str:
    """Return the configured device only if PyTorch can actually use it."""
    import torch

    if device is Device.MPS and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise DeviceUnavailableError(
            "Configured device 'mps' is unavailable "
            f"(PyTorch MPS built={built}). No CPU fallback was used."
        )
    if device is Device.CUDA and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "Configured device 'cuda' is unavailable. No CPU fallback was used."
        )
    return device.value


def seed_torch(seed: int) -> None:
    """Seed PyTorch inference state without changing the requested device."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
