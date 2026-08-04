from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class EnvironmentReport:
    python: str
    platform: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    ultralytics_available: bool
    timm_available: bool


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def _nvidia_smi_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    name = result.stdout.strip().splitlines()
    return name[0] if name else None


def inspect_environment() -> EnvironmentReport:
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = _nvidia_smi_name()
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return EnvironmentReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch_available=torch_version is not None,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        ultralytics_available=_module_version("ultralytics") is not None,
        timm_available=_module_version("timm") is not None,
    )


def print_environment_report() -> EnvironmentReport:
    report = inspect_environment()
    print("Environment")
    for key, value in asdict(report).items():
        print(f"  {key}: {value}")
    return report
