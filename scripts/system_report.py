import json
import platform
import re
import subprocess
from pathlib import Path

import torch

from utils.config import PROJECT_ROOT


def amd_smi_gpu_name():
    try:
        output = subprocess.run(
            ["amd-smi"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    match = re.search(r"Instinct\s+MI\d+\w*", output, re.IGNORECASE)
    return match.group(0) if match else None


def build_report():
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_api_available": torch.cuda.is_available(),
        "rocm_version": getattr(torch.version, "hip", None),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        fallback_name = amd_smi_gpu_name()
        report["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index) or fallback_name or "AMD GPU",
                "memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024**3,
                    2,
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    return report


def main():
    report = build_report()
    output_dir = PROJECT_ROOT / "artifacts" / "system"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "amd_environment.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
