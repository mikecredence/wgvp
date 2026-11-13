"""One-line install check for the GPU stage: tells exactly what is missing and how to fix it.
    python -m wgvp.check_gpu --gvhmr-root /path/to/GVHMR            (run INSIDE the GVHMR environment)
    wgvp check-gpu --gvhmr-root ... --gvhmr-python /path/to/env/python   (runs the check in that interpreter)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

WEIGHTS = ["inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
           "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt", "inputs/checkpoints/yolo/yolov8x.pt",
           "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"]


def check(gvhmr_root: str | None) -> list[tuple[str, bool, str]]:
    rows = []
    def add(name, ok, hint=""):
        rows.append((name, bool(ok), hint))
    add("python >= 3.10", sys.version_info >= (3, 10), f"found {sys.version.split()[0]}")
    add("nvidia-smi on PATH", shutil.which("nvidia-smi") is not None, "install the NVIDIA driver")
    try:
        import torch
        add("torch importable", True, torch.__version__)
        add("torch CUDA build", torch.version.cuda is not None, f"torch.version.cuda={torch.version.cuda}; install a CUDA build (see env/install_gvhmr.md)")
        add("CUDA device visible", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible to torch (driver / CUDA_VISIBLE_DEVICES)")
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            add("GPU memory >= 8 GB", mem >= 8, f"{mem:.1f} GB")
    except Exception as e:
        add("torch importable", False, f"{e}; pip install torch (CUDA build)")
    for mod, hint in (("pytorch3d", "pip install from source: see env/install_gvhmr.md"), ("pytorch_lightning", "pip install pytorch-lightning==2.4.0"),
                      ("hydra", "pip install hydra-core==1.3.2"), ("ultralytics", "pip install ultralytics"), ("smplx", "pip install smplx"),
                      ("einops", "pip install einops"), ("timm", "pip install timm==0.9.0"), ("imageio", "pip install imageio"), ("cv2", "pip install opencv-python")):
        try:
            __import__(mod); add(f"{mod} importable", True)
        except Exception as e:
            add(f"{mod} importable", False, f"{hint} ({type(e).__name__})")
    try:
        import ffmpeg
        add("ffmpeg-python (not the 'ffmpeg' stub)", hasattr(ffmpeg, "input"), "rm -rf site-packages/ffmpeg; pip install --force-reinstall --no-deps ffmpeg-python==0.2.0")
    except Exception as e:
        add("ffmpeg-python importable", False, f"pip install ffmpeg-python==0.2.0 ({e})")
    add("ffmpeg binary on PATH", shutil.which("ffmpeg") is not None, "apt/brew/conda install ffmpeg")
    if gvhmr_root:
        root = Path(gvhmr_root)
        add("GVHMR root exists", root.exists(), str(root))
        add("tools/demo/demo.py present", (root / "tools/demo/demo.py").exists(), "git clone https://github.com/zju3dv/GVHMR")
        if (root / "tools/demo/demo.py").exists():
            add("demo.py render guard (GVHMR_RENDER)", "GVHMR_RENDER" in (root / "tools/demo/demo.py").read_text(encoding="utf-8"), "optional: without it QC videos are rendered (slower) and deleted afterwards")
        for w in WEIGHTS:
            p = root / w
            add(f"weight {w}", p.exists(), f"{p.stat().st_size / 2**20:.0f} MB" if p.exists() else "download per GVHMR docs/INSTALL.md (SMPL/SMPL-X need registration at smpl-x.is.tue.mpg.de)")
        sys.path.insert(0, str(root))
        try:
            import hmr4d  # noqa
            from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa
            add("hmr4d importable from root", True)
        except Exception as e:
            add("hmr4d importable from root", False, f"cd {root} && pip install -e . ({type(e).__name__}: {e})")
        try:
            free = shutil.disk_usage(root).free / 2**30
            add("free disk >= 10 GB at GVHMR root", free >= 10, f"{free:.1f} GB free")
        except Exception:
            pass
    else:
        add("GVHMR root given", False, "pass --gvhmr-root")
    try:
        import psutil
        add("RAM >= 16 GB", psutil.virtual_memory().total >= 16 * 2**30, f"{psutil.virtual_memory().total / 2**30:.0f} GB")
    except Exception:
        pass
    return rows


def format_rows(rows) -> str:
    out = []
    for name, ok, hint in rows:
        out.append(f"  [{'OK ' if ok else 'MISSING'}] {name:<45} {hint}")
    n_bad = sum(1 for r in rows if not r[1])
    out.append("GPU stage READY" if n_bad == 0 else f"GPU stage NOT ready: {n_bad} item(s) missing (see hints above)")
    return "\n".join(out)


def run_in_interpreter(python: str, gvhmr_root: str | None) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUTF8="1")
    cmd = [python, "-m", "wgvp.check_gpu"] + (["--gvhmr-root", gvhmr_root] if gvhmr_root else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout + p.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gvhmr-root", default=None)
    a = ap.parse_args(argv)
    rows = check(a.gvhmr_root)
    print(format_rows(rows))
    return 0 if all(r[1] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
