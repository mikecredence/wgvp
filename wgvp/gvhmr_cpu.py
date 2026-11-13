"""Run a GVHMR script on a machine where torch sees no CUDA GPU.

GVHMR's demo moves its models and its batches to the GPU by name (`.cuda()`). This wrapper makes those moves leave
the data where it is and then runs the script unchanged, so the same code lifts on the CPU. It is a fallback for a
site with no NVIDIA GPU, and it is slow (README, "The lift without a GPU", has the measured time).

    python -m wgvp.gvhmr_cpu <script.py> [the script's own arguments]
"""
from __future__ import annotations

import runpy
import sys


def patch_torch():
    """Make every move-to-GPU a no-op and every checkpoint load land on the CPU."""
    import torch

    def stay(self, *a, **k):
        return self

    torch.Tensor.cuda = stay
    torch.nn.Module.cuda = stay
    for mod in ("lightning_fabric.utilities.device_dtype_mixin", "lightning.fabric.utilities.device_dtype_mixin",
                "pytorch_lightning.core.mixins.device_dtype_mixin"):      # Lightning modules carry their own .cuda()
        try:
            cls = getattr(__import__(mod, fromlist=["x"]), "_DeviceDtypeModuleMixin", None) or getattr(__import__(mod, fromlist=["x"]), "DeviceDtypeModuleMixin")
            cls.cuda = stay
        except Exception:
            pass
    torch.cuda.get_device_name = lambda *a, **k: "none (lifting on the CPU)"
    torch.cuda.get_device_properties = lambda *a, **k: "none (lifting on the CPU)"
    for name, value in (("synchronize", None), ("empty_cache", None), ("reset_peak_memory_stats", None), ("current_device", 0),
                        ("max_memory_allocated", 0), ("memory_allocated", 0)):          # timing and memory calls GVHMR makes
        setattr(torch.cuda, name, lambda *a, _v=value, **k: _v)
    load = torch.load

    def load_on_cpu(*a, **k):
        if len(a) < 2:
            k.setdefault("map_location", "cpu")
        return load(*a, **k)

    torch.load = load_on_cpu


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        print(__doc__); return 2
    patch_torch()
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
