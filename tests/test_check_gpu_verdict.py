"""The GPU check's last line: only GPU rows missing means the lift runs on the CPU, a lift row missing means not
ready, and the rows that serve only GVHMR's own renders never decide it."""
from wgvp.check_gpu import ADVICE, GPU, LIFT, RENDER, format_rows, ready, verdict


def test_the_verdict_follows_the_lift_rows_only():
    ok = [("torch importable", True, "2.5.1", LIFT), ("hmr4d importable from root", True, "", LIFT)]
    gpu = [("CUDA device visible", False, "no GPU visible to torch", GPU), ("nvidia-smi on PATH", False, "", GPU)]
    extras = [("ffmpeg binary on PATH", False, "", RENDER), ("RAM >= 16 GB", False, "15.6 GiB", ADVICE)]
    assert verdict(ok) == "GPU stage READY" and ready(ok)
    assert verdict(ok + extras) == "GPU stage READY" and ready(ok + extras)
    assert verdict(ok + gpu + extras).startswith("GPU stage READY for the lift on the CPU, slowly (2 GPU item(s) missing)") and ready(ok + gpu)
    bad = ok + [("weight inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", False, "download", LIFT)]
    assert verdict(bad + gpu) == "GPU stage NOT ready: 1 item(s) missing for the lift (see hints above)" and not ready(bad)
    text = format_rows(ok + gpu + extras)
    assert text.index("needed for the lift") < text.index("the GPU;") < text.index("only GVHMR's own QC renders") < text.index("advice:")
    assert "[MISSING] CUDA device visible" in text and "[absent ] ffmpeg binary on PATH" in text and text.endswith("--lift-device cpu")
