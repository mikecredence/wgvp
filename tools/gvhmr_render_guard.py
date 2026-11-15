"""Put the render guard into a GVHMR checkout's tools/demo/demo.py (env/install_gvhmr.md).

GVHMR's demo renders two QC videos after inference, a third of the wall time on a GPU and far more on a CPU. The
pipeline sets GVHMR_RENDER=0 for its runs; with this guard in place the script then stops after inference. Without
it the stage still works, slower, and deletes the videos afterwards. The change is three lines, applied once;
running this again does nothing.

    python tools/gvhmr_render_guard.py <GVHMR checkout>
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = """    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
"""
NEW = """    # ===== Render ===== #
    if os.environ.get("GVHMR_RENDER", "1") == "1":                     # wgvp: GVHMR_RENDER=0 skips the QC videos
        render_incam(cfg)
        render_global(cfg)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
"""


def guard(gvhmr_root: Path) -> str:
    f = Path(gvhmr_root) / "tools" / "demo" / "demo.py"
    if not f.is_file():
        raise FileNotFoundError(f"no tools/demo/demo.py under {gvhmr_root}")
    s = f.read_text(encoding="utf-8")
    if "GVHMR_RENDER" in s:
        return f"{f}: the guard is already there"
    if OLD not in s:
        raise RuntimeError(f"{f}: the render block does not look like GVHMR commit 088caff's; put the guard in by hand (env/install_gvhmr.md)")
    s = s.replace(OLD, NEW)
    if "\nimport os\n" not in s:
        s = s.replace("import cv2\n", "import os\nimport cv2\n", 1)
    f.write_text(s, encoding="utf-8")
    return f"{f}: render guard added"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    print(guard(Path(sys.argv[1])))
