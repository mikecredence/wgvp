import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.config import load_config, DEFAULT_CONFIG  # noqa: E402
from wgvp.discovery import scan_root  # noqa: E402


def test_scan_conventions(tmp_path):
    root = tmp_path
    s1 = root / "study" / "NLS036"; (s1 / "SP").mkdir(parents=True); (s1 / "TUG").mkdir()
    for n in ("NLS036_selfpace.csv", "NLS036_selfpace_mat.csv", "NLS036_tug.csv"):
        (s1 / n).write_text("Time\n")
    (s1 / "SP" / "NLS036_SP_DeID_90.mp4").write_bytes(b"x"); (s1 / "TUG" / "NLS036_TUG_DeID_90.mp4").write_bytes(b"x")
    (s1 / "NLS036_SP_labels.csv").write_text("Time,Walk\n0,0\n")
    s2 = root / "study" / "HC100"; (s2 / "selfpace").mkdir(parents=True)
    (s2 / "hc100_selfpace_matturn.csv").write_text("Time\n"); (s2 / "selfpace" / "visit1.mp4").write_bytes(b"x")
    (root / "stray.mp4").write_bytes(b"x"); (root / "study" / "readme.csv").write_text("a\n")
    man = scan_root(root, load_config(None))
    ids = [s["session_id"] for s in man["sessions"]]
    assert ids == ["study__HC100", "study__NLS036"]
    nls = man["sessions"][1]; recs = {r["recording_id"]: r for r in nls["recordings"]}
    assert set(recs) == {"NLS036_selfpace", "NLS036_selfpace_mat"}            # tug not in the task families
    assert [Path(v["path"]).name for v in recs["NLS036_selfpace"]["videos"]] == ["NLS036_SP_DeID_90.mp4"]
    assert recs["NLS036_selfpace"]["videos"][0]["tags"] == "DeID_90"
    assert [Path(x).name for x in nls["labels"]] == ["NLS036_SP_labels.csv"]
    hc = man["sessions"][0]
    assert hc["recordings"][0]["videos"] and Path(hc["recordings"][0]["videos"][0]["path"]).name == "visit1.mp4"   # task from the sub-folder name
    reasons = {u["path"]: u["reason"] for u in man["unmatched"]}
    assert "stray.mp4" in reasons and any("TUG_DeID" in k for k in reasons)


def test_config_roundtrip(tmp_path):
    from wgvp.config import write_default_config
    p = write_default_config(tmp_path); cfg = json.loads(p.read_text())
    cfg["tasks"] = {"tug": {"csv_tokens": ["tug"], "dir_names": ["TUG"], "video_tokens": ["TUG"]}}; p.write_text(json.dumps(cfg))
    c = load_config(tmp_path)
    assert "tug" in c["tasks"] and c["views"]["layout"] == DEFAULT_CONFIG["views"]["layout"]


def test_pick_by_point_resolves_against_current_tracking():
    """An operator pick given as frame + point finds the fragment under it whatever its id is in this run."""
    import numpy as np
    from wgvp.selection import pick_selection
    from wgvp.tracking import TrackSet
    F = 100; frames = np.arange(F)
    det_frame = np.concatenate([frames, frames]); det_tid = np.concatenate([np.full(F, 7), np.full(F, 9)])
    boxes = np.concatenate([np.tile([100, 50, 200, 400], (F, 1)), np.tile([400, 60, 480, 380], (F, 1))]).astype(float)
    ts = TrackSet(video="v", view_id="full", box=[0, 0, 640, 480], width=640, height=480, fps_nominal=30.0, frame_idx=frames, pts_ms=frames / 30.0 * 1000,
                  det_frame=det_frame, det_tid=det_tid, det_box=boxes, det_conf=np.ones(2 * F), det_kp=np.zeros((2 * F, 17, 3)), det_app=np.zeros((2 * F, 48)), det_vis=np.full(2 * F, 2, np.int8), grid=np.zeros((F, 18, 32), np.uint8))
    r = pick_selection({"points": [{"frame": 50, "x": 150, "y": 200}]}, ts); assert r["ok"] and r["track_ids"] == [7]
    r = pick_selection({"points": [{"frame": 50, "x": 150, "y": 200}, {"frame": 80, "x": 440, "y": 100}]}, ts); assert r["track_ids"] == [7, 9]
    r = pick_selection({"points": [{"frame": 50, "x": 300, "y": 200}]}, ts); assert not r["ok"] and "no point resolved" in r["reason"]
    r = pick_selection({"track_ids": [123]}, ts); assert not r["ok"]
    r = pick_selection({"track_ids": [9], "points": [{"frame": 50, "x": 150, "y": 200}]}, ts); assert r["track_ids"] == [7, 9]
