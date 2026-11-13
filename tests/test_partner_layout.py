"""The folder scan on a partner-style layout: one folder per subject, the video under Video/ with the task token in
its name, the recording export under IMU/. The default rules must pair them without any config edit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.config import load_config, write_default_config   # noqa: E402
from wgvp.discovery import scan_root                        # noqa: E402


def test_scan_pairs_video_and_csv_in_partner_layout(tmp_path):
    for sub in ("Sub 001", "Sub 002", "Sub 003"):
        (tmp_path / sub / "Video").mkdir(parents=True); (tmp_path / sub / "IMU").mkdir()
        (tmp_path / sub / "Video" / "SP_combo.mp4").write_bytes(b"")
        (tmp_path / sub / "IMU" / f"{sub.replace(' ', '')}_selfpace.csv").write_text("Time,GeneralEvent\n", encoding="utf-8")
    (tmp_path / "Sub 002" / "Video" / "notes.txt").write_text("ignored", encoding="utf-8")
    write_default_config(tmp_path); man = scan_root(tmp_path, load_config(tmp_path))
    ids = sorted(s["session_id"] for s in man["sessions"]); assert ids == ["Sub 001", "Sub 002", "Sub 003"]
    for s in man["sessions"]:
        assert len(s["recordings"]) == 1 and len(s["videos"]) == 1
        r = s["recordings"][0]; assert r["family"] == "selfpace" and r["recording_id"].endswith("_selfpace")
        assert [Path(v["path"]).name for v in r["videos"]] == ["SP_combo.mp4"] and not r.get("flag")
    assert not [u for u in man["unmatched"] if u["path"].endswith(".mp4")]
