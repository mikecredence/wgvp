"""The demo finds a recording's video from the session summary's bare file name, under the session's folder, as the
bundled example's deliverables (written on another machine) name it."""
import json
from pathlib import Path

from wgvp.demo import _video_path


def test_the_video_is_found_under_the_session_folder_from_its_name(tmp_path):
    root = tmp_path / "data"; sdir = root / "S01" / "SP"; sdir.mkdir(parents=True); (sdir / "S01_SP.mp4").write_bytes(b"x")
    deliver = root / "wgvp_output" / "S01" / "deliver"; deliver.mkdir(parents=True)
    (deliver / "session_summary.json").write_text(json.dumps({"dir": "/elsewhere/S01", "videos": {"S01_SP": {"file": "S01_SP.mp4"}},
                                                             "items": {"s01_selfpace": {"video_id": "S01_SP"}}}), encoding="utf-8")
    assert _video_path(root, deliver, "s01_selfpace") == sdir / "S01_SP.mp4"
    assert _video_path(root, deliver, "no_such_recording") is None
