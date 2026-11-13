"""The first mistakes a new user makes get a sentence from the command line, not a traceback."""
import builtins
import json
from pathlib import Path

from wgvp import cli


def test_missing_data_folder_is_a_sentence(tmp_path, capsys):
    for cmd in ("scan", "run", "summary", "gvhmr", "demo"):
        assert cli.main([cmd, str(tmp_path / "nope")]) == 2
        assert "data folder not found" in capsys.readouterr().out


def test_scan_of_an_empty_folder_says_what_was_expected(tmp_path, capsys):
    assert cli.main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "sessions: 0" in out and "one folder per subject" in out and "wgvp_config.json" in out


def test_gui_without_tkinter_names_the_fix(monkeypatch, capsys):
    real_import = builtins.__import__

    def no_tk(name, *a, **k):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("No module named '_tkinter'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tk)
    assert cli.main([]) == 1
    out = capsys.readouterr().out
    assert "python-tk@" in out and "python3-tk" in out and "python -m wgvp run" in out


def test_gvhmr_with_nothing_configured_says_why(tmp_path, capsys):
    assert cli.main(["gvhmr", str(tmp_path)]) == 1
    assert "GPU stage not run: no GVHMR root configured" in capsys.readouterr().out


def test_summary_before_any_run_says_what_to_run(tmp_path, capsys):
    assert cli.main(["summary", str(tmp_path)]) == 1
    assert "nothing has been run" in capsys.readouterr().out


def test_qc_page_puts_the_reason_beside_the_status(tmp_path):
    from wgvp.summary import build_summary
    d = tmp_path / "wgvp_output" / "S1" / "deliver"; d.mkdir(parents=True)
    (d / "session_summary.json").write_text(json.dumps({"session_id": "S1", "status": "WARN", "items": {
        "R1": {"sync_flags": ["flash_disagrees (+0.616s)"]}, "R1/left": {"status": "WARN", "video": "S1_SP.mp4"}}}))
    out = build_summary(tmp_path, {}, gvhmr={"ran": False, "message": "GPU stage not run: no GVHMR root configured"},
                        log=lambda m: None, thumbnails=False)
    page = Path(out["html"]).read_text()
    assert "background:#fff" in page                       # readable whatever the browser's own canvas colour is
    assert page.index("<th>status</th>") < page.index("<th>flags</th>") < page.index("<th>video</th>")
    assert "<td>WARN</td><td>flash_disagrees (+0.616s)</td>" in page
    assert "GPU stage: not run (no GVHMR root configured)" in page
