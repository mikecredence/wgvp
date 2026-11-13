"""Text files are read and written as UTF-8 by name, so a run behaves the same on a PC whose default is not UTF-8."""
import ast
from pathlib import Path

from wgvp.qc import tee_logger

PKG = Path(__file__).resolve().parents[1] / "wgvp"


def test_every_text_file_call_in_the_package_names_its_encoding():
    missing = []
    for f in sorted(PKG.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("write_text", "read_text"):
                missing.append(f"{f.name}:{node.lineno}")
            elif isinstance(fn, ast.Name) and fn.id == "open":
                mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
                if mode is None or (isinstance(mode, ast.Constant) and "b" not in str(mode.value)):
                    missing.append(f"{f.name}:{node.lineno}")
    assert missing == [], missing


def test_the_run_log_keeps_a_folder_name_in_any_script(tmp_path):
    log = tee_logger(tmp_path / "logs" / "run.log")
    log("root: C:/Users/Ren\u00e9e/\u88ab\u9a13\u8005 01")
    log.close()
    assert "Ren\u00e9e/\u88ab\u9a13\u8005 01" in (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
