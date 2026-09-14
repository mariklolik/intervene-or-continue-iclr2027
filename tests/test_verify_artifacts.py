import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_artifacts", ROOT / "tools" / "verify_artifacts.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_main_text_page_accepts_numbered_or_empty_label():
    assert MODULE.main_text_page(r"\newlabel{main-end}{{8}{8}{AI use statement}{section*.2}{}}") == 8
    assert MODULE.main_text_page(r"\newlabel{main-end}{{}{9}{AI use statement}{section*.2}{}}") == 9
