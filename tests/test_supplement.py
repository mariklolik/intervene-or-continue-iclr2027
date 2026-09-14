import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_supplement", ROOT / "tools" / "build_supplement.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_anonymous_text_removes_local_identity_and_paths():
    value = (
        "/home/mekashirskiy/intervene-or-continue-iclr2027/raw "
        "/Users/mekashirskiy/project/file.json avi-gn-fsk42 mariklolik AlekseiSDev"
    )
    result = MODULE.anonymous_text(value)
    assert "$PROJECT_ROOT/raw" in result
    assert "$LOCAL_ARTIFACT" in result
    assert "GPU_HOST" in result
    assert not any(item in result for item in MODULE.BANNED)


def test_deterministic_zip_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    MODULE.write_zip(source, first)
    MODULE.write_zip(source, second)
    assert first.read_bytes() == second.read_bytes()


def test_verify_supplement_checks_internal_manifest(tmp_path):
    source = tmp_path / "source"
    artifact = source / "paper" / "main.pdf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"pdf")
    manifest = source / "artifacts" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"anonymous_derivative": true, "result_values_preserved": true, "source_commit": "0123456789012345678901234567890123456789", "files": [{"path": "paper/main.pdf", "sha256": "c35b21d6ca39aa7cc3b79a705d989f1a6e88b99ab43988d74048799e3db926a3"}]}')
    archive = tmp_path / "supplement.zip"
    MODULE.write_zip(source, archive)
    result = MODULE.verify_archive(archive)
    assert result["files"] == 1
