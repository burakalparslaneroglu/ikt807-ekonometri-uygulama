from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_private_references_are_ignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "references_private/" in gitignore
    assert ".streamlit/secrets.toml" in gitignore


def test_no_real_data_files_are_in_repository_tree() -> None:
    forbidden_suffixes = {".csv", ".dta", ".xlsx", ".dat"}
    files = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and path.suffix.lower() in forbidden_suffixes
    ]
    assert files == []
