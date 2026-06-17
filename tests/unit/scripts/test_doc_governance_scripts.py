from datetime import datetime
from pathlib import Path
import sys

import pytest

from scripts import check_docs_freshness, doc_check


def write_doc(path: Path, front_matter: str, body: str = "# Doc\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{front_matter}---\n\n{body}", encoding="utf-8")


def test_doc_check_default_cli_blocks_active_docs_without_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_root = tmp_path / "docs"
    write_doc(
        docs_root / "guide.md",
        "status: active\naudience: Developers\nlast-updated: 2026-06-10\n",
    )
    write_doc(
        docs_root / "README.md",
        "status: active\naudience: Developers\nlast-updated: 2026-06-10\nowner: Docs Owner\n",
    )

    monkeypatch.setattr(doc_check, "DOCS_ROOT", docs_root)
    monkeypatch.setattr(sys, "argv", ["doc_check.py"])

    with pytest.raises(SystemExit) as excinfo:
        doc_check.main()

    assert excinfo.value.code == 1


def test_freshness_uses_front_matter_last_updated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_root = tmp_path / "docs"
    write_doc(
        docs_root / "fresh.md",
        "status: active\naudience: Developers\nlast-updated: 2026-06-10\nowner: Docs Owner\n",
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.fromisoformat("2026-06-10T12:00:00+08:00")

    monkeypatch.setattr(check_docs_freshness, "datetime", FixedDateTime)
    monkeypatch.setattr(
        check_docs_freshness,
        "get_git_last_commit_date",
        lambda _path: datetime.fromisoformat("2026-03-01T00:00:00+08:00"),
    )

    stale_files = check_docs_freshness.check_freshness(docs_root, days_threshold=90)

    assert stale_files == []
