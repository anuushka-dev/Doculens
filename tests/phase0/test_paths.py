from pathlib import Path

import pytest

from app.core.paths import find_project_root


def test_finds_root_from_nested_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").touch()

    nested = root / "app" / "core"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == root.resolve()


def test_finds_root_when_start_is_a_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").touch()

    source_file = root / "app.py"
    source_file.touch()

    assert find_project_root(source_file) == root.resolve()


def test_prefers_nearest_project_root(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "pyproject.toml").touch()

    inner = outer / "inner"
    inner.mkdir()
    (inner / "pyproject.toml").touch()

    nested = inner / "package"
    nested.mkdir()

    assert find_project_root(nested) == inner.resolve()


def test_raises_when_no_project_root_exists(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan"
    orphan.mkdir()

    with pytest.raises(
        RuntimeError,
        match=r"pyproject\.toml was not found",
    ):
        find_project_root(orphan)