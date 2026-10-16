from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root using pyproject.toml."""
    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise RuntimeError(
        "Could not locate the project root: pyproject.toml was not found."
    )


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
