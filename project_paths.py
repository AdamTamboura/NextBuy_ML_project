from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    raise FileNotFoundError(
        f"Unable to locate project root from '{current}'. Expected a '.git' directory."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def raw_data_file(filename: str) -> Path:
    return RAW_DATA_DIR / filename
