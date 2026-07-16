from pathlib import Path
import tomllib

import quorum


def test_runtime_version_matches_project_metadata():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert quorum.__version__ == metadata["project"]["version"]