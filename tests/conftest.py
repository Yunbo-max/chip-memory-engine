from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def example_chips(project_root: Path) -> Path:
    return project_root / "examples" / "chips"

