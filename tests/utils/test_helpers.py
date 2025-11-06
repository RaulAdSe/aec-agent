"""Test utilities and helpers."""

import json
from pathlib import Path


def load_test_fixture(filename: str) -> dict:
    """Load test data from fixtures directory."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / filename
    with open(fixture_path, 'r') as f:
        return json.load(f)


def create_minimal_building() -> dict:
    """Create minimal building data for testing."""
    return {
        "metadata": {"project_name": "Test Building"},
        "rooms": [{"id": "R001", "area": 20.0}]
    }


def test_load_fixture():
    """Test fixture loading utility."""
    building = load_test_fixture("sample_building.json")
    assert building["metadata"]["project_name"] == "Sample Office Building"
    assert len(building["rooms"]) == 3


def test_create_minimal_building():
    """Test minimal building creation."""
    building = create_minimal_building()
    assert "metadata" in building
    assert "rooms" in building
    assert building["rooms"][0]["id"] == "R001"