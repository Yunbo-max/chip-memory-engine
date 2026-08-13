import json

import pytest

from chip_memory.loader import ChipLoader, ChipStore
from chip_memory.types import Layer


def test_loads_extended_and_classic_chip_variants(example_chips):
    loader = ChipLoader()
    extended = loader.load(example_chips / "example_coordination.chip.json")
    classic = loader.load(example_chips / "example_negative_transfer.chip.json")

    assert extended.chip_id == "EXAMPLE_coordination"
    assert classic.chip_id == "EXAMPLE_negative_transfer"
    assert extended.layer_counts() == {"L1": 7, "L2": 2, "L3": 2}
    assert classic.layer_counts() == {"L1": 4, "L2": 1, "L3": 4}
    assert all(item.source_path for item in (*extended.items, *classic.items))


def test_structure_events_are_l1_and_other_events_are_l3(example_chips):
    chip = ChipLoader().load(example_chips / "example_coordination.chip.json")
    structure = next(item for item in chip.items if item.kind == "Structure")
    episode = next(item for item in chip.items if item.kind == "CollaborationEpisode")

    assert structure.layer is Layer.L1
    assert episode.layer is Layer.L3
    assert {"planner", "critic", "projector"}.issubset(episode.references)


def test_loader_never_changes_source_file(example_chips):
    path = example_chips / "example_coordination.chip.json"
    before = path.read_bytes()
    ChipLoader().load(path)
    assert path.read_bytes() == before


def test_validation_reports_bad_json(tmp_path):
    path = tmp_path / "bad.chip.json"
    path.write_text("{not-json", encoding="utf-8")
    report = ChipLoader().validate(path)
    assert report["valid"] is False
    assert report["errors"]


def test_duplicate_chip_ids_are_rejected(example_chips):
    chip = ChipLoader().load(example_chips / "example_coordination.chip.json")
    with pytest.raises(ValueError, match="Duplicate chip_id"):
        ChipStore([chip, chip])


def test_compatibility_schema_is_valid_json(project_root):
    schema = json.loads((project_root / "schema" / "chip.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert "node" in schema["$defs"]


def test_single_object_event_is_normalized(tmp_path):
    path = tmp_path / "single-event.chip.json"
    path.write_text(
        json.dumps(
            {
                "chip_id": "single-event",
                "title": "Single event compatibility",
                "method_mechanism": {"summary": "test"},
                "nodes": [{"id": "parent", "kind": "Method", "text": "parent"}],
                "edges": [],
                "events": {"id": "structure", "kind": "Structure", "parent": "parent", "children": []},
            }
        ),
        encoding="utf-8",
    )
    chip = ChipLoader().load(path)
    assert any(item.kind == "Structure" and item.layer is Layer.L1 for item in chip.items)
