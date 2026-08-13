import json

from chip_memory.cli import main


def test_validate_cli(example_chips, tmp_path):
    output = tmp_path / "validation.json"
    code = main(["validate", "--chips", str(example_chips), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["summary"]["valid"] == 2


def test_build_index_cli(example_chips, tmp_path):
    output = tmp_path / "index.json"
    code = main(["build-index", "--chips", str(example_chips), "--output", str(output)])
    index = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert index["chip_count"] == 2
    assert index["item_count"] == 20


def test_query_cli_json(example_chips, tmp_path):
    output = tmp_path / "query.json"
    code = main(
        [
            "query",
            "--chips",
            str(example_chips),
            "--query",
            "one-round debate budget",
            "--role",
            "critic",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert result["role"] == "critic"
    assert result["projection"]["hits"]

