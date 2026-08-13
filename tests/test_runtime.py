import json

import pytest

from chip_memory.engine import ChipMemoryEngine
from chip_memory.runtime import RuntimeEventStore


def test_full_lifecycle_is_append_only_and_preserves_chips(example_chips, tmp_path):
    source = example_chips / "example_negative_transfer.chip.json"
    before = source.read_bytes()
    runtime_path = tmp_path / "runtime"
    engine = ChipMemoryEngine.from_paths([example_chips], runtime_path=runtime_path)

    context = engine.init_task_context(
        "Choose a coordination method under a one-round budget",
        agent_roles=["planner", "critic"],
    )
    result = engine.retrieve_memory(role="critic", token_budget=1000)
    agent_event_id = engine.add_agent_node(
        "critic-1",
        "The retrieved result is incompatible with the budget.",
        role="critic",
    )
    engine.move_memory_state("reject long debate", "budget respected", reward=True)
    saved = engine.save_task_context(True, "Avoided negative transfer")
    engine.backward(True)

    assert context.context_id == saved.context_id
    assert saved.completed and saved.success
    assert agent_event_id
    assert result.projection.hits
    assert source.read_bytes() == before

    runtime = RuntimeEventStore(runtime_path)
    event_types = [event["event_type"] for event in runtime.iter_events()]
    assert event_types == [
        "context_started",
        "retrieval",
        "agent_event",
        "state_transition",
        "task_completed",
        "backward",
    ]
    priors = runtime.feedback_priors()
    assert priors.completed_contexts == 1
    assert any(score > 0 for score in priors.item_scores.values())
    assert any(score > 0 for score in priors.chip_scores.values())


def test_runtime_file_contains_valid_json_lines(example_chips, tmp_path):
    engine = ChipMemoryEngine.from_paths([example_chips], runtime_path=tmp_path / "events.jsonl")
    engine.init_task_context("test task")
    engine.retrieve_memory("coordination")
    engine.save_task_context(False, "synthetic failure")
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_only_one_active_context_is_allowed(example_chips):
    engine = ChipMemoryEngine.from_paths([example_chips])
    engine.init_task_context("first")
    with pytest.raises(RuntimeError, match="already active"):
        engine.init_task_context("second")


def test_runtime_summary(example_chips, tmp_path):
    runtime = RuntimeEventStore(tmp_path / "runtime")
    engine = ChipMemoryEngine.from_paths([example_chips])
    engine.runtime_store = runtime
    engine.init_task_context("coordination")
    engine.retrieve_memory(role="planner")
    engine.save_task_context(True)
    summary = runtime.summary()
    assert summary["contexts"] == 1
    assert summary["event_counts"]["retrieval"] == 1


def test_existing_directory_with_dot_suffix_is_treated_as_directory(tmp_path):
    runtime_directory = tmp_path / "runtime.session"
    runtime_directory.mkdir()
    runtime = RuntimeEventStore(runtime_directory)
    runtime.append("test", "context", {"ok": True})
    assert runtime.path == runtime_directory / "usage_events.jsonl"
    assert runtime.path.is_file()
