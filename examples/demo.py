"""Minimal end-to-end example using the bundled illustrative Chips."""

from pathlib import Path

from chip_memory import ChipMemoryEngine
from chip_memory.projector import projection_to_markdown


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    engine = ChipMemoryEngine.from_paths(
        [ROOT / "examples" / "chips"],
        runtime_path=ROOT / ".runtime",
    )
    engine.init_task_context(
        "Design a planner-critic team under a one-round communication budget",
        agent_roles=["planner", "critic", "verifier"],
    )
    result = engine.retrieve_memory(role="critic", token_budget=1500)
    print(projection_to_markdown(result.projection))
    engine.add_agent_node("critic", "The long-debate method conflicts with the task budget.", role="critic")
    engine.move_memory_state("reject incompatible transfer", "selected a lower-cost mechanism", reward=True)
    engine.save_task_context(True, "Negative-evidence gate prevented an incompatible transfer.")
    engine.backward(True)


if __name__ == "__main__":
    main()

