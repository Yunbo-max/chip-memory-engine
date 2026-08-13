from chip_memory.engine import ChipMemoryEngine
from chip_memory.loader import ChipStore


def _engine(example_chips):
    return ChipMemoryEngine(ChipStore.from_paths([example_chips]))


def test_retrieval_returns_all_three_layers(example_chips):
    result = _engine(example_chips).retrieve_memory(
        "planner critic with a one-round communication budget",
        role="critic",
        token_budget=1800,
    )
    layers = {hit.item.layer.value for hit in result.projection.hits}
    assert layers == {"L1", "L2", "L3"}
    assert result.candidate_chips[0][0] in {"EXAMPLE_coordination", "EXAMPLE_negative_transfer"}


def test_layer_filter_is_respected(example_chips):
    result = _engine(example_chips).retrieve_memory(
        "communication budget failure",
        role="verifier",
        layers=["L3"],
    )
    assert result.projection.hits
    assert {hit.item.layer.value for hit in result.projection.hits} == {"L3"}


def test_negative_transfer_is_blocked(example_chips):
    result = _engine(example_chips).retrieve_memory(
        "use debate under a one-round communication budget",
        role="critic",
    )
    findings = result.projection.findings
    assert any(finding.severity == "block" for finding in findings)
    assert any(finding.code == "explicit_incompatibility" for finding in findings)


def test_role_projection_changes_ranking(example_chips):
    engine = _engine(example_chips)
    critic = engine.retrieve_memory("coordination failure result implementation", role="critic")
    executor = engine.retrieve_memory("coordination failure result implementation", role="executor")

    critic_ids = [hit.item.item_id for hit in critic.projection.hits]
    executor_ids = [hit.item.item_id for hit in executor.projection.hits]
    assert critic_ids != executor_ids or [hit.score for hit in critic.projection.hits] != [
        hit.score for hit in executor.projection.hits
    ]


def test_projection_respects_token_budget(example_chips):
    result = _engine(example_chips).retrieve_memory(
        "planner critic coordination budget evaluation results",
        role="researcher",
        token_budget=350,
    )
    assert result.projection.estimated_tokens <= 350
    assert result.projection.omitted_hits > 0


def test_graph_expansion_returns_connected_context(example_chips):
    result = _engine(example_chips).retrieve_memory(
        "projector",
        role="planner",
        total_hit_limit=20,
    )
    assert result.diagnostics["graph_expansion_count"] > 0
    assert any(hit.graph_bonus > 0 for hit in result.projection.hits)

