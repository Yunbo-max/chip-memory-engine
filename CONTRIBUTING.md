# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The runtime package intentionally has no mandatory third-party dependencies.

## Design invariants

Changes should preserve these boundaries:

1. Paper `.chip.json` files are immutable source-grounded records.
2. Runtime observations are written only to the append-only runtime store.
3. L1, L2, and L3 retain their distinct semantics.
4. Returned claims retain Chip and item provenance.
5. Feedback is bounded and cannot silently rewrite source truth.
6. Schema compatibility must be covered by regression tests.

## Before submitting a change

Run:

```bash
pytest
python -m chip_memory validate --chips examples/chips
python -m chip_memory query \
  --chips examples/chips \
  --query "one-round multi-agent communication budget" \
  --role critic \
  --format json
```

New retrieval behavior should include tests for ranking, provenance, layer coverage, and context-budget behavior. New runtime behavior should demonstrate that the source Chip remains byte-for-byte unchanged.

## Research claims

Do not describe a design hypothesis as an empirical advantage without adding the corresponding controlled evaluation. Report failures and negative-transfer cases alongside successful examples.

