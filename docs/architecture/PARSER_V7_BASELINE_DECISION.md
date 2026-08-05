# Parser v7 baseline decision

Status: **ACCEPT_V7_WITH_KNOWN_LIMITATIONS**

Date: 2026-08-05  
Parser profile: `legal-decision-parser.cz-courts.v7`  
Parser commit: `a53bf53c1904585f9bd9f81367971be2b43f3dbb`  
Decision recorded at HEAD: `1141cc0668a9af3280c1b679c48989a129cd876a` (docs baseline may advance)

Binding plan: [`NALUS_LEGAL_RAG_MASTER_PLAN.md`](./NALUS_LEGAL_RAG_MASTER_PLAN.md)

---

## Decision

```text
ACCEPT_V7_WITH_KNOWN_LIMITATIONS
```

Not:

```text
FIX_V8
```

Parser tuning for non-blocking internal label noise is **stopped**. Further parser
changes require a demonstrated impact on chunk boundaries, retrieval, context
assembly, or citation accuracy, measured against a frozen benchmark.

---

## Baseline evidence

| Check | Result |
|---|---|
| Exact goldens 05 / 11 / 16 | pass |
| Targeted structural regressions 06 / 07 / 10 / 14 / 17–20 | pass |
| Text conservation / duplication / ordering / parser exceptions | 0 failures |
| Manual decision store | unchanged |
| Manual history store | unchanged |
| V6 historical exports | unchanged |
| Snapshot | 20 documents / 1407 lines / 1387 boundaries / 629 blocks |

Full-review export (local / gitignored):

```text
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.json
artifacts/legal_v2/parser_v7_full_review/parser_v7_remaining_17_full.md
```

Recorded checksums at acceptance:

| File | Size | SHA-256 |
|---|---:|---|
| `parser_v7_remaining_17_full.json` | 4041484 | `A25D4299D13EA3A48CD02E4A3DE0E1D8DD94228A05C2019A20D87BCA66D1C6C1` |
| `parser_v7_remaining_17_full.md` | 1906773 | `B3D2A2BE12C28AEB9AA21C2BA76E42B127C7E5D2EFC552DB07D9E15379FB174D` |

---

## Known limitations

### KNOWN-PARSER-001

```text
KNOWN-PARSER-001
Closing location/date without the word "dne" may be classified as heading
instead of metadata. Boundaries and source reconstruction remain correct.
Retrieval impact: none currently demonstrated.
```

Example:

- Document review index: `08`
- Source ID: `3-1566-26_1`
- Line text: `V Brně 1. července 2026`
- Observed class: `heading`
- Expected preferred class: `metadata`
- Boundaries before/after: correct
- Independent block: yes
- Does not split or merge legal units
- Does not block exact citation

Action: document only. Do **not** open parser v8 for this label.

### Export/tooling backlog (not parser v8)

Deferred non-parser tooling issues:

- export metadata may show a stale pre-v7 commit identity;
- some export field names still say `v5`/`v6` while meaning previous/current;
- baseline labeling in prose may be imprecise.

These are audit-exporter defects. Fix later as a small tooling task when export
infrastructure is needed for new benchmarks.

---

## Current state

```text
Parser v7: ACCEPTED AS BASELINE
Parser tuning: STOP
Known limitation: KNOWN-PARSER-001 documented
Push/merge: allowed after ordinary Git review; not required by this decision
Next phase: Phase 1 archetypes + locked holdout, then benchmark-driven RAG work
```

---

## Next step

Create and maintain:

```text
docs/architecture/parser_benchmark/archetypes_v1.json
artifacts/legal_v2/parser_benchmark/archetypes_v1.json
```

Then continue with canonical block/chunk schema and the retrieval golden
(100–150 span-level queries). Do not resume broad parser polishing first.
