# Clarification Gate E2E Report

## Executive summary
The clarification gate is integrated into the long-form query flow through a wrapper in front of the existing orchestrator. Ambiguous dovolani queries stop before full retrieval, clear criminal/civil queries proceed, semantic reuse works, wrong family-law reuse is avoided, and a mixed CDO/TDO preview is blocked by the post-retrieval guard.

## Test matrix

| Scenario | Input type | Expected decision | Actual decision | cache_hit | semantic_reuse_hit | llm_called | retrieval_ran | PASS/FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original client-longform-04 | ambiguous dovolani narrative | ask_clarifying_question | ask_clarifying_question | False | True | False | False | PASS |
| Clear criminal query | criminal dovolani | proceed_to_retrieval | proceed_to_retrieval | False | False | False | True | PASS |
| Clear civil query | civil dovolani | proceed_to_retrieval | proceed_to_retrieval | False | False | False | True | PASS |
| Similar ambiguous query reuse | semantic-near ambiguous query | ask_clarifying_question | ask_clarifying_question | True | True | False | False | PASS |
| Family-law ambiguity avoids wrong reuse | family/remedy ambiguity | ask_clarifying_question | ask_clarifying_question | False | False | False | False | PASS |
| Post-retrieval mixed-domain guard | mixed preview fixture | ask_clarifying_question | ask_clarifying_question | False | False | False | True | PASS |

## Example output for original client-longform-04

- Decision: `ask_clarifying_question`
- Returned answer: `Jedná se o trestní dovolání podle trestního řádu, nebo o civilní dovolání podle občanského soudního řádu?`
- Retrieval ran: `False`
- LLM called: `False`

## Integration confirmations

1. Retrieval logic was not changed. The existing retrieval/ranking modules remain untouched; the gate only decides whether the orchestrator is allowed to continue.
2. Qdrant judgment retrieval remains unchanged. Court-judgment retrieval still belongs to the main Qdrant collection used by the existing retriever path.
3. Redis is used only for clarification cache/payloads. Court judgment embeddings are not moved into Redis; clarification cache wiring remains isolated in `app/rag/clarification/cache.py`.

## Final verdict

Yes.
The clarification gate is ready to protect long-form legal retrieval provided the current wrapper stays in front of `/api/rag/query`, the clarification cache remains scoped to payload reuse only, and the mixed-domain preview guard stays enabled.

## Scenario notes

### Original client-longform-04
- Detected legal domain: `unknown`
- Detected procedure stage: `dovolani`
- Reason: Dotaz je sémanticky podobný dříve známému nejednoznačnému vzoru; je potřeba upřesnit právní doménu před vyhledáváním.
- Preview hits: `none`
- Result: `PASS`

### Clear criminal query
- Detected legal domain: `criminal`
- Detected procedure stage: `dovolani`
- Reason: Dotaz obsahuje dostatečné právní ukotvení pro bezpečné vyhledávání.
- Preview hits: `ECLI:CZ:NS:2024:8.TDO.1022.2024.1, ECLI:CZ:NS:2025:4.TDO.1056.2024.1`
- Result: `PASS`

### Clear civil query
- Detected legal domain: `civil`
- Detected procedure stage: `dovolani`
- Reason: Dotaz obsahuje dostatečné právní ukotvení pro bezpečné vyhledávání.
- Preview hits: `ECLI:CZ:NS:2024:23.CDO.271.2024.1, ECLI:CZ:NS:2024:30.CDO.1111.2024.1`
- Result: `PASS`

### Similar ambiguous query reuse
- Detected legal domain: `unknown`
- Detected procedure stage: `appeal`
- Reason: Dotaz je sémanticky podobný dříve známému nejednoznačnému vzoru; je potřeba upřesnit právní doménu před vyhledáváním.
- Preview hits: `none`
- Result: `PASS`

### Family-law ambiguity avoids wrong reuse
- Detected legal domain: `family`
- Detected procedure stage: `appeal`
- Reason: Není jasné, zda má jít o návrh, odvolání, nebo jiný procesní krok.
- Preview hits: `none`
- Result: `PASS`

### Post-retrieval mixed-domain guard
- Detected legal domain: `civil`
- Detected procedure stage: `dovolani`
- Reason: Po prvním vyhledávání se top výsledky míchají mezi civilní a trestní judikaturou.
- Preview hits: `ECLI:CZ:NS:2024:23.CDO.271.2024.1, ECLI:CZ:NS:2024:8.TDO.1022.2024.1, ECLI:CZ:NS:2025:4.TDO.1056.2024.1`
- Result: `PASS`
