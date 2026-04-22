Audit `data/conversations.log` for retrieval quality issues, normalization bugs, and anomalies.

## Cursor Management

1. Read `data/.log-audit-cursor` — it contains a single ISO timestamp (e.g. `2026-04-11T14:00:00`).
2. If the file exists and the timestamp is **< 48 hours old**, scan entries **from that timestamp onward**.
3. If the file is **missing** or the timestamp is **> 48 hours old**, scan the **last 48 hours** of entries.
4. After analysis completes, **update** (or create) `data/.log-audit-cursor` with the timestamp of the **last entry processed**.

## Log Format

Each line in `conversations.log`:
```
YYYY-MM-DD HH:MM:SS user=ID channel=ID q='raw' cleaned='cleaned' chunks=N pages=[...] scores=[...] model=MODEL latency_ms=N response='text'
```

Error entries use `chunks=-1 model=error` and include `error=ExcType: message`.

## Analysis Categories

Run ALL of the following checks on each entry in the scan range. Collect issues into a structured report.

### 1. Normalization Mismatch

The log's `cleaned` field does NOT reflect `_normalize_query()` from `rag/retriever.py` — that runs inside `retrieve()` after logging. To detect normalization bugs:

- Import `_normalize_query` from `rag.retriever`
- For each entry, run `_normalize_query(cleaned_value)` and compare to `cleaned_value`
- If they differ, check whether the **normalized form** plausibly matches the returned pages
- Flag entries where normalization rewrote the query but returned pages don't match the **original** intent (e.g. "demon lord haki" rewritten to "daemon lord haki" → returned Daemon Lord race pages instead of the Demon Lord Haki skill)

### 2. Wrong Page Retrieval

Flag entries where the query clearly asks about topic X but all returned pages are about topic Y:
- Extract the key entity/noun from the query
- Check if ANY returned page title contains that entity (case-insensitive, partial match OK)
- If zero pages match the query's key entity → flag as **wrong retrieval**

### 3. Low Confidence

Flag entries where the **best** score (first in list) is below `0.75`. These indicate the retriever found nothing relevant. Exclude `chunks=-1` error entries.

### 4. Repeated Failures

Group queries by normalized form (lowercase, strip trailing punctuation). If 2+ distinct users asked the same/similar question and ALL got the same wrong result → flag as **systematic failure** with user count.

### 5. Non-Question Noise

Flag queries that are greetings, tests, or not real questions:
- Patterns: `hello?`, `hi?`, `test?`, `you there?`, `anyone?`, `hey?`
- These waste API calls. Report count and suggest they could be filtered.

### 6. Errors

Report all `model=error` entries with their error type and message. Group by error type.

### 7. High Latency

Flag entries where `latency_ms > 10000` (10 seconds). Report query and model used.

### 8. Page Monoculture

Flag entries where ALL returned chunks come from the same page (and chunks > 3). This may indicate over-indexing of one page or a retrieval path that bypasses semantic search.

### 9. Score Cliff

Flag entries where there's a sharp drop between top scores and bottom scores (top score - bottom score > 0.15 with chunks > 5). This suggests the retriever is padding results with irrelevant chunks.

## Execution

Run the analysis using a Python script via Bash. Parse the log, apply all checks, collect results. Do NOT use subagents.

```python
# Use this pattern:
import re, json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# 1. Read cursor
# 2. Parse entries in range
# 3. Run all checks
# 4. Print structured report
# 5. Update cursor
```

## Report Format

Output a structured summary with sections for each category. Include:
- Total entries scanned and time range
- Issue counts per category
- Top 5 most impactful issues (most users affected or most queries)
- Specific examples with timestamps for each flagged issue
- Suggested Linear issues to create for systematic problems

Sort issues by severity: Normalization Mismatch and Repeated Failures first (they affect all users), then Wrong Retrieval, then the rest.

## After Analysis

- Ask the user if they want Linear issues created for any systematic problems found
- If normalization bugs are found, suggest specific fixes to `_RACE_SYNONYM_PHRASES` or `_normalize_query()`
