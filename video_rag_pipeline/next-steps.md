Below is a kickoff packet you can hand to a lead agent / engineering team. It’s intentionally “overarching + actionable,” not overly specific on market/broker choices.

---

# Kickoff Packet: Video-to-Knowledge → Specs → Automated Trading System

## 1) Objective

Build an automated trading system whose strategy logic is grounded in a corpus of training videos (presentations/workshops). The system must enable coding agents to:

1. **Extract knowledge** from videos into structured, machine-usable artifacts
2. **Generate PRDs/specs** for strategies, signals, risk rules, and system components
3. **Implement and test** an automated trading platform (backtest → paper → live) using those artifacts as the single source of truth

Non-goal for this phase: selecting exact instruments/broker/modeling every edge case. The first goal is a repeatable pipeline from source material → build-ready knowledge.

---

## 2) High-level approach

Use a **two-layer knowledge system**:

### Layer A — Compiled Knowledge (preferred by agents)

Structured records extracted from transcripts/OCR that are easy to use for engineering:

* Strategy specs (templates)
* Signal definitions
* Risk/management rules
* Glossary / concepts
* Assumptions + decision log
* Evaluation/backtesting playbooks

### Layer B — Source RAG (evidence fallback)

RAG index over raw video-derived chunks:

* transcript chunks with timestamps
* OCR slide text
* “visual captions” (diagram descriptions)

Agents consult Layer A first, then use Layer B when they need nuance or to resolve ambiguity.

---

## 3) Inputs (from your video pipeline)

Assume the video processing pipeline outputs per video:

* `chunks.jsonl` (primary)

  * `{chunk_id, video_id, start, end, text, transcript, ocr_text?, visual_caption?, metadata}`
* Optional supporting files:

  * `transcript.json`, `scenes.json`, `ocr.json`, `captions.json`

**Important:** timestamps are kept internally for traceability and debugging, even if never surfaced to end users.

---

## 4) Outputs (what the “knowledge base” should become)

The system produces two categories of deliverables:

### A) Knowledge Base Artifacts (JSONL + Markdown)

Stored as versioned files and indexed for retrieval.

1. `concepts.jsonl`

* terms, definitions, examples, related terms

2. `signals.jsonl`

* operational definitions + required data + parameters + edge cases

3. `strategies.jsonl`

* strategy spec per strategy (see template below)

4. `risk_rules.jsonl`

* max loss/day, circuit breakers, sizing constraints, no-trade conditions

5. `decisions.jsonl`

* assumptions, constraints, “we chose X because…”

6. `evaluation_playbook.md`

* backtest methodology, slippage modeling expectations, walk-forward rules

7. `open_questions.jsonl`

* unresolved gaps agents must not guess on

### B) Engineering Specs (generated from artifacts)

1. System PRD (high level)
2. Per-module design docs (data ingest, signal engine, execution, risk, backtest)
3. Task backlog (epics/stories with acceptance criteria)
4. Test plan (unit + integration + simulation)

---

## 5) Knowledge extraction plan (from chunks → compiled artifacts)

### Step 1: Build the “raw evidence index” (Source RAG)

* Embed `chunks.jsonl` into a vector store
* Add keyword/BM25 index if available
* Support metadata filters (video_id/module/topic tags)

### Step 2: Create “compilation jobs”

Run LLM extraction jobs over chunks to populate structured records. Start with coarse passes; refine later.

**Jobs to run:**

1. **Concept extraction**

   * Identify terms and define them using retrieved context
2. **Strategy identification**

   * Detect strategy names/variants and group supporting chunks
3. **Signal extraction**

   * Pull any measurable constructs (thresholds, conditions, patterns) into operational definitions
4. **Rules extraction**

   * Risk, trade management, session filters, “don’t trade when…”
5. **Conflict detection**

   * If two sources disagree, create an “Open Question” record and do not auto-resolve

### Step 3: Normalize into strict templates

Convert extracted knowledge into canonical schemas so implementation is consistent.

---

## 6) Canonical templates (minimal, but enforceable)

### Strategy Spec (store as JSON)

* `strategy_id`
* `name`
* `description`
* `market_scope` (unknown allowed)
* `timeframe_scope` (unknown allowed)
* `prerequisites` (data feeds required)
* `setup_conditions` (what must be true before an entry is considered)
* `entry_trigger` (must be operational, not vibes)
* `order_placement` (market/limit/stop; can be TBD)
* `initial_stop`
* `take_profit`
* `trade_management`
* `invalidation`
* `no_trade_conditions`
* `parameters` (name, default, range)
* `failure_modes`
* `metrics` (what to evaluate)
* `evidence` (list of chunk_ids that support this)

**Rule:** if a field cannot be made operational, fill with `TBD` and add an Open Question. No guessing.

### Signal Spec (store as JSON)

* `signal_id`
* `name`
* `definition` (operational logic)
* `inputs` (trades, candles, L2, session calendar)
* `computation_frequency`
* `normalization`
* `edge_cases`
* `validation_tests`
* `evidence` (chunk_ids)

### Risk Rule Spec

* `rule_id`
* `type` (daily_loss, max_position, volatility_halt, time_filter, etc.)
* `condition`
* `action`
* `severity`
* `evidence`

---

## 7) Retrieval policy for coding agents

Agents must follow this retrieval discipline:

1. Search compiled artifacts first:

* strategy spec(s)
* signal specs
* risk rules
* glossary/decisions

2. If insufficient, search source chunks:

* pull 3–10 chunks max
* prefer chunks with OCR text when query is slide/diagram oriented

3. Rerank + dedupe:

* remove repeated intros/boilerplate
* avoid feeding low-quality OCR into context unless relevant

4. If evidence is weak:

* produce `open_questions` rather than inventing requirements

---

## 8) System build plan (staged delivery)

### Phase 0 — Knowledge platform MVP

**Goal:** Agents can retrieve grounded knowledge and compile specs.

Deliverables:

* Ingestion script: `chunks.jsonl` → embeddings/index
* Retrieval API: `/search`, `/context`
* Compilation jobs: generate `concepts/signals/strategies/risk_rules/open_questions`
* Minimal UI optional: internal search page for human review

### Phase 1 — Research + backtest framework

**Goal:** Implement signals/strategies in a deterministic backtest environment.

Deliverables:

* Data model + storage (raw → normalized bars/ticks)
* Signal engine library + unit tests
* Strategy engine (state machine)
* Backtest runner with slippage/fees hooks
* Experiment tracker (results per strategy/version)

### Phase 2 — Paper trading

**Goal:** Run strategies end-to-end with execution simulation or real paper account.

Deliverables:

* Execution engine integration (paper)
* Risk engine enforcement
* Monitoring/alerts
* Trade journal + replay (“why did it trade?”)

### Phase 3 — Live (guarded)

**Goal:** Controlled live trading with strict safety.

Deliverables:

* Kill switch + circuit breakers
* Permissioned deploy workflow
* Observability (latency, fill quality, PnL attribution)
* Runbooks

---

## 9) Repo structure (recommended)

```
trading-system/
  knowledge/
    raw/            # chunks.jsonl etc.
    compiled/       # strategies.jsonl, signals.jsonl, concepts.jsonl ...
    specs/          # PRD.md, module-designs/
  services/
    retrieval-api/  # vector search + rerank + filters
    ingestion-worker/
    compiler/       # extraction jobs producing compiled artifacts
  trading/
    data-ingest/
    signals/
    strategies/
    execution/
    risk/
    backtest/
  ops/
    docker-compose.yml
    monitoring/
  eval/
    queries.jsonl
    golden_sets/
```

---

## 10) “Done means…” acceptance criteria (for the knowledge base)

A knowledge base is “usable for building” when:

* Given “Strategy X”, agents can retrieve:

  * a Strategy Spec + referenced Signal Specs + Risk Rules
  * with clear `evidence` pointers (chunk_ids)

* The system can generate:

  * PRD skeleton for the platform
  * design doc skeleton per module
  * implementation backlog (tasks w/ acceptance tests)

* No critical strategy requirement exists only “in agent memory.”
  Everything must appear in compiled artifacts or be flagged as an open question.

---

## 11) Initial task list (what you assign agents first)

1. **Set up storage + indexing**

* load `chunks.jsonl`
* embed + store in vector DB
* build `/search` + `/context` API

2. **Create compilation pipeline**

* strategy extraction job → `strategies.jsonl`
* signal extraction job → `signals.jsonl`
* glossary extraction job → `concepts.jsonl`
* open question detector → `open_questions.jsonl`

3. **Create a spec generator**

* reads compiled artifacts
* emits:

  * `System_PRD.md`
  * `Strategy_<name>_Spec.md`
  * `Signals_Catalog.md`
  * `Risk_Policy.md`

4. **Build evaluation harness**

* 30–50 real queries (“define absorption”, “entry criteria for X”)
* measure retrieval accuracy and spec completeness

---

## 12) Operating rules for agents (important)

* Don’t implement anything that isn’t operationally defined.
* If a concept is visual/discretionary, either:

  * translate to measurable logic, or
  * mark as manual input / research item.
* Always attach `evidence` chunk IDs to compiled records.
* Treat OCR as noisy: use it, but don’t trust it blindly.

---

If you want, paste 2–3 example `chunks.jsonl` lines (or one video’s `chunks.jsonl`) and I’ll:

* propose the exact JSON schemas (fields + examples),
* and a “compilation prompt pack” your agents can run to reliably generate `strategies/signals/risk_rules` from your content.

