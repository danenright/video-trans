# Knowledge Platform Plan
## Automated Trading System from Video Education

**Created**: 2025-01-09
**Last Updated**: 2025-01-09
**Status**: Transcription in progress, RAG infrastructure ready

---

## 0. Current Status & Next Steps

### Where We Are

| Phase | Status | Details |
|-------|--------|---------|
| Video Transcription | **In Progress** | 45/408 videos (11%), ~211 chunks generated |
| Knowledge Platform Install | **Complete** | `kp-ingest`, `kp-serve` CLI tools ready |
| LanceDB Ingestion | **Pending** | Waiting for transcriptions to complete |
| Content Analysis | **Pending** | Read transcripts to understand structure |
| Extraction Approach | **Pending** | Design based on actual content review |

### Immediate Next Steps

**Step 1: Complete Transcriptions** (in progress)
```bash
# Monitor progress
find video_rag_pipeline/output -name "chunks.jsonl" | wc -l
```

**Step 2: Ingest into LanceDB**
```bash
source video_rag_pipeline/vrag-env/bin/activate
cd knowledge_platform

# Merge all chunks
cat ../video_rag_pipeline/output/*/chunks.jsonl > data/raw/all_chunks.jsonl

# Vectorize into LanceDB
kp-ingest data/raw/all_chunks.jsonl -v
```

**Step 3: Start RAG Server**
```bash
kp-serve
# API available at http://localhost:8000
# - POST /search {"query": "...", "top_k": 10}
# - GET /chunks/{id}
# - GET /stats
```

**Step 4: Review Content Structure**
- Read sample transcripts from theory units, strategy units, drills
- Understand how concepts vs strategies are actually presented
- Identify what's explicit vs implicit in the content

**Step 5: Design Extraction Approach**
- Based on actual content review (not assumptions)
- Determine what can be batch-extracted vs needs careful review
- Plan RAG-assisted extraction workflow

### Key Decision Made

We will **NOT** pre-design the extraction approach. Instead:
1. Get all content into RAG first
2. Read and understand the actual transcript content
3. Then design extraction based on evidence

This avoids building assumptions about "concepts" vs "strategies" without seeing the material.

### File Locations

```
video_rag_pipeline/
  videos/                    # Source videos (408 total)
  output/*/chunks.jsonl      # Transcribed chunks
  vrag-env/                  # Python virtual environment

knowledge_platform/
  data/raw/                  # Merged chunks go here
  data/vectordb/             # LanceDB storage
  kp/                        # Platform code
  config.yaml                # Configuration
```

---

## 1. Project Goal

Transform ~400 hours of trading education video into structured, codifiable knowledge that drives an automated trading system. The system should:

- Remove emotion from trading decisions
- Execute well-defined, probabilistically-favored setups
- Run as an "always-on" automated solution
- Be backtestable and verifiable

**Two outputs:**
1. **PRD Documents** - Human-readable specifications for system design
2. **Knowledge Base (RAG)** - Machine-queryable knowledge for engineering agents

---

## 2. System Architecture

```
video_rag_pipeline/                    knowledge_platform/
====================                   ===================

[MP4/AVI Videos]
       |
       v
+------------------+
| run_pipeline.py  |
| - transcribe     |
| - scene detect   |
| - chunk          |
+------------------+
       |
       v
[output/*/chunks.jsonl] ──────────────> [data/raw/]
                                              |
                                              v
                                       +-------------+
                                       | kp-ingest   |
                                       | (vectorize) |
                                       +-------------+
                                              |
                                              v
                                       [LanceDB Vector Store]
                                              |
                        +---------------------+---------------------+
                        |                     |                     |
                        v                     v                     v
                  +------------+       +------------+       +------------+
                  | kp-extract |       | kp-extract |       | kp-extract |
                  | concepts   |       | strategies |       | signals    |
                  +------------+       +------------+       +------------+
                        |                     |                     |
                        v                     v                     v
                  [concepts.jsonl]    [strategies.jsonl]   [signals.jsonl]
                        |                     |                     |
                        +---------------------+---------------------+
                                              |
                                              v
                                       +-------------+
                                       | kp-specgen  |
                                       +-------------+
                                              |
                        +---------------------+---------------------+
                        |                     |                     |
                        v                     v                     v
                  [PRD.md]           [Signal Catalog.md]    [Strategy Specs/]

                                       +-------------+
                                       | kp-serve    |
                                       | (FastAPI)   |
                                       +-------------+
                                              |
                                              v
                                       [Engineering Agents]
                                       - Query knowledge
                                       - Semantic search
                                       - Get evidence
```

---

## 3. Source Material

### Course Inventory (Verified from File System)

| Course | Videos | Modules | Named Strategies |
|--------|--------|---------|------------------|
| **Footprint Course** | 90 | 16 Units | 10 + 2 bonus |
| **Price Ladder & Order Flow** | 69 | 16 Modules + Bonus | 8+ patterns |
| **Volume Profiling** | 245 | 3 Modules (12 Parts) | **11 strategies** |
| **Standalone Videos** | 4 | - | 2 strategies |
| **Total** | **408** | | **33+ strategies** |

### Complete Strategy Inventory

**Footprint Course Strategies (12):**
1. Strategy 1: Absorption and Auctioning (Unit 6)
2. Strategy 2: Hiding Behind The Elephant (Unit 7)
3. Strategy 3: Failed Break of Support/Resistance (Unit 8)
4. Strategy 4: Auction Imbalances (Unit 9)
5. Strategy 5: Exhaustion High and Low (Unit 10)
6. Strategy 6: The Initiative Drive (Unit 11)
7. Strategy 7: Key Auction Reversals (Unit 12)
8. Strategy 8: Breakout Trading (Unit 13)
9. Strategy 9: Footprint Delta Position Unwind (Unit 14)
10. Strategy 10: Risk Event Trading (Unit 15)
11. The Zeros (Loose video)
12. Absorption Wall (Loose video)

**Price Ladder Order Flow Patterns (8+):**
13. Large Orders (Module 8)
14. Absorption Order Flow Events (Module 9)
15. Market Flipping, Layering & Spoofing (Module 10)
16. Trend Reversal Order Flow Indicator (Module 11)
17. Momentum Breakout Order Flow (Module 12)
18. Confluence of Order Flow Strategies (Module 13)
19. Evolution of Order Flow Patterns (Module 14)
20. Elite Order Flow Trader Patterns (Module 15)

**Volume Profile Strategies (11):**
21. Strategy 1: Anomaly Strategy (Part 9)
22. Strategy 2: Anomaly Strategy 2 (Part 9)
23. Strategy 3: Momentum Strategy (Part 10)
24. Strategy 4: Momentum Strategy 2 (Part 10)
25. Strategy 5: Momentum Strategy 3 (Part 10)
26. Strategy 6: Momentum Strategy 4 (Part 10)
27. Strategy 7: Trending Strategy (Part 11)
28. Strategy 8: Trending Strategy 2 (Part 11)
29. Strategy 9: Trending Strategy 3 (Part 11)
30. Strategy 10: Reversal Strategy (Part 12)
31. Strategy 11: Reversal Strategy 2 (Part 12)

**Other (2):**
32. Gap Reversal (Loose video)
33. Initial Balance Strategy (Volume Profile Part 5)

---

## 3. Pipeline Architecture

### Stage 1: Video Processing (Current)
```
video.mp4 → run_pipeline.py → output/<video_id>/
                                ├── transcript.json
                                ├── chunks.jsonl
                                ├── frames/
                                └── meta.json
```

**Status**: Running on 408 videos

### Stage 2: Knowledge Extraction (Next)
```
chunks.jsonl → extract_knowledge.py → knowledge.json
                (or in-conversation)
```

**Schema** (`vrag/knowledge_schema.py`):
- `concepts` - Trading terms, metrics, measurements
- `principles` - IF/THEN rules and trading logic
- `procedures` - Step-by-step workflows
- `gotchas` - Warnings, pitfalls, safety guards
- `setups` - Named trading patterns/signals
- `entry_criteria` - Conditions for entering trades
- `exit_criteria` - Targets, stops, trail logic
- `risk_parameters` - Position sizing, risk rules
- `market_contexts` - When strategies apply/don't apply

### Stage 3: Knowledge Aggregation
```
output/*/knowledge.json → aggregate.py → unified_knowledge.json
```

Tasks:
- Deduplicate concepts across videos
- Merge related setups from theory + drill videos
- Cross-reference strategies with their entry/exit criteria
- Build concept glossary with relationships

### Stage 4: PRD Generation
```
unified_knowledge.json → generate_prd.py → prd/
                                           ├── system_overview.md
                                           ├── data_requirements.md
                                           ├── strategy_specs/
                                           │   ├── 01_absorption_auctioning.md
                                           │   └── ...
                                           ├── risk_management.md
                                           └── backtest_criteria.md
```

---

## 4. Knowledge Schema Detail

### What Makes Knowledge "Automatable"

| Field | Purpose | Automation Use |
|-------|---------|----------------|
| `quantifiable: true` | Can be measured from data | Algorithm input |
| `data_source` | Where to get the data | Data pipeline design |
| `detection_criteria` | How to identify pattern | Signal detection |
| `conditions` | Entry/exit rules | Execution logic |
| `codifiable: true` | Expressible as code | Rule engine |

### Example: Extracted Knowledge → Code

**From knowledge.json:**
```json
{
  "name": "Delta",
  "definition": "Net difference between buying and selling volume at a price",
  "quantifiable": true,
  "data_source": "footprint",
  "examples": ["If sellers=500, buyers=96, delta=-404"]
}
```

**Becomes:**
```python
def calculate_delta(footprint_row):
    return footprint_row.buy_volume - footprint_row.sell_volume
```

---

## 5. Output Structure

### Final Deliverables

```
knowledge_platform/
├── data/
│   ├── unified_knowledge.json      # All extracted knowledge merged
│   ├── concepts_glossary.json      # Deduplicated concept dictionary
│   └── strategy_index.json         # Strategy name → knowledge mapping
│
├── strategies/                      # Per-strategy knowledge bundles
│   ├── absorption_auctioning/
│   │   ├── knowledge.json          # Setup, entry, exit, risk
│   │   ├── examples.json           # From drill videos
│   │   └── source_videos.json      # Traceability
│   └── .../
│
├── prd/                            # Generated specifications
│   ├── 00_system_overview.md
│   ├── 01_data_requirements.md
│   ├── 02_strategy_specs/
│   │   ├── absorption_auctioning.md
│   │   └── .../
│   ├── 03_risk_management.md
│   ├── 04_market_contexts.md
│   └── 05_backtest_requirements.md
│
└── validation/
    ├── coverage_report.json        # What % of strategies have full specs
    └── missing_knowledge.json      # Gaps requiring manual review
```

---

## 6. Complete Course Content Inventory (Verified)

### Course 1: The Footprint Course (90 videos)

**Location**: `videos/1 Axia MAIN Futures - The Footprint Course/`
**Focus**: Footprint chart analysis and 10 named strategies + drill videos

| Unit | Topic | Videos | Artifacts to Extract |
|------|-------|--------|---------------------|
| 01 | The Footprint Tool | 3 | **Concept**: Footprint, Delta, Cumulative Delta, Color Shading, Rotation, Order Flow |
| 02 | Setting Up Free Footprint Charts | 1 | **Concept**: Tool configuration, workspace setup |
| 03 | Key Footprint Trading Principles | 2 | **Concept**, **RiskRule**: Core trading principles |
| 04 | Advanced Footprint Charting | 2 | **Concept**: Advanced patterns, chart reading |
| 05 | Footprint Chart Strategy Development | 2 | **Concept**, **RiskRule**: Strategy development framework |
| 06 | **Strategy 1: Absorption and Auctioning** | 6 | **Strategy**, **Signal**: absorption_detected |
| 07 | **Strategy 2: Hiding Behind The Elephant** | 6 | **Strategy**, **Signal**: large_order_detected |
| 08 | **Strategy 3: Failed Break of S/R** | 3 | **Strategy**, **Signal**: failed_break_detected |
| 09 | **Strategy 4: Auction Imbalances** | 4 | **Strategy**, **Signal**: imbalance_detected |
| 10 | **Strategy 5: Exhaustion High and Low** | 3 | **Strategy**, **Signal**: exhaustion_detected |
| 11 | **Strategy 6: The Initiative Drive** | 2 | **Strategy**, **Signal**: initiative_drive_detected |
| 12 | **Strategy 7: Key Auction Reversals** | 5 | **Strategy**, **Signal**: reversal_detected |
| 13 | **Strategy 8: Breakout Trading** | 1 | **Strategy**, **Signal**: breakout_detected |
| 14 | **Strategy 9: Footprint Delta Position Unwind** | 4 | **Strategy**, **Signal**: position_unwind_detected |
| 15 | **Strategy 10: Risk Event Trading** | 2 | **Strategy**, **Signal**: risk_event_setup, **RiskRule** |
| 16 | The Footprint Playbook Debrief | 1 | **RiskRule**, **OpenQuestion**: Best practices, Bitcoin example |

**Drill Videos (Per Strategy):**
Each strategy has Drill 1A (practice) and Drill 1B (guided tuition) videos with real market examples:
- Absorption: Oil, EUR, Gold examples
- Hide Behind Elephant: Oil examples
- Failed Break S/R: Oil examples
- Auction Imbalances: EUR, ES500, Gold examples
- Exhaustion: Oil, ES500, Gold examples
- Initiative Drive: ES500, GBP examples
- Key Auction Reversal: GBP, ES500, Oil examples
- Breakout: Oil, GBP, Gold examples
- Delta Position Unwind: EUR, Oil examples
- Risk Event: GBP, EUR (CPI), Gold (FOMC, Flynn/Trump) examples

**Key Footprint Signals to Extract:**
- Delta (buy volume - sell volume at price)
- Cumulative Delta (running total)
- Color Shading intensity (relative delta strength)
- Volume at price
- Absorption patterns (large orders absorbed)
- Imbalance ratios (stacked bid/ask imbalances)
- Exhaustion patterns (diminishing delta at extremes)
- Initiative vs responsive activity
- Rotation analysis (time-based vs point-and-figure)

---

### Course 2: Price Ladder & Order Flow (69 videos)

**Location**: `videos/Axia Futures - Trading with Price Ladder and Order Flow Strategies/`
**Focus**: DOM/Price Ladder execution and order flow pattern recognition

| Module | Topic | Content Type | Artifacts to Extract |
|--------|-------|--------------|---------------------|
| 01 | Learning Objectives, Outcomes & Methodology | Intro (9 videos) | **Concept**: Course framework, learning methodology |
| 02 | Introduction to The DOM (Depth of Market) | Foundation (17 videos) | **Concept**: Price Ladder, DOM, Order Book, Order Types, Workspace setup |
| 03 | Course Curriculum Outline and Roadmap | Roadmap | **Concept**: Learning path structure |
| 04 | What Ladders Tell You That Charts Don't | Comparison | **Concept**: Information edge, data advantages over charts |
| 05 | Market Participants with Algos & HFT | Context (12 videos) | **Concept**: Algos, HFT, Market Makers, Execution patterns, **RiskRule**: HFT gotchas |
| 06 | Auctioning Exchange and Market Velocity | Core (8 videos) | **Signal**: market_velocity, **Concept**: Auction theory |
| 07 | Introduction to Order Flow Price Patterns | Patterns | **Concept**: Pattern recognition framework |
| 08 | **Large Orders** | Pattern (10 videos) | **Strategy**, **Signal**: large_order_detected |
| 09 | **Absorption Order Flow Events** | Pattern (14 videos) | **Strategy**, **Signal**: dom_absorption_detected |
| 10 | **Market Flipping, Layering & Spoofing** | Pattern (18 videos) | **Signal**: spoofing_detected, layering_detected, flipping_detected, **RiskRule** |
| 11 | **Trend Reversal Order Flow Indicator** | Pattern (14 videos) | **Strategy**, **Signal**: trend_reversal_detected |
| 12 | **Momentum Breakout Order Flow Patterns** | Pattern (20 videos) | **Strategy**, **Signal**: momentum_breakout_detected |
| 13 | Practice - Confluence of Order Flow Strategies | Practice | **Strategy**: confluence trading, combining multiple patterns |
| 14 | Evolution of Order Flow and Price Patterns | Advanced | **Concept**: Pattern evolution, adaptation |
| 15 | What Makes an Elite Order Flow Trader | Mastery (13 videos) | **RiskRule**, **Concept**: Elite trader characteristics |
| 16 | Interviews with Elite Order Flow Traders | Interviews | **Concept**: Real-world insights, trader psychology |
| Bonus | Advanced Replay Drills and Practice | Practice (6 videos) | **Evidence**: Advanced analysis examples |

**Key Price Ladder Signals to Extract:**
- Large Order detection (size thresholds)
- Absorption events (bid/offer absorption)
- Spoofing patterns (fake orders placed and pulled)
- Layering detection (stacked fake orders creating illusion)
- Flipping behavior (rapid side-switching manipulation)
- Trend reversal indicators (order flow shift)
- Momentum breakout patterns (aggressive directional flow)
- Market velocity measurement (speed of tape)
- Bid/offer pulling and refreshing patterns
- Confluence detection (multiple patterns aligning)

---

### Course 3: Volume Profiling with Strategy Development (245 videos)

**Location**: `videos/Axia Futures - Volume Profiling with Strategy Development/`
**Focus**: Comprehensive volume profile analysis with **11 named strategies**

**MODULE 1 - THE APPRENTICESHIP (Parts 1-4)**

| Part | Topic | Videos | Artifacts to Extract |
|------|-------|--------|---------------------|
| 1 | Foundation & Methodology | 24 | **Concept**: Why traders fail, Professional use of VP, Edge definition, Objective vs Subjective analysis |
| 2 | Market Participants & Auction Process | 15 | **Concept**: Key market participants, Timeframe participants, Auction process |
| 3 | Volume Profile Tool Fundamentals | 12 | **Concept**: VP metrics, Value Area, Traditional vs Volume VA, Hover profiles |
| 4 | Volume Profile Anomalies & Process | 18 | **Concept**: Anomalies, Trading process, Basic setup, Software selection |

**MODULE 2 - THE PROFICIENCY (Parts 5-8)**

| Part | Topic | Videos | Artifacts to Extract |
|------|-------|--------|---------------------|
| 5 | Principles, Symmetry, Day Types | 35 | **Concept**: Time/Price/Volume identifiers, Initiative vs Responsive, Symmetrical profiles, **Strategy**: Initial Balance Strategy |
| 6 | Bracketing vs Trending Markets | 30 | **Concept**: Market state identification, Bracketing patterns, Trending patterns |
| 7 | Auction Rotations & Control | 14 | **Concept**: Daytime rotations, Market excess/gaps/tails, VPOC, Auction control variables, HVN/LVN |
| 8 | Profiling Patterns & Templating | 14 | **Concept**: Key control identifiers, Three distinct patterns, **RiskRule**: Deliberate practice |

**MODULE 3 - THE SPECIALIST (Parts 9-12) - 11 STRATEGIES**

| Part | Topic | Videos | Artifacts to Extract |
|------|-------|--------|---------------------|
| 9 | Composite Profiling & Anomaly Strategies | 38 | **Concept**: Composite VP, Volume as market identifier, HVA/LVA, **Strategy 1-2**: Anomaly strategies |
| 10 | Trading Routine & Momentum Strategies | 17 | **Concept**: Pre-open routine, Cash hours, **Strategy 3-6**: Momentum strategies with characteristics |
| 11 | Price Action & Trending Strategies | 14 | **Concept**: Price action with profiling, The trading secret, **Strategy 7-9**: Trending strategies |
| 12 | Reversal Strategies & Playbook Development | 15 | **Concept**: Uncertainty handling, Playbook development, **Strategy 10-11**: Reversal strategies |

**11 Named Volume Profile Strategies:**
1. Strategy 1: Anomaly Strategy (Part 9-32)
2. Strategy 2: Anomaly Strategy 2 (Part 9-34)
3. Strategy 3: Momentum Strategy (Part 10-7)
4. Strategy 4: Momentum Strategy 2 (Part 10-9)
5. Strategy 5: Momentum Strategy 3 (Part 10-11)
6. Strategy 6: Momentum Strategy 4 (Part 10-13)
7. Strategy 7: Trending Strategy (Part 11-6)
8. Strategy 8: Trending Strategy 2 (Part 11-8)
9. Strategy 9: Trending Strategy 3 (Part 11-10)
10. Strategy 10: Reversal Strategy (Part 12-8)
11. Strategy 11: Reversal Strategy 2 (Part 12-10)

**Key Volume Profile Signals to Extract:**
- Point of Control (POC/VPOC) - highest volume price
- Value Area High (VAH) - upper 70% volume boundary
- Value Area Low (VAL) - lower 70% volume boundary
- High Volume Areas (HVA/HVN) - support/resistance zones
- Low Volume Areas (LVA/LVN) - fast move zones
- Initial Balance (IB) - first hour range
- Single prints - initiative activity
- Poor highs/lows - unfinished auction
- Profile shape classification (P, b, D shapes)
- Composite vs session profile analysis
- Bracketing vs trending state detection
- Day type classification (trend, normal, double distribution)

---

### Loose Videos (4 additional)

**Location**: `videos/` (root level)

| Video | Format | Primary Tool | Artifacts to Extract |
|-------|--------|--------------|---------------------|
| **Footprint Strategies: The Absorption Wall** | .avi | footprint | **Strategy**, **Signal**: absorption_wall_detected |
| **Footprint Strategies: The Zeros** (Drill Practice) | .avi | footprint | **Evidence**: Practice examples for zeros pattern |
| **Footprint Strategies: The Zeros** (Trade) | .avi | footprint | **Strategy**, **Signal**: zeros_pattern_detected |
| **Gap Reversal Strategy DW** | .mp4 | combined | **Strategy**, **Signal**: gap_reversal_setup |

---

### Additional Content Identified

**Footprint Course Extras:**
- Bitcoin Footprint Strategy Debrief (Premium)
- Breakout Footprint Strategy Debrief (Premium)
- How to do a Structured Debrief
- Supplementary Blackboard Learning

**All Courses Include:**
- Playbook PDFs (strategy templates)
- Summary Checker videos (knowledge validation)
- Playbook videos (consolidation of learning)

---

## 7. Complete Strategy Inventory (33+ Strategies)

### Footprint Strategies (12) - data_source: `footprint`

| # | Strategy Name | Unit | Primary Signal | Instruments Demonstrated |
|---|---------------|------|----------------|--------------------------|
| 1 | Absorption and Auctioning | 06 | absorption_detected | Oil, EUR, Gold |
| 2 | Hiding Behind The Elephant | 07 | large_order_detected | Oil |
| 3 | Failed Break of S/R | 08 | failed_break_detected | Oil |
| 4 | Auction Imbalances | 09 | imbalance_detected | EUR, ES500, Gold |
| 5 | Exhaustion High and Low | 10 | exhaustion_detected | Oil, ES500, Gold |
| 6 | The Initiative Drive | 11 | initiative_drive_detected | ES500, GBP |
| 7 | Key Auction Reversals | 12 | reversal_detected | GBP, ES500, Oil |
| 8 | Breakout Trading | 13 | breakout_detected | Oil, GBP, Gold |
| 9 | Footprint Delta Position Unwind | 14 | position_unwind_detected | EUR, Oil |
| 10 | Risk Event Trading | 15 | risk_event_setup | GBP, EUR (CPI), Gold (FOMC) |
| 11 | The Zeros | Loose | zeros_pattern_detected | TBD from content |
| 12 | Absorption Wall | Loose | absorption_wall_detected | TBD from content |

### Price Ladder Order Flow Patterns (8) - data_source: `dom`

| # | Strategy Name | Module | Primary Signal | Secondary Signals |
|---|---------------|--------|----------------|-------------------|
| 13 | Large Order Trading | 08 | large_order_detected | size_threshold_exceeded |
| 14 | Absorption Order Flow Events | 09 | dom_absorption_detected | bid_absorption, offer_absorption |
| 15 | Market Flipping/Layering/Spoofing | 10 | spoofing_detected | layering_detected, flipping_detected |
| 16 | Trend Reversal Order Flow | 11 | trend_reversal_detected | order_flow_shift |
| 17 | Momentum Breakout Order Flow | 12 | momentum_breakout_detected | aggressive_flow_detected |
| 18 | Confluence of Order Flow | 13 | confluence_detected | multiple_pattern_alignment |
| 19 | Order Flow Evolution Patterns | 14 | pattern_evolution_detected | adaptation_required |
| 20 | Elite Trader Patterns | 15 | elite_pattern_detected | institutional_flow |

### Volume Profile Strategies (11) - data_source: `volume_profile`

| # | Strategy Name | Part | Primary Signal | Strategy Type |
|---|---------------|------|----------------|---------------|
| 21 | VP Strategy 1: Anomaly | 9 | anomaly_detected | Anomaly |
| 22 | VP Strategy 2: Anomaly 2 | 9 | anomaly_2_detected | Anomaly |
| 23 | VP Strategy 3: Momentum | 10 | momentum_1_detected | Momentum |
| 24 | VP Strategy 4: Momentum 2 | 10 | momentum_2_detected | Momentum |
| 25 | VP Strategy 5: Momentum 3 | 10 | momentum_3_detected | Momentum |
| 26 | VP Strategy 6: Momentum 4 | 10 | momentum_4_detected | Momentum |
| 27 | VP Strategy 7: Trending | 11 | trending_1_detected | Trending |
| 28 | VP Strategy 8: Trending 2 | 11 | trending_2_detected | Trending |
| 29 | VP Strategy 9: Trending 3 | 11 | trending_3_detected | Trending |
| 30 | VP Strategy 10: Reversal | 12 | reversal_1_detected | Reversal |
| 31 | VP Strategy 11: Reversal 2 | 12 | reversal_2_detected | Reversal |

### Combined/Multi-Tool Strategies (2) - data_source: `combined`

| # | Strategy Name | Source | Primary Signal | Tools Required |
|---|---------------|--------|----------------|----------------|
| 32 | Gap Reversal | Loose | gap_reversal_setup | candlestick, volume_profile |
| 33 | Initial Balance Strategy | VP Part 5 | ib_breakout_detected | volume_profile, time |

---

## 8. Cross-Course Relationships

Several concepts appear across courses and should be **linked** using `CrossToolRelationship`:

```
Footprint Absorption ←→ Price Ladder Absorption Events ←→ Volume Profile HVA
         ↓                        ↓                              ↓
    absorption_detected    dom_absorption_detected      hva_support_detected
         ↓                        ↓                              ↓
    └──────────────────→ Unified Absorption Strategy ←──────────┘
```

### Cross-Tool Relationship Matrix

| Concept | Footprint Course | Price Ladder Course | Volume Profile |
|---------|------------------|---------------------|----------------|
| **Absorption** | Strategy 1: Absorption and Auctioning | Module 9: Absorption Events | HVA/LVA (Part 9) |
| **Large Orders** | Strategy 2: Hiding Behind Elephant | Module 8: Large Orders | - |
| **Imbalance** | Strategy 4: Auction Imbalances | Module 10: Layering detection | - |
| **Exhaustion** | Strategy 5: Exhaustion High/Low | Module 11: Trend Reversal | Poor highs/lows (Part 7) |
| **Breakout** | Strategy 8: Breakout Trading | Module 12: Momentum Breakout | Single prints, IB Strategy (Part 5) |
| **Support/Resistance** | Strategy 3: Failed Break S/R | Module 11: Trend Reversal | Value Area (VAH/VAL) |
| **Reversal** | Strategy 7: Key Auction Reversals | Module 11: Trend Reversal | Reversal Strategies 10-11 (Part 12) |
| **Market Velocity** | Unit 3: Principles | Module 6: Market Velocity | Initiative vs Responsive (Part 5) |
| **Auction Theory** | Unit 1: Footprint Tool | Module 6: Auctioning Exchange | Parts 2, 7: Auction Process |
| **Momentum** | Strategy 6: Initiative Drive | Module 12: Momentum Breakout | Momentum Strategies 3-6 (Part 10) |
| **Risk Events** | Strategy 10: Risk Event Trading | Module 5: Algos & HFT | Day Types (Part 5) |
| **Trending Markets** | Unit 3: Principles | Module 14: Pattern Evolution | Trending Strategies 7-9 (Part 11) |

### Shared Concepts Across All Three Courses

| Concept | All Courses Cover | Primary Tool | Automation Priority |
|---------|-------------------|--------------|---------------------|
| **Auction Theory** | Foundation for all trading | All | HIGH |
| **Initiative vs Responsive** | Key market participant behavior | footprint, volume_profile | HIGH |
| **Market Structure** | S/R, trends, ranges | All | HIGH |
| **Volume Analysis** | Core metric interpretation | All | HIGH |
| **Order Flow** | Transaction analysis | footprint, dom | HIGH |
| **Value Area** | Price acceptance zones | volume_profile | MEDIUM |
| **Market Velocity** | Speed of price movement | dom | MEDIUM |
| **Day Types** | Market classification | volume_profile | MEDIUM |

### Extraction Capture Requirements

1. **Same concept, different tool** - e.g., "Absorption" on footprint vs DOM vs VP HVA
2. **Complementary signals** - e.g., footprint delta + ladder order flow + VP levels for confirmation
3. **Strategy variants** - same logic applied to different instruments (Oil, Gold, EUR, ES500, GBP)
4. **Timeframe relationships** - how concepts apply across different timeframes
5. **Confluence requirements** - when multiple tools must agree for high-probability setups

---

## 7. Knowledge Platform Integration

The `knowledge_platform/` project provides the RAG infrastructure for engineering agents.

### Knowledge Platform Schemas (Enhanced for Multi-Course)

| Artifact | Purpose | Key Fields | Course Fields |
|----------|---------|------------|---------------|
| **Concept** | Trading terminology | name, definition, examples | `data_source`, `source_course` |
| **Signal** | Measurable conditions | inputs, computation, thresholds | `data_source`, `tools_required`, `source_course` |
| **Strategy** | Full trade specs | setup, entry, exit, management | `primary_tool`, `instruments_demonstrated`, `source_course` |
| **RiskRule** | Risk management | condition, action, severity | `applies_to`, `source_course` |
| **CrossToolRelationship** | Link related concepts | artifact_ids, relationship_type | `tools_involved`, `integration_notes` |
| **OpenQuestion** | Unresolved items | question, context, priority | - |

**Data Source Options:**
- `footprint` - Footprint chart data (delta, volume at price)
- `dom` - Price ladder / depth of market
- `volume_profile` - Volume profile / market profile
- `candlestick` - Traditional OHLC data
- `tick` - Tick-by-tick data
- `combined` - Multiple sources required

All artifacts include **Evidence** links back to source chunks (chunk_id, quote, confidence).

### Engineering Agent Workflow

```python
# 1. Query knowledge base
response = requests.post("http://localhost:8000/search", json={
    "query": "How do I detect absorption on footprint",
    "top_k": 10
})

# 2. Get relevant chunks with evidence
chunks = response.json()["results"]

# 3. Access compiled strategies
strategies = load_jsonl("data/compiled/strategies.jsonl")
absorption = [s for s in strategies if "absorption" in s["name"].lower()]

# 4. Get full context for a chunk
context = requests.get(f"http://localhost:8000/chunks/{chunk_id}/context")
```

### CLI Commands

```bash
# Step 1: Merge all chunks into single file
cat video_rag_pipeline/output/*/chunks.jsonl > knowledge_platform/data/raw/all_chunks.jsonl

# Step 2: Ingest into vector DB
cd knowledge_platform
kp-ingest data/raw/all_chunks.jsonl

# Step 3: Extract all knowledge types
kp-extract all --output-dir data/compiled/

# Step 4: Generate PRD documents
kp-specgen --compiled-dir data/compiled/ --output-dir data/specs/

# Step 5: Start RAG server for agents
kp-serve --port 8000
```

---

## 7. Next Steps

### Phase 1: Data Pipeline (Current)
- [x] Pipeline running on all 408 videos
- [x] Knowledge platform installed (`kp-ingest`, `kp-serve` ready)
- [ ] Complete video transcription (45/408 done)
- [ ] Merge all chunks.jsonl files
- [ ] Run kp-ingest to vectorize into LanceDB
- [ ] Start kp-serve RAG API

### Phase 2: Content Understanding
- [ ] Read sample transcripts from each course type
- [ ] Understand how content is actually structured
- [ ] Identify what's explicit vs implicit
- [ ] Note patterns in how concepts/strategies are taught

### Phase 3: Extraction Design
- [ ] Design extraction approach based on actual content (not assumptions)
- [ ] Determine batch-extractable vs careful-review items
- [ ] Plan RAG-assisted extraction workflow
- [ ] Build extraction prompts informed by real examples

### Phase 4: Knowledge Extraction
- [ ] Extract and validate knowledge artifacts
- [ ] Generate PRD documents
- [ ] Review and refine

### Phase 5: System Design
- [ ] Design trading system architecture based on extracted knowledge
- [ ] Build backtesting framework
- [ ] Implement first strategy as proof-of-concept

---

## 8. Key Decisions Made

1. **Transcript-based chunking** - More reliable than scene-based for this content
2. **OCR disabled** - Charts don't benefit from text extraction
3. **Captions disabled** - Generic image descriptions add no value
4. **RAG-first approach** - Ingest all content before designing extraction
5. **Evidence-based extraction design** - Read actual transcripts before assuming content structure
6. **No premature optimization** - Don't pre-design extraction without understanding the material

---

## 9. Open Questions

1. **Data feeds**: What market data sources will the automated system use?
   - Footprint data provider?
   - DOM/Level 2 data?
   - Historical data for backtesting?

2. **Execution**: What broker/platform for order execution?
   - API capabilities needed
   - Supported order types

3. **Scope**: Which strategies to implement first?
   - Suggest starting with simpler, more quantifiable ones
   - Build complexity gradually

4. **Validation**: How to validate extracted knowledge?
   - Manual review of key strategies
   - Cross-reference with PDF course materials

---

## 10. Resources

### Files
- `config.yaml` - Pipeline configuration
- `run_pipeline.py` - Video processing orchestrator
- `extract_knowledge.py` - LLM knowledge extraction script
- `vrag/knowledge_schema.py` - Pydantic models for knowledge types

### PDFs (in videos folder)
- `Footprint-Index-Overview.pdf` - Course structure
- `Footprint-Strategy-Template.pdf` - Strategy documentation template

### Sample Output
- `output/UNIT_1___The_Footprint_Tool_1-17/knowledge.json` - Example extraction
