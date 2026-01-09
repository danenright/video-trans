"""
Specification document generator from compiled artifacts.

Generates markdown documents for:
- System PRD
- Strategy specifications
- Signal catalog
- Risk policy
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from kp.schemas import Concept, OpenQuestion, RiskRule, Signal, Strategy

logger = logging.getLogger(__name__)


def load_jsonl(path: Path, model_class: type) -> list:
    if not path.exists():
        return []
    artifacts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                artifacts.append(model_class(**json.loads(line)))
    return artifacts


def generate_system_prd(
    concepts: list[Concept],
    signals: list[Signal],
    strategies: list[Strategy],
    rules: list[RiskRule],
    questions: list[OpenQuestion],
) -> str:
    lines = [
        "# Trading System PRD",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
        "This document describes a trading system derived from video course material.",
        "",
        "## Knowledge Base Summary",
        "",
        f"- **Concepts**: {len(concepts)} trading terms and definitions",
        f"- **Signals**: {len(signals)} measurable market conditions",
        f"- **Strategies**: {len(strategies)} trading strategies",
        f"- **Risk Rules**: {len(rules)} risk management rules",
        f"- **Open Questions**: {len(questions)} items requiring clarification",
        "",
        "## Strategies",
        "",
    ]
    
    for strategy in strategies:
        lines.append(f"### {strategy.name}")
        lines.append("")
        lines.append(strategy.description)
        lines.append("")
        lines.append(f"- **Entry**: {strategy.entry_trigger}")
        lines.append(f"- **Stop**: {strategy.initial_stop}")
        lines.append(f"- **Target**: {strategy.take_profit}")
        lines.append(f"- **Status**: {strategy.status}")
        lines.append("")
    
    lines.extend([
        "## Risk Management",
        "",
    ])
    
    for rule in rules:
        lines.append(f"- **{rule.name}** ({rule.rule_type}): {rule.condition} → {rule.action}")
    
    if questions:
        lines.extend([
            "",
            "## Open Questions",
            "",
            "The following items require clarification before implementation:",
            "",
        ])
        for q in questions:
            priority_marker = "🔴" if q.priority == "high" else "🟡" if q.priority == "medium" else "🟢"
            lines.append(f"- {priority_marker} **{q.question}**")
            if q.context:
                lines.append(f"  - Context: {q.context}")
    
    return "\n".join(lines)


def generate_strategy_spec(strategy: Strategy) -> str:
    lines = [
        f"# Strategy Specification: {strategy.name}",
        "",
        f"**ID**: {strategy.strategy_id}",
        f"**Status**: {strategy.status}",
        "",
        "## Description",
        "",
        strategy.description,
        "",
        "## Scope",
        "",
        f"- **Markets**: {strategy.market_scope}",
        f"- **Timeframes**: {strategy.timeframe_scope}",
        "",
        "## Setup Conditions",
        "",
    ]
    
    if strategy.setup_conditions:
        for condition in strategy.setup_conditions:
            lines.append(f"- {condition}")
    else:
        lines.append("- TBD")
    
    lines.extend([
        "",
        "## Entry",
        "",
        f"**Trigger**: {strategy.entry_trigger}",
        "",
        "## Exit",
        "",
        f"- **Initial Stop**: {strategy.initial_stop}",
        f"- **Take Profit**: {strategy.take_profit}",
        f"- **Invalidation**: {strategy.invalidation}",
        "",
        "## Trade Management",
        "",
    ])
    
    if strategy.trade_management:
        for rule in strategy.trade_management:
            lines.append(f"- {rule}")
    else:
        lines.append("- TBD")
    
    if strategy.no_trade_conditions:
        lines.extend([
            "",
            "## No-Trade Conditions",
            "",
        ])
        for condition in strategy.no_trade_conditions:
            lines.append(f"- {condition}")
    
    if strategy.parameters:
        lines.extend([
            "",
            "## Parameters",
            "",
            "| Parameter | Default | Range | Description |",
            "|-----------|---------|-------|-------------|",
        ])
        for p in strategy.parameters:
            lines.append(f"| {p.name} | {p.default_value} | {p.value_range} | {p.description} |")
    
    if strategy.related_signals:
        lines.extend([
            "",
            "## Related Signals",
            "",
        ])
        for sig in strategy.related_signals:
            lines.append(f"- {sig}")
    
    lines.extend([
        "",
        "## Evidence",
        "",
    ])
    for ev in strategy.evidence[:5]:
        lines.append(f"- Chunk: {ev.chunk_id}")
        if ev.quote:
            lines.append(f'  > "{ev.quote}"')
    
    return "\n".join(lines)


def generate_signals_catalog(signals: list[Signal]) -> str:
    lines = [
        "# Signals Catalog",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Total signals: {len(signals)}",
        "",
    ]
    
    for signal in signals:
        lines.extend([
            f"## {signal.name}",
            "",
            f"**ID**: {signal.signal_id}",
            f"**Status**: {signal.status}",
            "",
            "### Definition",
            "",
            signal.definition,
            "",
            "### Inputs",
            "",
        ])
        
        if signal.inputs:
            for inp in signal.inputs:
                lines.append(f"- **{inp.name}** ({inp.data_type}): {inp.description}")
        else:
            lines.append("- TBD")
        
        lines.extend([
            "",
            f"**Computation**: {signal.computation_frequency}",
            f"**Output**: {signal.output_type}",
            "",
        ])
        
        if signal.thresholds:
            lines.append("### Thresholds")
            lines.append("")
            for k, v in signal.thresholds.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        
        if signal.edge_cases:
            lines.append("### Edge Cases")
            lines.append("")
            for ec in signal.edge_cases:
                lines.append(f"- {ec}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def generate_risk_policy(rules: list[RiskRule]) -> str:
    lines = [
        "# Risk Policy",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Total rules: {len(rules)}",
        "",
    ]
    
    by_severity = {"critical": [], "warning": [], "info": []}
    for rule in rules:
        by_severity.get(rule.severity, by_severity["warning"]).append(rule)
    
    if by_severity["critical"]:
        lines.extend([
            "## Critical Rules (Must Not Violate)",
            "",
        ])
        for rule in by_severity["critical"]:
            lines.append(f"### {rule.name}")
            lines.append("")
            lines.append(f"**Type**: {rule.rule_type}")
            lines.append(f"**Condition**: {rule.condition}")
            lines.append(f"**Action**: {rule.action}")
            lines.append("")
    
    if by_severity["warning"]:
        lines.extend([
            "## Warning Rules",
            "",
        ])
        for rule in by_severity["warning"]:
            lines.append(f"### {rule.name}")
            lines.append("")
            lines.append(f"**Type**: {rule.rule_type}")
            lines.append(f"**Condition**: {rule.condition}")
            lines.append(f"**Action**: {rule.action}")
            lines.append("")
    
    if by_severity["info"]:
        lines.extend([
            "## Informational Rules",
            "",
        ])
        for rule in by_severity["info"]:
            lines.append(f"- **{rule.name}**: {rule.condition} → {rule.action}")
        lines.append("")
    
    return "\n".join(lines)


def generate_all_specs(compiled_dir: Path, output_dir: Path) -> dict[str, Path]:
    concepts = load_jsonl(compiled_dir / "concepts.jsonl", Concept)
    signals = load_jsonl(compiled_dir / "signals.jsonl", Signal)
    strategies = load_jsonl(compiled_dir / "strategies.jsonl", Strategy)
    rules = load_jsonl(compiled_dir / "risk_rules.jsonl", RiskRule)
    questions = load_jsonl(compiled_dir / "open_questions.jsonl", OpenQuestion)
    
    logger.info(f"Loaded: {len(concepts)} concepts, {len(signals)} signals, {len(strategies)} strategies, {len(rules)} rules, {len(questions)} questions")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    
    prd_path = output_dir / "System_PRD.md"
    prd_path.write_text(generate_system_prd(concepts, signals, strategies, rules, questions))
    outputs["prd"] = prd_path
    logger.info(f"Wrote {prd_path}")
    
    signals_path = output_dir / "Signals_Catalog.md"
    signals_path.write_text(generate_signals_catalog(signals))
    outputs["signals"] = signals_path
    logger.info(f"Wrote {signals_path}")
    
    risk_path = output_dir / "Risk_Policy.md"
    risk_path.write_text(generate_risk_policy(rules))
    outputs["risk"] = risk_path
    logger.info(f"Wrote {risk_path}")
    
    strategies_dir = output_dir / "strategies"
    strategies_dir.mkdir(exist_ok=True)
    for strategy in strategies:
        safe_name = strategy.name.replace(" ", "_").replace("/", "_")[:50]
        strategy_path = strategies_dir / f"Strategy_{safe_name}.md"
        strategy_path.write_text(generate_strategy_spec(strategy))
        outputs[f"strategy_{strategy.strategy_id}"] = strategy_path
    logger.info(f"Wrote {len(strategies)} strategy specs to {strategies_dir}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate specification documents from compiled artifacts")
    parser.add_argument(
        "--compiled-dir",
        type=Path,
        default=Path("./data/compiled"),
        help="Directory containing compiled artifacts",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data/specs"),
        help="Output directory for spec documents",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    outputs = generate_all_specs(args.compiled_dir, args.output_dir)
    print(f"Generated {len(outputs)} specification documents in {args.output_dir}")


if __name__ == "__main__":
    main()
