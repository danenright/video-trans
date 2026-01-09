"""
Extraction job runner CLI.

Runs extraction jobs and writes results to compiled artifacts.

Usage:
    kp-extract concepts --output ./data/compiled/concepts.jsonl
    kp-extract strategies --output ./data/compiled/strategies.jsonl
    kp-extract all --output-dir ./data/compiled/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kp.extract.base import load_config
from kp.extract.concepts import ConceptExtractor
from kp.extract.strategies import StrategyExtractor
from kp.extract.signals import SignalExtractor
from kp.extract.rules import RiskRuleExtractor
from kp.extract.conflicts import run_conflict_detection
from kp.schemas import Concept, Strategy, Signal, RiskRule

logger = logging.getLogger(__name__)


def write_jsonl(artifacts: list[Any], output_path: Path) -> int:
    """Write artifacts to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for artifact in artifacts:
            if hasattr(artifact, "model_dump"):
                f.write(json.dumps(artifact.model_dump()) + "\n")
            else:
                f.write(json.dumps(artifact) + "\n")
    
    return len(artifacts)


def run_concept_extraction(
    output_path: Path,
    topics: list[str] | None = None,
    batch_size: int = 5,
) -> int:
    """
    Run concept extraction.
    
    Args:
        output_path: Path to write concepts.jsonl
        topics: Optional list of topics to search for (extracts all if None)
        batch_size: Chunks per extraction batch
    
    Returns:
        Number of concepts extracted
    """
    extractor = ConceptExtractor()
    
    if topics:
        # Extract from specific topics
        all_concepts = []
        for topic in topics:
            logger.info(f"Extracting concepts for topic: {topic}")
            concepts = extractor.extract_by_topic(topic, top_k=15)
            all_concepts.extend(concepts)
    else:
        # Extract from all chunks
        logger.info("Extracting concepts from all chunks...")
        all_concepts = extractor.extract_all(batch_size=batch_size)
    
    # Deduplicate by concept_id
    seen = set()
    unique_concepts = []
    for concept in all_concepts:
        if concept.concept_id not in seen:
            seen.add(concept.concept_id)
            unique_concepts.append(concept)
    
    count = write_jsonl(unique_concepts, output_path)
    logger.info(f"Wrote {count} concepts to {output_path}")
    return count


def run_strategy_extraction(
    output_path: Path,
    strategy_names: list[str] | None = None,
    batch_size: int = 5,
) -> int:
    """
    Run strategy extraction.
    
    Args:
        output_path: Path to write strategies.jsonl
        strategy_names: Optional list of strategy names to search for
        batch_size: Chunks per extraction batch
    
    Returns:
        Number of strategies extracted
    """
    extractor = StrategyExtractor()
    
    if strategy_names:
        # Extract specific strategies
        all_strategies = []
        for name in strategy_names:
            logger.info(f"Extracting strategy: {name}")
            strategies = extractor.extract_strategy_by_name(name, top_k=20)
            all_strategies.extend(strategies)
    else:
        # Extract from all chunks
        logger.info("Extracting strategies from all chunks...")
        all_strategies = extractor.extract_all(batch_size=batch_size)
    
    # Deduplicate by strategy_id
    seen = set()
    unique_strategies = []
    for strategy in all_strategies:
        if strategy.strategy_id not in seen:
            seen.add(strategy.strategy_id)
            unique_strategies.append(strategy)
    
    count = write_jsonl(unique_strategies, output_path)
    logger.info(f"Wrote {count} strategies to {output_path}")
    return count


def run_signal_extraction(
    output_path: Path,
    signal_names: list[str] | None = None,
    batch_size: int = 5,
) -> int:
    extractor = SignalExtractor()
    
    if signal_names:
        all_signals = []
        for name in signal_names:
            logger.info(f"Extracting signal: {name}")
            signals = extractor.extract_signal_by_name(name, top_k=15)
            all_signals.extend(signals)
    else:
        logger.info("Extracting signals from all chunks...")
        all_signals = extractor.extract_all(batch_size=batch_size)
    
    seen = set()
    unique_signals = []
    for signal in all_signals:
        if signal.signal_id not in seen:
            seen.add(signal.signal_id)
            unique_signals.append(signal)
    
    count = write_jsonl(unique_signals, output_path)
    logger.info(f"Wrote {count} signals to {output_path}")
    return count


def run_risk_rule_extraction(
    output_path: Path,
    rule_types: list[str] | None = None,
    batch_size: int = 5,
) -> int:
    extractor = RiskRuleExtractor()
    
    if rule_types:
        all_rules = []
        for rule_type in rule_types:
            logger.info(f"Extracting rule type: {rule_type}")
            rules = extractor.extract_rules_by_type(rule_type, top_k=15)
            all_rules.extend(rules)
    else:
        logger.info("Extracting risk rules from all chunks...")
        all_rules = extractor.extract_all(batch_size=batch_size)
    
    seen = set()
    unique_rules = []
    for rule in all_rules:
        if rule.rule_id not in seen:
            seen.add(rule.rule_id)
            unique_rules.append(rule)
    
    count = write_jsonl(unique_rules, output_path)
    logger.info(f"Wrote {count} risk rules to {output_path}")
    return count


def run_all_extraction(
    output_dir: Path,
    batch_size: int = 5,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Concepts
    logger.info("=" * 50)
    logger.info("Running concept extraction...")
    results["concepts"] = run_concept_extraction(
        output_dir / "concepts.jsonl",
        batch_size=batch_size,
    )
    
    # Strategies
    logger.info("=" * 50)
    logger.info("Running strategy extraction...")
    results["strategies"] = run_strategy_extraction(
        output_dir / "strategies.jsonl",
        batch_size=batch_size,
    )
    
    # Signals
    logger.info("=" * 50)
    logger.info("Running signal extraction...")
    results["signals"] = run_signal_extraction(
        output_dir / "signals.jsonl",
        batch_size=batch_size,
    )
    
    # Risk rules
    logger.info("=" * 50)
    logger.info("Running risk rule extraction...")
    results["risk_rules"] = run_risk_rule_extraction(
        output_dir / "risk_rules.jsonl",
        batch_size=batch_size,
    )
    
    # Conflict detection
    logger.info("=" * 50)
    logger.info("Running conflict detection...")
    results["open_questions"] = run_conflict_detection(
        compiled_dir=output_dir,
        output_path=output_dir / "open_questions.jsonl",
    )
    
    # Summary
    logger.info("=" * 50)
    logger.info("Extraction complete:")
    for artifact_type, count in results.items():
        logger.info(f"  {artifact_type}: {count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run knowledge extraction jobs"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Concepts command
    concepts_parser = subparsers.add_parser("concepts", help="Extract concepts")
    concepts_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/compiled/concepts.jsonl"),
        help="Output path for concepts.jsonl",
    )
    concepts_parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        help="Optional topics to focus on (searches for these terms)",
    )
    concepts_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per extraction batch",
    )
    
    # Strategies command
    strategies_parser = subparsers.add_parser("strategies", help="Extract strategies")
    strategies_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/compiled/strategies.jsonl"),
        help="Output path for strategies.jsonl",
    )
    strategies_parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Optional strategy names to focus on",
    )
    strategies_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per extraction batch",
    )
    
    signals_parser = subparsers.add_parser("signals", help="Extract signals")
    signals_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/compiled/signals.jsonl"),
        help="Output path for signals.jsonl",
    )
    signals_parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Optional signal names to focus on",
    )
    signals_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per extraction batch",
    )
    
    rules_parser = subparsers.add_parser("rules", help="Extract risk rules")
    rules_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/compiled/risk_rules.jsonl"),
        help="Output path for risk_rules.jsonl",
    )
    rules_parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        help="Optional rule types to focus on",
    )
    rules_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per extraction batch",
    )
    
    conflicts_parser = subparsers.add_parser("conflicts", help="Detect conflicts and generate open questions")
    conflicts_parser.add_argument(
        "--compiled-dir",
        type=Path,
        default=Path("./data/compiled"),
        help="Directory containing compiled artifacts",
    )
    conflicts_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/compiled/open_questions.jsonl"),
        help="Output path for open_questions.jsonl",
    )
    
    all_parser = subparsers.add_parser("all", help="Run all extraction jobs")
    all_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data/compiled"),
        help="Output directory for all artifacts",
    )
    all_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per extraction batch",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    if args.command == "concepts":
        run_concept_extraction(
            output_path=args.output,
            topics=args.topics,
            batch_size=args.batch_size,
        )
    elif args.command == "strategies":
        run_strategy_extraction(
            output_path=args.output,
            strategy_names=args.names,
            batch_size=args.batch_size,
        )
    elif args.command == "signals":
        run_signal_extraction(
            output_path=args.output,
            signal_names=args.names,
            batch_size=args.batch_size,
        )
    elif args.command == "rules":
        run_risk_rule_extraction(
            output_path=args.output,
            rule_types=args.types,
            batch_size=args.batch_size,
        )
    elif args.command == "conflicts":
        run_conflict_detection(
            compiled_dir=args.compiled_dir,
            output_path=args.output,
        )
    elif args.command == "all":
        run_all_extraction(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
