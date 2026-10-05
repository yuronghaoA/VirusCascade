from __future__ import annotations

import argparse

from agentpoisoning import AgentPoisoningConfig, AgentPoisoningRunner


def _parse_target_item_ids(raw_value: str) -> list[int]:
    item_ids = []
    seen = set()
    for value in raw_value.split(","):
        value = value.strip()
        if not value or not value.isdigit():
            continue
        item_id = int(value)
        if item_id in seen:
            continue
        seen.add(item_id)
        item_ids.append(item_id)
    return item_ids


def _normalize_target_item_ids(
    config: AgentPoisoningConfig,
    parser: argparse.ArgumentParser,
) -> None:
    if config.dataset.target_item_id is not None:
        legacy_item_id = int(config.dataset.target_item_id)
        if config.dataset.target_item_ids and legacy_item_id not in config.dataset.target_item_ids:
            parser.error(
                "Target selection is ambiguous. Use only dataset.target_item_ids or --target-items."
            )
        if not config.dataset.target_item_ids:
            config.dataset.target_item_ids = [legacy_item_id]
        config.dataset.target_item_id = None
    if config.dataset.targets_from or config.dataset.targets_percent is not None:
        parser.error(
            "targets_from / targets_percent are no longer supported here. "
            "Use dataset.target_item_ids or --target-items."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgentPoisoning scaffolding.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, help="Path to JSON config", required=False)
    parser.add_argument("--data-dir", type=str, help="Dataset directory", required=False)
    parser.add_argument("--dataset", type=str, help="Dataset name", required=False)
    parser.add_argument(
        "--agentcf-dataset",
        type=str,
        help="AgentCF dataset name used to resolve rectext_mapping.json",
        required=False,
    )
    parser.add_argument("--output", type=str, help="Output directory", required=False)
    parser.add_argument("--target-items", type=str, help="Comma-separated target item ids", required=False)
    parser.add_argument(
        "--contrastive-lambda",
        type=float,
        help="Override induction.contrastive_lambda",
        required=False,
    )
    parser.add_argument(
        "--cost-alpha",
        type=float,
        help="Override trigger.cost_alpha",
        required=False,
    )
    parser.add_argument(
        "--execute-llm",
        action="store_true",
        help="Call the configured LLM instead of only preparing attack context and prompts.",
    )
    args = parser.parse_args()

    if args.config:
        config = AgentPoisoningConfig.from_json(args.config)
    else:
        config = AgentPoisoningConfig()

    if args.data_dir:
        config.dataset.data_dir = args.data_dir
    if args.dataset:
        config.dataset.dataset_name = args.dataset
    if args.agentcf_dataset:
        config.dataset.mapping_dataset_name = args.agentcf_dataset
    if args.output:
        config.output_dir = args.output
    if args.target_items:
        config.dataset.target_item_ids = _parse_target_item_ids(args.target_items)
    if args.contrastive_lambda is not None:
        config.induction.contrastive_lambda = float(args.contrastive_lambda)
    if args.cost_alpha is not None:
        config.trigger.cost_alpha = float(args.cost_alpha)
    if args.execute_llm:
        config.attack.dry_run = False

    _normalize_target_item_ids(config, parser)
    runner = AgentPoisoningRunner(config)
    runner.run()
    print(f"Saved result to {runner.output_path()}")


if __name__ == "__main__":
    main()
