"""
AutoML Agent — CLI entry point.

Usage:
  python run.py --dataset titanic --target Survived --iterations 3 --mock
  python run.py --dataset adult --target income --iterations 2
  python run.py --dataset house_prices --target SalePrice --iterations 3

Flags:
  --dataset     Dataset name: titanic | adult | house_prices
  --target      Target column name (overrides default from loader)
  --iterations  Max improvement iterations (default: 3)
  --mock        Use mock LLM mode (no API calls, fast for testing)
  --run-id      Custom run ID for output directory (default: timestamp)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoML Agent — multi-agent autonomous ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["titanic", "adult", "adult_income", "house_prices", "houses"],
        default="titanic",
        help="Which dataset to run on (default: titanic)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. If omitted, uses the dataset loader's default.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Max improvement iterations (default: 3)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock LLM mode (no API calls)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID for output directory (default: timestamp)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("automl_agent.run")

    # Set mock mode env var so all modules pick it up
    if args.mock:
        os.environ["MOCK_MODE"] = "true"
        logger.info("🔧 Mock mode enabled — no LLM API calls will be made.")

    # Load dataset
    from automl_agent.data.loader import load_dataset

    logger.info(f"📂 Loading dataset: {args.dataset}")
    dataset_info = load_dataset(args.dataset)

    target = args.target or dataset_info.target_column
    logger.info(f"  Target column: '{target}'")
    logger.info(f"  Description: {dataset_info.description}")
    logger.info(f"  Shape: {dataset_info.df.shape}")

    # Validate target exists
    if target not in dataset_info.df.columns:
        logger.error(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {list(dataset_info.df.columns)}"
        )
        sys.exit(1)

    # Run the pipeline
    from automl_agent.graph import run_pipeline

    final_state = run_pipeline(
        dataset_path=dataset_info.dataset_path,
        target_column=target,
        max_iterations=args.iterations,
        run_id=args.run_id,
    )

    # Print summary
    print("\n" + "="*60)
    print("✅ AutoML Agent Pipeline Complete")
    print("="*60)
    print(f"  Task type:    {final_state.get('task_type', 'N/A')}")
    print(f"  Best model:   {final_state.get('_current_best_model_id', 'N/A')}")
    print(f"  Stop reason:  {final_state.get('stop_reason', 'N/A')}")
    print(f"  Iterations:   {final_state.get('iteration', 0)}")
    print(f"  Diagnoses:    {len(final_state.get('error_analysis', []))}")
    print(f"  Critic verdicts: {len(final_state.get('critic_review', []))}")
    print(f"  Feature changes: {len(final_state.get('feature_changes', []))}")

    # Print metric evolution
    eval_results = final_state.get("eval_results", [])
    if eval_results:
        print("\n  Metric evolution (best per iteration):")
        from config import PRIMARY_METRICS
        primary = PRIMARY_METRICS.get(final_state.get("task_type", "classification"), "f1_weighted")
        by_iter: dict[int, float] = {}
        for r in eval_results:
            val = r.get("metrics", {}).get(primary)
            if val is not None:
                it = r["iteration"]
                by_iter[it] = max(by_iter.get(it, -1e9), val)
        for it in sorted(by_iter.keys()):
            print(f"    Iteration {it}: {primary}={by_iter[it]:.4f}")

    from config import RUNS_DIR
    run_dir = RUNS_DIR / (args.run_id or "current")
    print(f"\n  📁 Run output: {run_dir}")
    print(f"  📄 State JSON: {run_dir / 'state.json'}")
    print(f"  📝 Report:     {run_dir / 'report.md'}")
    print("="*60)


if __name__ == "__main__":
    main()
