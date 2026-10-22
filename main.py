import argparse
from pathlib import Path

from spirit.config import ExperimentConfig
from spirit.training import run_experiment


def parse_config() -> tuple[ExperimentConfig, int | None]:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Train SPIRIT for cross-instrument spectral regression."
    )
    parser.add_argument(
        "--source-domain",
        choices=("v", "o"),
        help="Transfer direction is inferred from this value.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "pretrain", "regress"),
        help="Run both stages, stage one only, or stage two only.",
    )
    parser.add_argument(
        "--physics-checkpoint",
        type=Path,
        help="Stage-one physics.pt file required by --stage regress.",
    )
    parser.add_argument(
        "--v-csv",
        type=Path,
    )
    parser.add_argument(
        "--o-csv",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--run",
        type=int,
        choices=range(1, defaults.num_runs + 1),
        help=(
            f"Run one evaluation partition (1-{defaults.num_runs}). "
            "Without this option, all partitions are run except in "
            "regression-only mode, which uses config.run_index."
        ),
    )
    parser.add_argument("--device", help="For example: cpu, cuda, or cuda:1")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--regression-epochs", type=int)
    args = parser.parse_args()

    options = vars(args).copy()
    run_number = options.pop("run", None)
    config = ExperimentConfig(
        **{key: value for key, value in options.items() if value is not None}
    )

    if config.stage == "regress":
        if config.physics_checkpoint is None:
            parser.error("--physics-checkpoint is required with --stage regress")
        if not config.physics_checkpoint.is_file():
            parser.error(f"checkpoint not found: {config.physics_checkpoint}")
    elif config.physics_checkpoint is not None:
        parser.error("--physics-checkpoint is only used with --stage regress")

    if run_number is not None:
        selected_run = run_number - 1
    elif config.stage == "regress":
        selected_run = config.run_index
    else:
        selected_run = None
    return config, selected_run


def main() -> None:
    config, selected_run = parse_config()
    run_experiment(config, selected_run)


if __name__ == "__main__":
    main()
