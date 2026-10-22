from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch


Domain = Literal["v", "o"]
Stage = Literal["all", "pretrain", "regress"]


@dataclass
class ExperimentConfig:
    v_csv: Path = Path("data/instrument_v.csv")
    o_csv: Path = Path("data/instrument_o.csv")
    output_dir: Path = Path("outputs")
    experiment_id: str | None = None

    # This is the only setting needed to reverse the transfer direction.
    source_domain: Domain = "v"
    stage: Stage = "all"
    physics_checkpoint: Path | None = None
    seed: int = 42
    split_seed: int = 42
    num_runs: int = 5
    run_index: int = 0
    source_train_size: int = 630
    target_val_size: int = 20
    target_pool_size: int = 200

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32

    v_sg_window: int = 9
    v_sg_poly: int = 3
    o_sg_window: int = 37
    o_sg_poly: int = 11

    v_to_o_num_peaks: int = 21
    v_to_o_baseline_degree: int = 3
    o_to_v_num_peaks: int = 21
    o_to_v_baseline_degree: int = 3

    wavelength_min: float = 850.0
    wavelength_max: float = 1700.0
    peak_min_width: float = 1.0
    peak_max_width: float = 200.0
    peak_max_amplitude: float = 3.0
    encoder_channels: int = 64
    positional_features: int = 16

    regressor_hidden_dim: int = 64
    regressor_transformer_heads: int = 4
    regressor_transformer_layers: int = 1
    regressor_feedforward_dim: int = 128
    regressor_attention_dim: int = 32
    regressor_attention_temperature: float = 5.0
    regressor_head_dim: int = 32
    regressor_dropout: float = 0.1
    regressor_output_bias: float = 0.5

    learning_rate: float = 1e-3
    pretrain_epochs: int = 500
    regression_epochs: int = 200
    log_every: int = 20

    reconstruction_weight: float = 0.8
    center_consistency_weight: float = 0.2
    reconstruction_mse_weight: float = 0.8
    reconstruction_cosine_weight: float = 0.2
    curvature_weight: float = 0.3
    plot_samples: int = 5

    v_id_column: str = "number"
    o_id_column: str = "number"
    v_label_column: str = "Al2O3/%"
    o_label_column: str = "Al2O3"
    v_spectrum_start: int = 5
    v_spectrum_stop: int = 130
    o_spectrum_start: int = 5
    o_spectrum_stop: int = 517

    @property
    def target_domain(self) -> Domain:
        return "o" if self.source_domain == "v" else "v"

    @property
    def direction(self) -> str:
        return f"{self.source_domain}_to_{self.target_domain}"

    @property
    def num_peaks(self) -> int:
        if self.source_domain == "v":
            return self.v_to_o_num_peaks
        return self.o_to_v_num_peaks

    @property
    def baseline_degree(self) -> int:
        if self.source_domain == "v":
            return self.v_to_o_baseline_degree
        return self.o_to_v_baseline_degree

    def validate(self) -> None:
        if self.source_domain not in {"v", "o"}:
            raise ValueError("source_domain must be 'v' or 'o'")
        if self.stage not in {"all", "pretrain", "regress"}:
            raise ValueError("stage must be 'all', 'pretrain', or 'regress'")
        if self.stage == "regress" and self.physics_checkpoint is None:
            raise ValueError("physics_checkpoint is required for regression-only training")
        if self.num_runs < 2:
            raise ValueError("num_runs must be at least 2")
        if not 0 <= self.run_index < self.num_runs:
            raise ValueError(f"run_index must be in [0, {self.num_runs - 1}]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_peaks < 1:
            raise ValueError("num_peaks must be positive")
        if self.baseline_degree < 0:
            raise ValueError("baseline_degree cannot be negative")
        if self.target_pool_size % self.num_runs:
            raise ValueError("target_pool_size must be divisible by num_runs")
        target_train_size = self.target_pool_size // self.num_runs
        if self.batch_size > min(self.source_train_size, target_train_size):
            raise ValueError(
                "batch_size cannot exceed the smallest training split when drop_last=True"
            )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["num_peaks"] = self.num_peaks
        values["baseline_degree"] = self.baseline_degree
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
