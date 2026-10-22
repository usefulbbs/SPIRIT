import json
import math
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import build_dataloaders
from .losses import center_stability_loss, reconstruction_loss
from .model import SPIRIT
from .plots import prediction_scatter, reconstruction_examples


class InfiniteLoader:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _physics_checkpoint(
    model: SPIRIT,
    cfg: ExperimentConfig,
    path: Path,
) -> None:
    physics_prefixes = (
        "wavelength_encoding.",
        "encoders.",
        "peak_head.",
        "synthesizer.",
        "wavelengths_",
    )
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(physics_prefixes)
    }
    torch.save(
        {
            "state_dict": state,
            "metadata": {
                "source_domain": cfg.source_domain,
                "run_index": cfg.run_index,
                "split_seed": cfg.split_seed,
                "num_peaks": cfg.num_peaks,
                "baseline_degree": cfg.baseline_degree,
            },
        },
        path,
    )


def _load_physics(
    model: SPIRIT,
    cfg: ExperimentConfig,
    path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
        expected = {
            "source_domain": cfg.source_domain,
            "run_index": cfg.run_index,
            "split_seed": cfg.split_seed,
            "num_peaks": cfg.num_peaks,
            "baseline_degree": cfg.baseline_degree,
        }
        mismatches = {
            name: (metadata[name], value)
            for name, value in expected.items()
            if name in metadata and metadata[name] != value
        }
        if mismatches:
            raise RuntimeError(f"physics checkpoint configuration mismatch: {mismatches}")
    else:
        state = checkpoint
    if any(name.startswith("regressor.") for name in state):
        raise RuntimeError("expected a stage-one physics checkpoint, not a full model")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_without_regressor = [
        name for name in missing if not name.startswith("regressor.")
    ]
    if missing_without_regressor or unexpected:
        raise RuntimeError(
            "invalid physics checkpoint: "
            f"missing={missing_without_regressor}, unexpected={unexpected}"
        )


def pretrain_physics(
    cfg: ExperimentConfig,
    model: SPIRIT,
    loaders: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    optimizer = torch.optim.Adam(
        list(model.physics_parameters()),
        lr=cfg.learning_rate,
    )
    source_loader = loaders["source_train"]
    target_loader = InfiniteLoader(loaders["target_train"])
    best_validation_loss = math.inf

    for epoch in range(cfg.pretrain_epochs):
        model.train()
        epoch_losses = []
        for source_batch in source_loader:
            target_batch = target_loader.next()
            source_x = source_batch["x"].to(device)
            target_x = target_batch["x"].to(device)
            optimizer.zero_grad(set_to_none=True)

            source_peaks, source_baseline = model.decompose(
                cfg.source_domain,
                source_x,
            )
            target_peaks, target_baseline = model.decompose(
                cfg.target_domain,
                target_x,
            )
            source_reconstruction = model.synthesize(
                cfg.source_domain,
                source_peaks,
                source_baseline,
            )[0]
            target_reconstruction = model.synthesize(
                cfg.target_domain,
                target_peaks,
                target_baseline,
            )[0]

            source_loss = reconstruction_loss(
                source_x,
                source_reconstruction,
                model.wavelengths(cfg.source_domain),
                cfg.curvature_weight,
                cfg.reconstruction_mse_weight,
                cfg.reconstruction_cosine_weight,
            )
            target_loss = reconstruction_loss(
                target_x,
                target_reconstruction,
                model.wavelengths(cfg.target_domain),
                cfg.curvature_weight,
                cfg.reconstruction_mse_weight,
                cfg.reconstruction_cosine_weight,
            )
            center_loss = center_stability_loss(source_peaks, target_peaks)
            domain_reconstruction = 0.5 * (source_loss + target_loss)
            loss = cfg.reconstruction_weight * domain_reconstruction
            loss = loss + cfg.center_consistency_weight * center_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        validation_loss = _reconstruction_validation_loss(
            cfg,
            model,
            loaders["target_val"],
            device,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            _physics_checkpoint(model, cfg, checkpoint_path)

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(
                f"  pretrain {epoch + 1:4d}/{cfg.pretrain_epochs}: "
                f"train={np.mean(epoch_losses):.5f}, val={validation_loss:.5f}"
            )


def _reconstruction_validation_loss(
    cfg: ExperimentConfig,
    model: SPIRIT,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            spectra = batch["x"].to(device)
            peaks, baseline = model.decompose(cfg.target_domain, spectra)
            reconstruction = model.synthesize(
                cfg.target_domain,
                peaks,
                baseline,
            )[0]
            loss = reconstruction_loss(
                spectra,
                reconstruction,
                model.wavelengths(cfg.target_domain),
                cfg.curvature_weight,
                cfg.reconstruction_mse_weight,
                cfg.reconstruction_cosine_weight,
            )
            total += loss.item() * spectra.size(0)
            samples += spectra.size(0)
    return total / samples


def train_regressor(
    cfg: ExperimentConfig,
    model: SPIRIT,
    loaders: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    model.freeze_physics()
    optimizer = torch.optim.Adam(
        model.regressor.parameters(),
        lr=cfg.learning_rate,
    )
    target_loader = InfiniteLoader(loaders["target_train"])
    best_r2 = -math.inf

    for epoch in range(cfg.regression_epochs):
        model.train()
        epoch_losses = []
        for source_batch in loaders["source_train"]:
            target_batch = target_loader.next()
            source_x = source_batch["x"].to(device)
            source_y = source_batch["y"].to(device)
            target_x = target_batch["x"].to(device)
            target_y = target_batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                source_peaks, _ = model.decompose(cfg.source_domain, source_x)
                target_peaks, _ = model.decompose(cfg.target_domain, target_x)
            source_prediction, _ = model.predict(source_peaks)
            target_prediction, _ = model.predict(target_peaks)
            loss = 0.5 * (
                F.mse_loss(source_prediction, source_y)
                + F.mse_loss(target_prediction, target_y)
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        metrics, _ = evaluate(
            model,
            loaders["target_val"],
            cfg.target_domain,
            device,
        )
        if np.isfinite(metrics["r2"]) and metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(
                f"  regress  {epoch + 1:4d}/{cfg.regression_epochs}: "
                f"train={np.mean(epoch_losses):.5f}, val_r2={metrics['r2']:.5f}"
            )

    if not checkpoint_path.exists():
        raise RuntimeError("regression did not produce a finite validation checkpoint")


def _batch_ids(batch_ids) -> np.ndarray:
    if isinstance(batch_ids, torch.Tensor):
        return batch_ids.detach().cpu().numpy()
    return np.asarray(batch_ids)


def evaluate(
    model: SPIRIT,
    loader,
    domain: str,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    observed_parts = []
    predicted_parts = []
    id_parts = []
    with torch.no_grad():
        for batch in loader:
            spectra = batch["x"].to(device)
            labels = batch["y"].to(device)
            peaks, _ = model.decompose(domain, spectra)
            predictions, _ = model.predict(peaks)
            observed_parts.append(labels.cpu().numpy())
            predicted_parts.append(predictions.cpu().numpy())
            id_parts.append(_batch_ids(batch["sample_id"]))

    observed = np.concatenate(observed_parts)
    predicted = np.concatenate(predicted_parts)
    sample_ids = np.concatenate(id_parts)
    metrics = {
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "observed": observed,
            "predicted": predicted,
            "absolute_error": np.abs(observed - predicted),
        }
    ).sort_values("absolute_error", ascending=False)
    return metrics, predictions


def run_partition(cfg: ExperimentConfig) -> dict[str, object]:
    if cfg.experiment_id is None:
        cfg = replace(
            cfg,
            experiment_id=datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S_%f"
            ),
        )
    cfg.validate()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = (
        cfg.output_dir
        / cfg.direction
        / cfg.experiment_id
        / f"run_{cfg.run_index + 1:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, wavelengths = build_dataloaders(cfg)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(cfg.to_dict(), file, indent=2)

    model = SPIRIT(cfg, wavelengths).to(device)
    physics_path = run_dir / "physics.pt"
    regression_path = run_dir / "model.pt"
    print(
        f"run {cfg.run_index + 1}/{cfg.num_runs} | "
        f"{cfg.source_domain.upper()} -> {cfg.target_domain.upper()} | "
        f"stage={cfg.stage}"
    )
    print(f"  output: {run_dir}")

    if cfg.stage in {"all", "pretrain"}:
        pretrain_physics(cfg, model, loaders, device, physics_path)
    if cfg.stage == "pretrain":
        result = {
            "seed": cfg.seed,
            "run": cfg.run_index + 1,
            "source_domain": cfg.source_domain,
            "target_domain": cfg.target_domain,
            "stage": cfg.stage,
            "physics_checkpoint": str(physics_path.resolve()),
        }
        with (run_dir / "stage_result.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"  physics checkpoint: {physics_path}")
        return result

    checkpoint_path = (
        physics_path if cfg.stage == "all" else cfg.physics_checkpoint
    )
    if checkpoint_path is None:
        raise RuntimeError("physics checkpoint is not configured")
    _load_physics(model, cfg, checkpoint_path, device)
    train_regressor(cfg, model, loaders, device, regression_path)

    state = torch.load(regression_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    metrics, predictions = evaluate(
        model,
        loaders["target_test"],
        cfg.target_domain,
        device,
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    prediction_scatter(
        predictions["observed"].to_numpy(),
        predictions["predicted"].to_numpy(),
        metrics["r2"],
        run_dir / "prediction_scatter.png",
    )
    reconstruction_examples(
        model,
        loaders["source_train"],
        cfg.source_domain,
        device,
        run_dir / "reconstructions",
        cfg.plot_samples,
    )
    reconstruction_examples(
        model,
        loaders["target_test"],
        cfg.target_domain,
        device,
        run_dir / "reconstructions",
        cfg.plot_samples,
    )
    result = {
        "seed": cfg.seed,
        "run": cfg.run_index + 1,
        "source_domain": cfg.source_domain,
        "target_domain": cfg.target_domain,
        "stage": cfg.stage,
        "physics_checkpoint": str(checkpoint_path.resolve()),
        **metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(
        f"  test: R2={metrics['r2']:.5f}, "
        f"RMSE={metrics['rmse']:.5f}, MAE={metrics['mae']:.5f}"
    )
    return result


def run_experiment(
    cfg: ExperimentConfig,
    selected_run: int | None = None,
) -> pd.DataFrame:
    if cfg.stage == "regress" and selected_run is None:
        raise ValueError(
            "regression-only training requires one selected run because the "
            "physics checkpoint belongs to a specific data partition"
        )
    if cfg.experiment_id is None:
        cfg = replace(
            cfg,
            experiment_id=datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S_%f"
            ),
        )
    runs = range(cfg.num_runs) if selected_run is None else [selected_run]
    results = [
        run_partition(replace(cfg, run_index=run_index))
        for run_index in runs
    ]
    frame = pd.DataFrame(results)
    summary_dir = cfg.output_dir / cfg.direction / cfg.experiment_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    result_name = (
        "physics_checkpoints.csv"
        if cfg.stage == "pretrain"
        else "run_metrics.csv"
    )
    frame.to_csv(summary_dir / result_name, index=False)
    if cfg.stage != "pretrain" and len(frame) > 1:
        metrics = frame[["r2", "rmse", "mae"]]
        summary = pd.DataFrame(
            [metrics.mean(), metrics.std(ddof=0)],
            index=["mean", "std"],
        )
        summary.to_csv(summary_dir / "summary.csv")
        print(summary.to_string())
    return frame
