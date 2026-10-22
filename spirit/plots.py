from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import SPIRIT


PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.5,
}


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _format_axis(axis: plt.Axes, label_size: int = 14) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.5,
        labelsize=label_size,
    )
    axis.grid(
        True,
        axis="both",
        color="#cccccc",
        linestyle="-",
        alpha=0.4,
        zorder=0,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)


def _local_cosine_similarity(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    padding = window // 2
    observed_padded = np.pad(observed, padding, mode="edge")
    reconstructed_padded = np.pad(reconstructed, padding, mode="edge")
    similarity = np.empty_like(observed, dtype=np.float64)

    for index in range(observed.size):
        observed_window = observed_padded[index : index + window]
        reconstructed_window = reconstructed_padded[index : index + window]
        observed_norm = np.linalg.norm(observed_window)
        reconstructed_norm = np.linalg.norm(reconstructed_window)
        if observed_norm == 0.0 or reconstructed_norm == 0.0:
            similarity[index] = float(observed_norm == reconstructed_norm)
        else:
            similarity[index] = np.dot(
                observed_window,
                reconstructed_window,
            ) / (observed_norm * reconstructed_norm)
    return np.clip(similarity, -1.0, 1.0)


def prediction_scatter(
    observed: np.ndarray,
    predicted: np.ndarray,
    r2: float,
    path: Path,
) -> None:
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    padding = max((upper - lower) * 0.05, 1e-3)
    limits = (lower - padding, upper + padding)

    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(
            observed,
            predicted,
            s=36,
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.6,
            zorder=3,
        )
        axis.plot(
            limits,
            limits,
            color="#d62728",
            linewidth=2,
            linestyle="--",
            label="$y=x$",
            zorder=2,
        )
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("True Al$_2$O$_3$ (wt.%)", fontsize=14, fontweight="bold")
        axis.set_ylabel(
            "Predicted Al$_2$O$_3$ (wt.%)",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_title("Target Test Predictions", fontsize=16, fontweight="bold", pad=12)
        axis.text(
            0.05,
            0.95,
            f"$R^2$ = {r2:.4f}",
            transform=axis.transAxes,
            fontsize=12,
            va="top",
        )
        axis.legend(loc="lower right", frameon=False, fontsize=11)
        _format_axis(axis, label_size=12)
        axis.grid(True, linestyle="--", alpha=0.5)
        figure.tight_layout()
        _save_figure(figure, path)


def _reconstruction_plot(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    individual_peaks: np.ndarray,
    baseline: np.ndarray,
    wavelengths: np.ndarray,
    title: str,
    path: Path,
    peak_threshold_ratio: float = 0.02,
    cosine_window: int = 15,
) -> None:
    threshold = peak_threshold_ratio * max(float(observed.max()), 1e-8)
    visible_peaks = [
        peak for peak in individual_peaks if float(peak.max()) > threshold
    ]
    residuals = observed - reconstructed
    local_cosine = _local_cosine_similarity(
        observed,
        reconstructed,
        window=cosine_window,
    )

    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=(10, 10.5))
        grid = figure.add_gridspec(5, 1, hspace=0.0)
        spectrum_axis = figure.add_subplot(grid[:3, 0])
        residual_axis = figure.add_subplot(
            grid[3, 0],
            sharex=spectrum_axis,
        )
        cosine_axis = figure.add_subplot(
            grid[4, 0],
            sharex=spectrum_axis,
        )

        spectrum_axis.plot(
            wavelengths,
            observed,
            marker="o",
            markersize=4,
            linestyle="none",
            markeredgecolor="black",
            markerfacecolor="none",
            alpha=0.6,
            label="Processed Spectrum",
            zorder=2,
        )
        spectrum_axis.plot(
            wavelengths,
            reconstructed,
            color="#d62728",
            linewidth=2.5,
            label="Reconstructed Spectrum",
            zorder=4,
        )
        for index, peak in enumerate(visible_peaks):
            label = "Superimposed Peaks" if index == 0 else None
            spectrum_axis.fill_between(
                wavelengths,
                baseline,
                baseline + peak,
                color="#1f77b4",
                alpha=0.25,
                zorder=3,
            )
            spectrum_axis.plot(
                wavelengths,
                baseline + peak,
                color="#1f77b4",
                linewidth=1.2,
                alpha=0.8,
                label=label,
                zorder=3,
            )
        spectrum_axis.plot(
            wavelengths,
            baseline,
            color="#111010",
            linestyle="-.",
            linewidth=2,
            label="Polynomial Drift",
            zorder=4,
        )
        spectrum_axis.set_ylabel("Intensity", fontsize=16, fontweight="bold")
        spectrum_axis.set_title(title, fontsize=18, fontweight="bold", pad=15)
        spectrum_axis.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            fontsize=12,
        )
        spectrum_axis.tick_params(labelbottom=False)

        residual_limit = max(float(np.abs(residuals).max()) * 1.5, 0.1)
        residual_axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        residual_axis.plot(
            wavelengths,
            residuals,
            color="#7f8c8d",
            linewidth=1.5,
        )
        residual_axis.fill_between(
            wavelengths,
            0.0,
            residuals,
            color="#bdc3c7",
            alpha=0.5,
        )
        residual_axis.set_ylim(-residual_limit, residual_limit)
        residual_axis.set_yticks(
            [-0.6 * residual_limit, 0.0, 0.6 * residual_limit]
        )
        residual_axis.set_ylabel("Residuals", fontsize=14, fontweight="bold")
        residual_axis.tick_params(labelbottom=False)

        cosine_minimum = min(float(local_cosine.min()), 0.98)
        cosine_axis.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        cosine_axis.plot(
            wavelengths,
            local_cosine,
            color="#27ae60",
            linewidth=1.5,
        )
        cosine_axis.fill_between(
            wavelengths,
            cosine_minimum,
            local_cosine,
            color="#2ecc71",
            alpha=0.3,
        )
        cosine_axis.set_ylim(cosine_minimum - 0.002, 1.002)
        cosine_axis.set_xlabel(
            "Wavelength (nm)",
            fontsize=16,
            fontweight="bold",
        )
        cosine_axis.set_ylabel("Cos Sim", fontsize=14, fontweight="bold")

        for axis in (spectrum_axis, residual_axis, cosine_axis):
            _format_axis(axis)

        figure.tight_layout()
        _save_figure(figure, path)


def reconstruction_examples(
    model: SPIRIT,
    loader,
    domain: str,
    device: torch.device,
    output_dir: Path,
    count: int,
) -> None:
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    spectra = batch["x"].to(device)
    labels = batch["y"]
    sample_ids = batch["sample_id"]
    count = min(count, spectra.size(0))
    wavelengths = model.wavelengths(domain).detach().cpu().numpy()

    with torch.no_grad():
        peaks, baseline_parameters = model.decompose(domain, spectra[:count])
        reconstructed, _, baseline, individual = model.synthesize(
            domain,
            peaks,
            baseline_parameters,
        )
        predicted, _ = model.predict(peaks)

    observed = spectra[:count].cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    baseline = baseline.cpu().numpy()
    individual = individual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels[:count].cpu().numpy()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        sample_id = sample_ids[index]
        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.item()
        safe_id = str(sample_id).replace("/", "_").replace(" ", "_")
        title = (
            f"Instrument {domain.upper()} Reconstruction "
            f"(ID: {sample_id}) | True: {labels[index]:.3f}, "
            f"Pred: {predicted[index]:.3f}"
        )
        _reconstruction_plot(
            observed[index],
            reconstructed[index],
            individual[index],
            baseline[index],
            wavelengths,
            title,
            output_dir / f"{domain}_{safe_id}.png",
        )
