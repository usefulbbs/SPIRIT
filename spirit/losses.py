import torch
import torch.nn.functional as F


def _second_derivative(
    spectra: torch.Tensor,
    wavelengths: torch.Tensor,
) -> torch.Tensor:
    left_spacing = wavelengths[1:-1] - wavelengths[:-2]
    right_spacing = wavelengths[2:] - wavelengths[1:-1]
    left_slope = (spectra[:, 1:-1] - spectra[:, :-2]) / left_spacing
    right_slope = (spectra[:, 2:] - spectra[:, 1:-1]) / right_spacing
    interior = 2.0 * (right_slope - left_slope) / (
        wavelengths[2:] - wavelengths[:-2]
    )
    return F.pad(interior, (1, 1), mode="replicate")


def reconstruction_loss(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    wavelengths: torch.Tensor,
    curvature_weight: float,
    mse_weight: float,
    cosine_weight: float,
) -> torch.Tensor:
    curvature = (-_second_derivative(observed, wavelengths)).clamp_min(0.0)
    curvature = curvature / curvature.amax(dim=1, keepdim=True).clamp_min(1e-8)
    weight = (1.0 + curvature_weight * curvature).detach()

    mse = (weight * (observed - reconstructed).square()).mean()
    square_root_weight = weight.sqrt()
    observed_weighted = observed * square_root_weight
    reconstructed_weighted = reconstructed * square_root_weight
    cosine = 1.0 - F.cosine_similarity(
        observed_weighted,
        reconstructed_weighted,
        dim=1,
    ).mean()
    return mse_weight * mse + cosine_weight * cosine


def center_stability_loss(
    source_peaks: torch.Tensor,
    target_peaks: torch.Tensor,
) -> torch.Tensor:
    centers = torch.cat(
        (source_peaks[:, :, 0], target_peaks[:, :, 0]),
        dim=0,
    )
    return centers.var(dim=0, unbiased=False).mean()
