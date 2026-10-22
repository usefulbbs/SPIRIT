from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig


class WavelengthEncoding(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        if features % 2:
            raise ValueError("positional_features must be even")
        frequency = 1.0 / (
            10000 ** (torch.arange(0, features, 2, dtype=torch.float32) / features)
        )
        self.register_buffer("frequency", frequency)

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        phase = torch.log(wavelengths.clamp_min(1e-6)).unsqueeze(1)
        phase = phase * self.frequency.unsqueeze(0)
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpectrumEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.input_block = DoubleConv1d(in_channels, channels)
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(channels, channels * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(channels * 2, channels * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(channels * 4, channels * 8),
        )

        self.up1 = nn.ConvTranspose1d(channels * 8, channels * 4, 2, stride=2)
        self.up_block1 = DoubleConv1d(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose1d(channels * 4, channels * 2, 2, stride=2)
        self.up_block2 = DoubleConv1d(channels * 4, channels * 2)
        self.up3 = nn.ConvTranspose1d(channels * 2, channels, 2, stride=2)
        self.up_block3 = DoubleConv1d(channels * 2, channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels

    @staticmethod
    def _join(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        difference = skip.size(2) - upsampled.size(2)
        upsampled = F.pad(
            upsampled,
            (difference // 2, difference - difference // 2),
        )
        return torch.cat((skip, upsampled), dim=1)

    def forward(
        self,
        spectrum: torch.Tensor,
        wavelength_encoding: torch.Tensor,
    ) -> torch.Tensor:
        batch_encoding = wavelength_encoding.T.unsqueeze(0).expand(
            spectrum.size(0), -1, -1
        )
        x = torch.cat((spectrum.unsqueeze(1), batch_encoding), dim=1)

        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up_block1(self._join(self.up1(x4), x3))
        x = self.up_block2(self._join(self.up2(x), x2))
        x = self.up_block3(self._join(self.up3(x), x1))
        return self.pool(x).squeeze(-1)


class PeakParameterHead(nn.Module):
    def __init__(self, in_features: int, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_peaks = cfg.num_peaks
        output_dim = cfg.num_peaks * 3 + cfg.baseline_degree + 1
        self.layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, output_dim),
        )
        self._initialize_output()

    def _initialize_output(self) -> None:
        output = self.layers[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(output.bias)

        center_grid = torch.linspace(0.1, 0.9, self.num_peaks)
        center_logits = torch.logit(center_grid)
        with torch.no_grad():
            for peak in range(self.num_peaks):
                output.bias[peak * 3] = center_logits[peak]
                output.bias[peak * 3 + 1] = -0.8
                output.bias[peak * 3 + 2] = -2.0

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.layers(features)
        peak_parameters = raw[:, : self.num_peaks * 3].view(
            -1, self.num_peaks, 3
        )
        baseline = raw[:, self.num_peaks * 3 :]

        centers = self.cfg.wavelength_min + torch.sigmoid(
            peak_parameters[:, :, 0]
        ) * (self.cfg.wavelength_max - self.cfg.wavelength_min)
        widths = self.cfg.peak_min_width + torch.sigmoid(
            peak_parameters[:, :, 1]
        ) * (self.cfg.peak_max_width - self.cfg.peak_min_width)
        amplitudes = self.cfg.peak_max_amplitude * torch.sigmoid(
            peak_parameters[:, :, 2]
        )
        peaks = torch.stack((centers, widths, amplitudes), dim=2)
        return peaks, baseline


class PeakSynthesizer(nn.Module):
    def __init__(self, baseline_degree: int) -> None:
        super().__init__()
        self.baseline_degree = baseline_degree

    def forward(
        self,
        peaks: torch.Tensor,
        baseline_parameters: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = peaks.size(0)
        wavelength_grid = wavelengths.view(1, 1, -1)
        centers = peaks[:, :, 0].unsqueeze(-1)
        widths = peaks[:, :, 1].unsqueeze(-1)
        amplitudes = peaks[:, :, 2].unsqueeze(-1)
        individual_peaks = amplitudes * torch.exp(
            -0.5 * (wavelength_grid - centers).square() / widths.square().clamp_min(1e-9)
        )
        peak_sum = individual_peaks.sum(dim=1)

        wavelength_min = wavelengths.amin()
        wavelength_range = wavelengths.amax() - wavelength_min
        normalized_wavelengths = 2.0 * (
            wavelengths - wavelength_min
        ) / wavelength_range - 1.0
        normalized_wavelengths = normalized_wavelengths.view(1, -1).expand(
            batch_size, -1
        )
        baseline = torch.zeros_like(peak_sum)
        for degree in range(self.baseline_degree + 1):
            baseline = baseline + baseline_parameters[:, degree : degree + 1] * (
                normalized_wavelengths**degree
            )
        return peak_sum + baseline, peak_sum, baseline, individual_peaks


class PeakRegressor(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        hidden = cfg.regressor_hidden_dim
        self.temperature = cfg.regressor_attention_temperature
        self.wavelength_min = cfg.wavelength_min
        self.wavelength_range = cfg.wavelength_max - cfg.wavelength_min
        self.width_min = cfg.peak_min_width
        self.width_range = cfg.peak_max_width - cfg.peak_min_width
        self.amplitude_max = cfg.peak_max_amplitude

        self.embedding = nn.Linear(3, hidden)
        self.embedding_norm = nn.BatchNorm1d(hidden)
        self.embedding_activation = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg.regressor_transformer_heads,
            dim_feedforward=cfg.regressor_feedforward_dim,
            dropout=cfg.regressor_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.regressor_transformer_layers,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, cfg.regressor_attention_dim),
            nn.ReLU(),
            nn.Linear(cfg.regressor_attention_dim, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, cfg.regressor_head_dim),
            nn.ReLU(),
            nn.Linear(cfg.regressor_head_dim, 1),
        )
        self._initialize(cfg.regressor_output_bias)

    def _initialize(self, output_bias: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.regression_head[-1].bias, output_bias)

    def _normalize(self, peaks: torch.Tensor) -> torch.Tensor:
        centers = (peaks[:, :, 0] - self.wavelength_min) / self.wavelength_range
        widths = (peaks[:, :, 1] - self.width_min) / self.width_range
        amplitudes = peaks[:, :, 2] / self.amplitude_max
        return torch.stack((centers, widths, amplitudes), dim=-1)

    def forward(self, peaks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(self._normalize(peaks))
        embedded = embedded.transpose(1, 2)
        embedded = self.embedding_norm(embedded)
        embedded = self.embedding_activation(embedded).transpose(1, 2)
        contextualized = self.transformer(embedded)
        scores = self.attention(contextualized).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=1)
        pooled = (weights.unsqueeze(-1) * contextualized).sum(dim=1)
        prediction = self.regression_head(pooled).squeeze(-1)
        return prediction, weights


class SPIRIT(nn.Module):
    def __init__(
        self,
        cfg: ExperimentConfig,
        wavelengths: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.wavelength_encoding = WavelengthEncoding(cfg.positional_features)
        encoder_input = 1 + cfg.positional_features
        self.encoders = nn.ModuleDict(
            {
                "v": SpectrumEncoder(encoder_input, cfg.encoder_channels),
                "o": SpectrumEncoder(encoder_input, cfg.encoder_channels),
            }
        )
        self.peak_head = PeakParameterHead(cfg.encoder_channels, cfg)
        self.synthesizer = PeakSynthesizer(cfg.baseline_degree)
        self.regressor = PeakRegressor(cfg)
        self.register_buffer("wavelengths_v", wavelengths["v"].clone())
        self.register_buffer("wavelengths_o", wavelengths["o"].clone())
        self._physics_frozen = False

    def _physics_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.wavelength_encoding,
            self.encoders,
            self.peak_head,
            self.synthesizer,
        )

    def physics_parameters(self) -> Iterator[nn.Parameter]:
        for module in self._physics_modules():
            yield from module.parameters()

    def freeze_physics(self) -> None:
        for module in self._physics_modules():
            module.requires_grad_(False)
            module.eval()
        self.regressor.requires_grad_(True)
        self._physics_frozen = True

    def train(self, mode: bool = True) -> "SPIRIT":
        super().train(mode)
        if mode and self._physics_frozen:
            for module in self._physics_modules():
                module.eval()
        return self

    def wavelengths(self, domain: str) -> torch.Tensor:
        if domain == "v":
            return self.wavelengths_v
        if domain == "o":
            return self.wavelengths_o
        raise ValueError(f"unknown domain: {domain}")

    def decompose(
        self,
        domain: str,
        spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wavelengths = self.wavelengths(domain)
        encoded_wavelengths = self.wavelength_encoding(wavelengths)
        features = self.encoders[domain](spectrum, encoded_wavelengths)
        return self.peak_head(features)

    def synthesize(
        self,
        domain: str,
        peaks: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.synthesizer(peaks, baseline, self.wavelengths(domain))

    def predict(self, peaks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.regressor(peaks)
