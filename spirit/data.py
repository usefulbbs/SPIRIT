from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig


@dataclass(frozen=True)
class DomainArrays:
    spectra: np.ndarray
    labels: np.ndarray
    wavelengths: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class SplitIds:
    source_train: np.ndarray
    target_train: np.ndarray
    target_val: np.ndarray
    target_test: np.ndarray

    def validate(self) -> None:
        groups = {
            "source_train": self.source_train,
            "target_train": self.target_train,
            "target_val": self.target_val,
            "target_test": self.target_test,
        }
        names = tuple(groups)
        for name, ids in groups.items():
            if len(ids) != len(np.unique(ids)):
                raise ValueError(f"{name} contains duplicate sample IDs")
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                overlap = np.intersect1d(groups[left_name], groups[right_name])
                if overlap.size:
                    raise ValueError(
                        f"{left_name} and {right_name} share {overlap.size} sample IDs"
                    )


class SpectrumDataset(Dataset):
    def __init__(
        self,
        spectra: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
        domain: str,
    ) -> None:
        self.spectra = torch.as_tensor(spectra, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.sample_ids = np.asarray(sample_ids)
        self.domain = domain

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "x": self.spectra[index],
            "y": self.labels[index],
            "sample_id": self.sample_ids[index],
            "domain": self.domain,
        }


def _snv(spectra: np.ndarray) -> np.ndarray:
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (spectra - mean) / std


def _smooth(spectra: np.ndarray, window: int, polynomial: int) -> np.ndarray:
    if window % 2 == 0 or window <= polynomial or window > spectra.shape[1]:
        raise ValueError(
            f"invalid Savitzky-Golay settings: window={window}, polynomial={polynomial}"
        )
    return savgol_filter(
        spectra,
        window_length=window,
        polyorder=polynomial,
        axis=1,
        mode="mirror",
    ).astype(np.float32)


def _spectrum_columns(df: pd.DataFrame, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    columns = df.columns[start:stop]
    try:
        wavelengths = columns.astype(np.float32).to_numpy()
    except (TypeError, ValueError) as error:
        raise ValueError("spectrum column names must be numeric wavelengths") from error
    spectra = df.iloc[:, start:stop].to_numpy(dtype=np.float32)
    return spectra, wavelengths


def _load_domain(path, domain: str, cfg: ExperimentConfig) -> DomainArrays:
    df = pd.read_csv(path)
    if domain == "v":
        id_column = cfg.v_id_column
        label_column = cfg.v_label_column
        start, stop = cfg.v_spectrum_start, cfg.v_spectrum_stop
        window, polynomial = cfg.v_sg_window, cfg.v_sg_poly
    else:
        id_column = cfg.o_id_column
        label_column = cfg.o_label_column
        start, stop = cfg.o_spectrum_start, cfg.o_spectrum_stop
        window, polynomial = cfg.o_sg_window, cfg.o_sg_poly

    missing = {id_column, label_column}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate values in {id_column}")

    spectra, wavelengths = _spectrum_columns(df, start, stop)
    spectra = _smooth(_snv(spectra), window, polynomial)

    return DomainArrays(
        spectra=spectra,
        labels=df[label_column].to_numpy(dtype=np.float32),
        wavelengths=wavelengths,
        sample_ids=df[id_column].to_numpy(),
    )


def load_aligned_domains(cfg: ExperimentConfig) -> dict[str, DomainArrays]:
    domains = {
        "v": _load_domain(cfg.v_csv, "v", cfg),
        "o": _load_domain(cfg.o_csv, "o", cfg),
    }
    instrument_v, instrument_o = domains["v"], domains["o"]
    if set(instrument_v.sample_ids) != set(instrument_o.sample_ids):
        only_v = len(set(instrument_v.sample_ids).difference(instrument_o.sample_ids))
        only_o = len(set(instrument_o.sample_ids).difference(instrument_v.sample_ids))
        raise ValueError(
            f"instrument sample IDs differ (only V: {only_v}, only O: {only_o})"
        )

    order = pd.Index(instrument_o.sample_ids).get_indexer(instrument_v.sample_ids)
    instrument_o = DomainArrays(
        spectra=instrument_o.spectra[order],
        labels=instrument_o.labels[order],
        wavelengths=instrument_o.wavelengths,
        sample_ids=instrument_o.sample_ids[order],
    )
    if not np.allclose(
        instrument_v.labels,
        instrument_o.labels,
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError("labels differ between paired V and O samples")
    domains["o"] = instrument_o
    return domains


def make_split_ids(sample_ids: np.ndarray, cfg: ExperimentConfig) -> SplitIds:
    expected = cfg.source_train_size + cfg.target_val_size + cfg.target_pool_size
    if len(sample_ids) != expected:
        raise ValueError(f"expected {expected} paired samples, found {len(sample_ids)}")

    shuffled = np.random.RandomState(cfg.split_seed).permutation(np.asarray(sample_ids))
    source_end = cfg.source_train_size
    validation_end = source_end + cfg.target_val_size
    source_ids = shuffled[:source_end]
    validation_ids = shuffled[source_end:validation_end]
    target_pool_ids = shuffled[validation_end:]

    partitions = KFold(
        n_splits=cfg.num_runs,
        shuffle=True,
        random_state=cfg.split_seed,
    )
    test_positions, train_positions = list(partitions.split(target_pool_ids))[
        cfg.run_index
    ]
    split = SplitIds(
        source_train=source_ids,
        target_train=target_pool_ids[train_positions],
        target_val=validation_ids,
        target_test=target_pool_ids[test_positions],
    )
    split.validate()
    return split


def _select(data: DomainArrays, ids: np.ndarray, domain: str) -> SpectrumDataset:
    index = pd.Index(data.sample_ids).get_indexer(ids)
    if np.any(index < 0):
        raise ValueError(f"split contains IDs missing from {domain}")
    return SpectrumDataset(
        data.spectra[index],
        data.labels[index],
        data.sample_ids[index],
        domain,
    )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_dataloaders(
    cfg: ExperimentConfig,
) -> tuple[dict[str, DataLoader], dict[str, torch.Tensor]]:
    cfg.validate()
    domains = load_aligned_domains(cfg)
    split = make_split_ids(domains["v"].sample_ids, cfg)
    source, target = cfg.source_domain, cfg.target_domain

    datasets = {
        "source_train": _select(domains[source], split.source_train, source),
        "target_train": _select(domains[target], split.target_train, target),
        "target_val": _select(domains[target], split.target_val, target),
        "target_test": _select(domains[target], split.target_test, target),
    }
    common = {
        "batch_size": cfg.batch_size,
        "pin_memory": cfg.device.startswith("cuda"),
    }
    loaders = {
        "source_train": DataLoader(
            datasets["source_train"],
            shuffle=True,
            drop_last=True,
            generator=_generator(cfg.seed + cfg.run_index * 10),
            **common,
        ),
        "target_train": DataLoader(
            datasets["target_train"],
            shuffle=True,
            drop_last=True,
            generator=_generator(cfg.seed + cfg.run_index * 10 + 1),
            **common,
        ),
        "target_val": DataLoader(datasets["target_val"], shuffle=False, **common),
        "target_test": DataLoader(datasets["target_test"], shuffle=False, **common),
    }
    wavelengths = {
        name: torch.as_tensor(data.wavelengths, dtype=torch.float32)
        for name, data in domains.items()
    }
    return loaders, wavelengths
