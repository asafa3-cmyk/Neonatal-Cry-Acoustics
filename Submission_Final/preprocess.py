"""Audio preprocessing for the neonatal cry data layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_SR = 8_000
PRE_EMPHASIS_COEF = 0.97
N_FFT = 512
HOP_LENGTH = 128
N_MFCC = 13


def check_resampling_required(metadata: pd.DataFrame, target_sr: int = TARGET_SR) -> dict[str, object]:
    """Verify source sample rates before deciding whether to resample.

    Resampling — Not Required: every sampled Donate-a-Cry file is already
    8,000 Hz mono. Resampling code exists in `librosa.resample` for the case
    where a future corpus sample mixes rates, but running it here would be a
    no-op that only risks introducing interpolation artifacts. This function
    is the explicit, auditable check that justifies skipping the step rather
    than silently omitting it.
    """
    unique_rates = sorted(int(r) for r in metadata["sample_rate"].unique())
    return {
        "unique_source_rates_hz": unique_rates,
        "target_rate_hz": target_sr,
        "resampling_required": unique_rates != [target_sr],
    }


def load_waveform(path: str | Path, sample_rate: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def pre_emphasis(y: np.ndarray, coefficient: float = PRE_EMPHASIS_COEF) -> np.ndarray:
    if y.size == 0:
        return y
    emphasized = np.empty_like(y, dtype=np.float32)
    emphasized[0] = y[0]
    emphasized[1:] = y[1:] - coefficient * y[:-1]
    return emphasized


def stft_spectrogram_db(
    y: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitude = np.abs(stft)
    return librosa.amplitude_to_db(magnitude, ref=np.max)


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Per-coefficient z-score normalization across the time axis.

    Raw MFCC magnitude varies with recording volume and microphone gain —
    two cries of the same clinical category can have very different absolute
    MFCC scale simply because one phone recorded louder. Z-scoring each
    coefficient (subtract its own mean, divide by its own std) removes this
    recording-level loudness bias so the model compares cry *shape* rather
    than cry *volume*, which is not clinically meaningful on its own.
    """
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    std_safe = np.where(std < 1e-8, 1.0, std)
    return (mfcc - mean) / std_safe


def preprocess_audio(path: str | Path) -> dict[str, np.ndarray | int]:
    raw, sr = load_waveform(path)
    emphasized = pre_emphasis(raw)
    spectrogram_db = stft_spectrogram_db(emphasized)
    mfcc = extract_mfcc(emphasized, sr)
    mfcc_normalized = normalize_mfcc(mfcc)
    return {
        "raw": raw,
        "sr": sr,
        "pre_emphasized": emphasized,
        "stft_db": spectrogram_db,
        "mfcc": mfcc,
        "mfcc_normalized": mfcc_normalized,
    }


def mfcc_summary_features(mfcc: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)
    for index, value in enumerate(means, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(value)
    for index, value in enumerate(stds, start=1):
        features[f"mfcc_{index:02d}_std"] = float(value)
    return features


def build_feature_table(
    metadata_path: Path = Path("data/metadata.csv"),
    output_path: Path = Path("data/processed/mfcc_features.csv"),
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    rows = []
    for row in metadata.itertuples(index=False):
        processed = preprocess_audio(row.path)
        mfcc = processed["mfcc_normalized"]
        rows.append(
            {
                "path": row.path,
                "filename": row.filename,
                "label": row.label,
                "split": row.split,
                "sample_rate": processed["sr"],
                "n_frames": int(mfcc.shape[1]),
                **mfcc_summary_features(mfcc),
            }
        )
    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _sample_for_figures(metadata: pd.DataFrame) -> pd.Series:
    ordered = metadata.sort_values(["label", "filename"]).reset_index(drop=True)
    return ordered.iloc[0]


def plot_raw_waveform(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    sample = _sample_for_figures(metadata)
    raw, sr = load_waveform(sample["path"])
    time = np.arange(raw.size) / sr
    limit = min(raw.size, sr * 4)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    ax.plot(time[:limit], raw[:limit], color="#22577a", linewidth=0.9)
    _style_axes(
        ax,
        "Before: raw neonatal cry waveform",
        "Time (seconds)",
        "Amplitude",
    )
    ax.text(
        0.01,
        0.96,
        f"Sample label: {sample['label']} | source sampling: {sample['sample_rate']} Hz",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#394046",
    )
    ax.margins(x=0)
    fig.tight_layout()
    output = figures_dir / "01_before_raw_waveform.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_pre_emphasis(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    sample = _sample_for_figures(metadata)
    raw, sr = load_waveform(sample["path"])
    emphasized = pre_emphasis(raw)
    time = np.arange(raw.size) / sr
    limit = min(raw.size, sr * 3)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), dpi=180, sharex=True)
    axes[0].plot(time[:limit], raw[:limit], color="#22577a", linewidth=0.9)
    _style_axes(axes[0], "Before: raw waveform", "", "Amplitude")
    axes[1].plot(time[:limit], emphasized[:limit], color="#c44536", linewidth=0.9)
    _style_axes(
        axes[1],
        "After: pre-emphasis filter",
        "Time (seconds)",
        "Amplitude",
    )
    fig.suptitle(
        "Pre-emphasis increases rapid acoustic changes while preserving timing",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = figures_dir / "02_before_after_pre_emphasis.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_stft(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    sample = _sample_for_figures(metadata)
    processed = preprocess_audio(sample["path"])
    raw = processed["raw"]
    sr = processed["sr"]
    spectrogram = processed["stft_db"]
    time = np.arange(raw.size) / sr
    limit = min(raw.size, sr * 4)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), dpi=180)
    axes[0].plot(time[:limit], raw[:limit], color="#22577a", linewidth=0.9)
    _style_axes(axes[0], "Before: time-domain waveform", "Time (seconds)", "Amplitude")
    image = librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[1],
    )
    _style_axes(
        axes[1],
        "After: STFT spectrogram from pre-emphasized signal",
        "Time (seconds)",
        "Frequency (Hz)",
    )
    colorbar = fig.colorbar(image, ax=axes[1], format="%+2.0f dB", pad=0.012)
    colorbar.set_label("Magnitude (dB)")
    fig.suptitle(
        "STFT converts cry audio into a time-frequency representation",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = figures_dir / "03_before_after_stft_spectrogram.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mfcc(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    sample = _sample_for_figures(metadata)
    processed = preprocess_audio(sample["path"])
    sr = processed["sr"]
    spectrogram = processed["stft_db"]
    mfcc = processed["mfcc"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), dpi=180)
    spec_image = librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[0],
    )
    _style_axes(
        axes[0],
        "Before: STFT spectrogram",
        "Time (seconds)",
        "Frequency (Hz)",
    )
    fig.colorbar(spec_image, ax=axes[0], format="%+2.0f dB", pad=0.012)

    mfcc_image = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        cmap="viridis",
        ax=axes[1],
    )
    axes[1].set_yticks(np.arange(N_MFCC))
    axes[1].set_yticklabels([str(i) for i in range(1, N_MFCC + 1)])
    _style_axes(
        axes[1],
        "After: MFCC feature map",
        "Time (seconds)",
        "MFCC coefficient",
    )
    colorbar = fig.colorbar(mfcc_image, ax=axes[1], pad=0.012)
    colorbar.set_label("Coefficient value")
    fig.suptitle(
        "MFCC extraction compresses the acoustic envelope into lightweight features",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = figures_dir / "04_before_after_mfcc.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_normalization(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    sample = _sample_for_figures(metadata)
    processed = preprocess_audio(sample["path"])
    sr = processed["sr"]
    mfcc = processed["mfcc"]
    mfcc_norm = processed["mfcc_normalized"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), dpi=180)
    raw_image = librosa.display.specshow(
        mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis="time", cmap="viridis", ax=axes[0],
    )
    axes[0].set_yticks(np.arange(N_MFCC))
    axes[0].set_yticklabels([str(i) for i in range(1, N_MFCC + 1)])
    _style_axes(axes[0], "Before: raw MFCC scale", "Time (seconds)", "MFCC coefficient")
    fig.colorbar(raw_image, ax=axes[0], pad=0.012).set_label("Coefficient value")

    norm_image = librosa.display.specshow(
        mfcc_norm, sr=sr, hop_length=HOP_LENGTH, x_axis="time", cmap="viridis", ax=axes[1],
    )
    axes[1].set_yticks(np.arange(N_MFCC))
    axes[1].set_yticklabels([str(i) for i in range(1, N_MFCC + 1)])
    _style_axes(
        axes[1],
        "After: per-coefficient z-score normalization",
        "Time (seconds)",
        "MFCC coefficient",
    )
    fig.colorbar(norm_image, ax=axes[1], pad=0.012).set_label("Normalized value (std units)")
    fig.suptitle(
        "Normalization removes recording-level loudness bias between devices",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = figures_dir / "05_before_after_normalization.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_class_balance(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    counts = metadata["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    bars = ax.bar(counts.index, counts.values, color="#2a9d8f", edgecolor="#1f2933")
    _style_axes(ax, "Sample class balance", "Cry category label", "Files")
    ax.set_ylim(0, max(counts.values) + 1)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    output = figures_dir / "06_sample_class_balance.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_figures(
    metadata_path: Path = Path("data/metadata.csv"),
    figures_dir: Path = Path("figures"),
) -> list[Path]:
    metadata = pd.read_csv(metadata_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_raw_waveform(metadata, figures_dir),
        plot_pre_emphasis(metadata, figures_dir),
        plot_stft(metadata, figures_dir),
        plot_mfcc(metadata, figures_dir),
        plot_normalization(metadata, figures_dir),
        plot_class_balance(metadata, figures_dir),
    ]


def run(args: argparse.Namespace) -> int:
    features = build_feature_table(Path(args.metadata), Path(args.features_out))
    figures = generate_figures(Path(args.metadata), Path(args.figures_dir))
    summary = {
        "features_shape": list(features.shape),
        "features_out": args.features_out,
        "figures": [str(path) for path in figures],
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--features-out", default="data/processed/mfcc_features.csv")
    parser.add_argument("--figures-dir", default="figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
