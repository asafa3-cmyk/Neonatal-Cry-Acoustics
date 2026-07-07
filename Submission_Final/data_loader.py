"""Data acquisition and loading utilities for the neonatal cry corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import pandas as pd
import soundfile as sf


REPO_OWNER = "gveres"
REPO_NAME = "donateacry-corpus"
REPO_BRANCH = "master"
SOURCE_DIR = "donateacry_corpus_cleaned_and_updated_data"
TREE_URL = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/"
    f"{REPO_BRANCH}?recursive=1"
)
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/"
    f"{REPO_BRANCH}"
)
DEFAULT_LABELS = ("belly_pain", "burping", "discomfort", "hungry", "tired")
REASON_LABELS = {
    "bp": "belly_pain",
    "bu": "burping",
    "dc": "discomfort",
    "hu": "hungry",
    "ti": "tired",
    "lo": "lonely",
    "ch": "cold_hot",
    "sc": "scared",
    "dk": "unknown",
}
AGE_GROUPS = {
    "04": "0-4 weeks",
    "48": "4-8 weeks",
    "26": "2-6 months",
    "72": "7 months-2 years",
    "22": "more than 2 years",
}


@dataclass(frozen=True)
class AudioRecord:
    source_path: str
    local_path: Path
    label: str
    filename: str
    gender: str | None
    age_code: str | None
    age_group: str | None
    reason_code: str | None
    timestamp_raw: str | None


def fetch_tree() -> dict:
    """Fetch the recursive GitHub tree for the public source repository."""
    with urllib.request.urlopen(TREE_URL, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def cleaned_wav_paths(tree: dict, labels: Iterable[str] = DEFAULT_LABELS) -> list[str]:
    """Return cleaned WAV paths from the requested corpus labels."""
    allowed_prefixes = {
        f"{SOURCE_DIR}/{label}/"
        for label in labels
    }
    paths: list[str] = []
    for entry in tree.get("tree", []):
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        if not path.lower().endswith(".wav"):
            continue
        if any(path.startswith(prefix) for prefix in allowed_prefixes):
            paths.append(path)
    return sorted(paths)


def select_sample_paths(
    paths: Iterable[str],
    samples_per_label: int,
    seed: int,
) -> list[str]:
    """Select a fixed-size stratified sample from the cleaned corpus."""
    by_label: dict[str, list[str]] = {}
    for path in paths:
        label = Path(path).parent.name
        by_label.setdefault(label, []).append(path)

    rng = random.Random(seed)
    selected: list[str] = []
    for label in sorted(by_label):
        label_paths = sorted(by_label[label])
        rng.shuffle(label_paths)
        selected.extend(sorted(label_paths[:samples_per_label]))
    return sorted(selected)


def raw_url_for_source_path(source_path: str) -> str:
    encoded = urllib.parse.quote(source_path)
    return f"{RAW_BASE_URL}/{encoded}"


def parse_filename(filename: str) -> dict[str, str | None]:
    stem = Path(filename).stem
    parts = stem.rsplit("-", 5)
    if len(parts) != 6:
        return {
            "timestamp_raw": None,
            "gender": None,
            "age_code": None,
            "age_group": None,
            "reason_code": None,
        }

    _, timestamp_raw, _, gender, age_code, reason_code = parts
    return {
        "timestamp_raw": timestamp_raw,
        "gender": gender,
        "age_code": age_code,
        "age_group": AGE_GROUPS.get(age_code),
        "reason_code": reason_code,
    }


def local_path_for_source(source_path: str, data_dir: Path) -> Path:
    relative = Path(source_path).relative_to(SOURCE_DIR)
    return data_dir / "raw" / SOURCE_DIR / relative


def download_file(source_path: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(raw_url_for_source_path(source_path), destination)


def acquire_sample(
    data_dir: Path = Path("data"),
    samples_per_label: int = 4,
    seed: int = 42,
    overwrite: bool = False,
) -> list[AudioRecord]:
    """Download a small deterministic sample from the cleaned Donate-a-Cry WAV set."""
    tree = fetch_tree()
    paths = cleaned_wav_paths(tree)
    selected_paths = select_sample_paths(paths, samples_per_label, seed)

    records: list[AudioRecord] = []
    for source_path in selected_paths:
        local_path = local_path_for_source(source_path, data_dir)
        download_file(source_path, local_path, overwrite=overwrite)
        metadata = parse_filename(local_path.name)
        label = local_path.parent.name
        reason_code = metadata["reason_code"]
        if reason_code and REASON_LABELS.get(reason_code, label) != label:
            label = REASON_LABELS.get(reason_code, label)
        records.append(
            AudioRecord(
                source_path=source_path,
                local_path=local_path,
                label=label,
                filename=local_path.name,
                gender=metadata["gender"],
                age_code=metadata["age_code"],
                age_group=metadata["age_group"],
                reason_code=reason_code,
                timestamp_raw=metadata["timestamp_raw"],
            )
        )
    return records


def scan_local_records(data_dir: Path = Path("data")) -> list[AudioRecord]:
    root = data_dir / "raw" / SOURCE_DIR
    records: list[AudioRecord] = []
    for path in sorted(root.glob("*/*.wav")):
        label = path.parent.name
        metadata = parse_filename(path.name)
        source_path = str(Path(SOURCE_DIR) / label / path.name)
        records.append(
            AudioRecord(
                source_path=source_path,
                local_path=path,
                label=label,
                filename=path.name,
                gender=metadata["gender"],
                age_code=metadata["age_code"],
                age_group=metadata["age_group"],
                reason_code=metadata["reason_code"],
                timestamp_raw=metadata["timestamp_raw"],
            )
        )
    return records


def audio_info(path: Path) -> dict[str, float | int | str]:
    info = sf.info(path)
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 4),
        "format": info.format,
        "subtype": info.subtype,
        "file_size_bytes": path.stat().st_size,
    }


def make_stratified_splits(
    records: list[AudioRecord],
    seed: int = 42,
) -> dict[str, str]:
    by_label: dict[str, list[AudioRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    rng = random.Random(seed)
    split_by_filename: dict[str, str] = {}
    for label_records in by_label.values():
        items = sorted(label_records, key=lambda item: item.filename)
        rng.shuffle(items)
        n_items = len(items)
        if n_items == 1:
            assignments = ["train"]
        elif n_items == 2:
            assignments = ["train", "test"]
        else:
            n_val = 1
            n_test = 1
            n_train = max(1, n_items - n_val - n_test)
            assignments = (
                ["train"] * n_train
                + ["val"] * n_val
                + ["test"] * n_test
            )
        for record, split in zip(items, assignments, strict=True):
            split_by_filename[record.filename] = split
    return split_by_filename


def build_manifest(
    records: list[AudioRecord],
    metadata_path: Path = Path("data/metadata.csv"),
    seed: int = 42,
) -> pd.DataFrame:
    split_by_filename = make_stratified_splits(records, seed=seed)
    rows = []
    for record in records:
        details = audio_info(record.local_path)
        rows.append(
            {
                "path": str(record.local_path),
                "source_path": record.source_path,
                "filename": record.filename,
                "label": record.label,
                "split": split_by_filename[record.filename],
                "gender": record.gender,
                "age_code": record.age_code,
                "age_group": record.age_group,
                "reason_code": record.reason_code,
                "timestamp_raw": record.timestamp_raw,
                **details,
            }
        )

    df = pd.DataFrame(rows).sort_values(["label", "filename"]).reset_index(drop=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def load_metadata(metadata_path: Path = Path("data/metadata.csv")) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def load_audio(
    path: str | Path,
    sample_rate: int = 8_000,
    mono: bool = True,
) -> tuple[pd.Series, int]:
    y, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return pd.Series(y, name="amplitude"), sr


def validate_manifest(df: pd.DataFrame) -> None:
    required_columns = {"path", "label", "split", "sample_rate", "duration_sec"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"metadata is missing columns: {missing_columns}")
    missing_paths = [path for path in df["path"] if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing audio files: {missing_paths[:3]}")
    if df["duration_sec"].le(0).any():
        raise ValueError("all audio files must have positive duration")


def summarize_manifest(df: pd.DataFrame) -> dict[str, object]:
    return {
        "n_files": int(len(df)),
        "labels": df["label"].value_counts().sort_index().to_dict(),
        "splits": df["split"].value_counts().sort_index().to_dict(),
        "duration_min_sec": float(df["duration_sec"].min()),
        "duration_max_sec": float(df["duration_sec"].max()),
        "duration_mean_sec": float(df["duration_sec"].mean()),
        "sample_rates": sorted(int(value) for value in df["sample_rate"].unique()),
        "channels": sorted(int(value) for value in df["channels"].unique()),
    }


def run_acquire(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    records = acquire_sample(
        data_dir=data_dir,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    df = build_manifest(
        records,
        metadata_path=Path(args.metadata),
        seed=args.seed,
    )
    validate_manifest(df)
    print(json.dumps(summarize_manifest(df), indent=2))
    return 0


def run_manifest(args: argparse.Namespace) -> int:
    records = scan_local_records(Path(args.data_dir))
    if not records:
        raise FileNotFoundError("no local WAV files found; run acquire first")
    df = build_manifest(records, metadata_path=Path(args.metadata), seed=args.seed)
    validate_manifest(df)
    print(json.dumps(summarize_manifest(df), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire", help="download a small corpus sample")
    acquire.add_argument("--data-dir", default="data")
    acquire.add_argument("--metadata", default="data/metadata.csv")
    acquire.add_argument("--samples-per-label", type=int, default=4)
    acquire.add_argument("--seed", type=int, default=42)
    acquire.add_argument("--overwrite", action="store_true")
    acquire.set_defaults(func=run_acquire)

    manifest = subparsers.add_parser("manifest", help="rebuild metadata from local WAVs")
    manifest.add_argument("--data-dir", default="data")
    manifest.add_argument("--metadata", default="data/metadata.csv")
    manifest.add_argument("--seed", type=int, default=42)
    manifest.set_defaults(func=run_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "samples_per_label", 1) < 1:
        parser.error("--samples-per-label must be at least 1")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
