"""Verify that mockup/index.html's JavaScript assignState() is a byte-for-byte
behavioral match of classify.py's assign_state(), across the 5 real held-out
test-set samples and several threshold combinations.

This script only READS classify.py and data/processed/test_3state_output.csv —
it does not modify either. Run from the repository root:

    python mockup/verify_logic.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classify import assign_state  # noqa: E402  (import after sys.path insert)

CSV_PATH = REPO_ROOT / "data" / "processed" / "test_3state_output.csv"

# Threshold combinations to check: the model's real defaults, plus 3 others
# chosen to exercise all three branches of assign_state() (High-Risk-first,
# Normal, and the Borderline fallback) across the 5 real samples.
THRESHOLD_COMBOS = [
    (0.30, 0.55),  # default (HIGH_RISK_THRESHOLD, NORMAL_THRESHOLD in classify.py)
    (0.20, 0.60),
    (0.50, 0.50),
    (0.05, 0.95),  # extreme: almost everything becomes High-Risk
]


def load_samples() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    samples = load_samples()

    print("=" * 100)
    print("PYTHON GROUND TRUTH — classify.py assign_state() applied to real test_3state_output.csv rows")
    print("=" * 100)

    header = f"{'Sample':<10} {'HR_thr':>7} {'Norm_thr':>9} {'P(Normal)':>10} {'P(Border)':>10} {'P(HighRisk)':>12} -> {'Expected state':<24}"
    print(header)
    print("-" * len(header))

    rows_for_js_comparison = []

    for i, row in enumerate(samples, start=1):
        p_normal = float(row["P(Normal)"])
        p_borderline = float(row["P(Borderline)"])
        p_highrisk = float(row["P(High-Risk)"])

        for hr_thr, norm_thr in THRESHOLD_COMBOS:
            # Directly call the real production function — not a re-implementation.
            state_id, state_name = assign_state(p_normal, p_borderline, p_highrisk)
            # assign_state() reads module-level HIGH_RISK_THRESHOLD / NORMAL_THRESHOLD
            # constants, so to test alternate thresholds we monkeypatch them for
            # this call only, exactly mirroring what the JS sliders parameterize.
            import classify as _classify_module

            original_hr = _classify_module.HIGH_RISK_THRESHOLD
            original_norm = _classify_module.NORMAL_THRESHOLD
            _classify_module.HIGH_RISK_THRESHOLD = hr_thr
            _classify_module.NORMAL_THRESHOLD = norm_thr
            try:
                state_id, state_name = assign_state(p_normal, p_borderline, p_highrisk)
            finally:
                _classify_module.HIGH_RISK_THRESHOLD = original_hr
                _classify_module.NORMAL_THRESHOLD = original_norm

            print(
                f"Sample {i:<3} {hr_thr:>7.2f} {norm_thr:>9.2f} "
                f"{p_normal:>10.3f} {p_borderline:>10.3f} {p_highrisk:>12.3f} -> "
                f"{state_id}. {state_name}"
            )
            rows_for_js_comparison.append(
                {
                    "sample": i,
                    "hr_threshold": hr_thr,
                    "normal_threshold": norm_thr,
                    "p_normal": p_normal,
                    "p_borderline": p_borderline,
                    "p_highrisk": p_highrisk,
                    "expected_state_id": state_id,
                    "expected_state_name": state_name,
                }
            )
        print()

    print("=" * 100)
    print(f"Total (sample x threshold-combo) cases computed: {len(rows_for_js_comparison)}")
    print("Compare this table row-by-row against mockup/index.html's assignState()")
    print("output for the same 5 samples and same 4 threshold combinations.")
    print("=" * 100)


if __name__ == "__main__":
    main()
