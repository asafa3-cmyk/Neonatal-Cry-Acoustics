"""Stage 3: 3-state clinical output mapping.

Maps the model's 5 corpus-label probabilities to 3 clinical risk states:

    belly_pain              → 3  High-Risk
    discomfort / burping    → 2  Borderline-Suspicious
    hungry / tired          → 1  Normal

Clinical rationale for the mapping
-----------------------------------
belly_pain is the only label carrying an acute visceral pain signal.
It is the sole High-Risk state because missing it has the highest clinical cost.

discomfort and burping are grouped as Borderline because both indicate
the infant is unsettled but neither carries the acute urgency of pain.
They warrant nurse monitoring but not immediate escalation.

hungry and tired are physiologically expected infant states that do not
require clinical intervention — hence Normal.

Threshold logic (clinically conservative)
------------------------------------------
HIGH_RISK_THRESHOLD = 0.30
    The threshold for flagging High-Risk is set BELOW 0.50 so that uncertain
    cases tip toward flagging rather than clearance.  A false positive (nurse
    checks a non-pain cry) costs a brief review; a false negative (missed pain
    cry) costs delayed treatment — asymmetric cost justifies the lower threshold.

NORMAL_THRESHOLD = 0.55
    The bar for clearing a cry as Normal is set ABOVE 0.50 to ensure that
    ambiguous cases are pushed into Borderline rather than cleared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_TO_RISK: dict[str, int] = {
    "belly_pain": 3,
    "discomfort": 2,
    "burping": 2,
    "hungry": 1,
    "tired": 1,
}

STATE_NAMES: dict[int, str] = {
    1: "Normal",
    2: "Borderline–Suspicious",
    3: "High-Risk",
}

HIGH_RISK_THRESHOLD = 0.30
NORMAL_THRESHOLD = 0.55


def collapse_probabilities(
    probs_5class: np.ndarray,
    le: LabelEncoder,
) -> tuple[float, float, float]:
    """Sum 5-class softmax probabilities into the 3 clinical risk levels."""
    p: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
    for idx, label in enumerate(le.classes_):
        p[LABEL_TO_RISK[label]] += float(probs_5class[idx])
    return p[1], p[2], p[3]  # (Normal, Borderline, High-Risk)


def assign_state(
    p_normal: float,
    p_borderline: float,
    p_highrisk: float,
) -> tuple[int, str]:
    """Apply threshold logic to assign a 3-state clinical flag.

    Order of precedence: High-Risk check first (patient safety first).
    """
    if p_highrisk >= HIGH_RISK_THRESHOLD:
        return 3, STATE_NAMES[3]
    if p_normal >= NORMAL_THRESHOLD:
        return 1, STATE_NAMES[1]
    return 2, STATE_NAMES[2]


def classify_batch(
    model,
    X: np.ndarray,
    le: LabelEncoder,
    filenames: list[str] | None = None,
    true_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Classify a batch of feature vectors and return a 3-state output DataFrame."""
    probs_all = model.predict_proba(X)
    rows: list[dict] = []
    for i, probs in enumerate(probs_all):
        p_n, p_b, p_h = collapse_probabilities(probs, le)
        state_id, state_name = assign_state(p_n, p_b, p_h)
        row: dict = {
            "filename": filenames[i] if filenames else f"sample_{i}",
            "P(Normal)": round(p_n, 3),
            "P(Borderline)": round(p_b, 3),
            "P(High-Risk)": round(p_h, 3),
            "state_id": state_id,
            "Clinical Flag": state_name,
        }
        if true_labels:
            row["true_label"] = true_labels[i]
            row["true_risk"] = LABEL_TO_RISK.get(true_labels[i], -1)
        rows.append(row)
    return pd.DataFrame(rows)
