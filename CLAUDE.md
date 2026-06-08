# applied-ml-medicine-final

## Goal
Build a final project for "Applied Machine Learning in Medicine" (B.Sc.) that demonstrates the *thinking* behind developing a real medical AI product — not a heavy or perfect model. Deliverable: a lightweight, CPU-trainable pipeline + formal documentation covering 4 mandated stages, ending in a 3-state clinical output.

## Core Philosophy (never violate)
Lecturer's grading rationale, verbatim: *"The goal is not to build a perfect or heavy model, but to demonstrate the thinking behind developing a real medical AI product."*
→ Every choice is justified by product + clinical reasoning, NOT by chasing accuracy. A simple model with sharp reasoning beats a complex model with none.

## Stack
Python 3.11, Jupyter Notebook, NumPy, pandas, matplotlib, scikit-learn, XGBoost. Minimal deps. Model MUST train in **minutes on a laptop CPU** — no GPU, no heavy/deep architectures, no large pretrained backbones.

## Hard Constraints (CRITICAL — any violation fails the project)
- **Lightweight only:** standard CNN, XGBoost, or basic LSTM. NO over-engineering.
- **Data:** public + accessible (Kaggle / PhysioNet), small or easily sample-able. Record exact source + license.
- **Uniqueness:** avoid clichés — NO basic lung/chest X-rays. Prefer audio/bioacoustics, wearable-sensor signals, or specific tabular data.
- **Forbidden word:** NEVER write "diagnose"/"diagnosis" anywhere (text, code, comments, UI). Use only: *assist, flag, support, screen, monitor*.
- **3-state output:** final user-facing result maps to exactly `1. Normal` / `2. Borderline–Suspicious` / `3. High-Risk`. Never a bare probability, never a binary diagnosis.
- **Before/After visuals:** Stage 2 MUST show explicit "Before" vs "After" plots for each meaningful preprocessing step.

## Project Stages (the spec — follow exactly)

### STAGE 1 — Topic & Product Definition (Lectures 1–3)
- Clinical Problem, Input **X**, Output **Y**.
- Product Name, Target User (Nurse / Radiologist / GP / patient), Workflow integration — pick one: **Screening / Decision Support / Monitoring**.
- Step-by-step **User Flow**.
- **Functional** + **Non-functional** requirements (latency, interpretability, privacy, robustness).

### STAGE 2 — Data Preprocessing (Lecture 4)
- Concrete steps for the data type: noise filtering, normalization, windowing/segmentation, resampling.
- **Before/After** visual per step (signal plot, spectrogram, distribution, or image pair).
- Clinical justification for each step.

### STAGE 3 — Model & Metrics (Weeks 5–6)
- Train + **compare 2 simple models**; pick a primary; explain pros/cons of each.
- Choose **2–3 clinical metrics** (AUC, Sensitivity, Specificity, PPV…) and **justify each clinically** (e.g. why Sensitivity > Accuracy here).
- Define + justify the thresholds that split output into the **3 states**.

### STAGE 4 — MVP Deliverable
- One simple **Jupyter Notebook**: raw data → preprocessing (with before/after) → model → 3-state output.
- Runs top-to-bottom on CPU in minutes. Commented, reproducible (fixed seed).

## Agent Roles

### Codex (Data Engineer — owns the data layer)
Strengths: deterministic ETL, scripted IO, repetitive cleaning, dataset wrangling.
Owns:
1. Find + sample the public dataset (Kaggle/PhysioNet); confirm size, license, accessibility.
2. Write `data_loader.py` — download/sample, validate, train/val/test split (fixed seed).
3. Write `preprocess.py` — filtering, normalization, windowing; deterministic and unit-testable.
4. Generate the **Before/After** plots and save to `figures/`.
Hand off a clean, ready-to-model dataset + a `data_report.md` (shape, class balance, null/outlier notes).

### Claude (Lead — owns reasoning, model, docs)
Strengths: clinical/product reasoning, model design, justification, writing.
Owns:
1. All Stage 1 product/clinical definitions + requirements.
2. `models.py` — the 2-model comparison, primary selection, pros/cons.
3. The metric choice + clinical justification, and the **3-state threshold logic** (`classify.py`).
4. The formal documentation text for every stage (paste-ready for the report).
5. Assemble the final MVP notebook; final review enforcing all Hard Constraints (esp. forbidden word + 3-state output).

### Handoff rule
Codex finishes the data layer and writes `data_report.md` BEFORE Claude touches the model. Claude does not re-clean data; Codex does not decide clinical thresholds.

## Proposed Structure
```
applied-ml-medicine-final/
├── CLAUDE.md
├── data/                # raw + sampled (gitignored if large)
├── data_loader.py       # Codex
├── preprocess.py        # Codex
├── data_report.md       # Codex output
├── models.py            # Claude
├── classify.py          # Claude — 3-state mapping
├── figures/             # before/after + metric plots
├── docs/                # per-stage report text (Claude)
├── mvp.ipynb            # final deliverable
└── requirements.txt
```

## Workflow
Read this file → Claude produces 3 candidate project ideas (First Task) → Asaf selects one → Codex builds data layer + `data_report.md` → Claude builds model, 3-state logic, and docs → assemble `mvp.ipynb` → Claude final-review against Hard Constraints.

## First Task (Claude)
Propose **3 unique, simple, lightweight project ideas** that satisfy ALL Hard Constraints. For each: (1) dataset source + why it's accessible, (2) clinical problem + target user, (3) specific preprocessing that yields clear before/after visuals, (4) the simple model. Then ask Asaf to pick one. Do not write any pipeline code before a selection is made.
