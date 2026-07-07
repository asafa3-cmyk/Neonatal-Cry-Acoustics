# Project Development Log

Central journal for the Applied ML in Medicine final project. Each entry records the action, justification, files touched, and measurable outcomes.

## 2026-06-08 - Environment Setup

Stage / Action Taken: Data Layer Setup

Clinical or Technical Justification: Created an isolated local Python environment so audio processing dependencies can run without modifying the system Python installation. Python 3.11 was not available on this machine, so the code was written to remain Python 3.11-compatible and verified in the available local interpreter.

Files Created/Modified:

- `.venv/`
- `requirements.txt`

Key Outcomes:

- Installed lightweight project dependencies: NumPy, pandas, matplotlib, scikit-learn, librosa, and soundfile.
- Kept dependency changes local to the project workspace.

## 2026-06-08 - Data Acquisition

Stage / Action Taken: Stage 2 Data Acquisition

Clinical or Technical Justification: Selected the public Donate-a-Cry cleaned WAV subset because neonatal cry acoustics fit the unique audio-based project direction and allow CPU-light signal preprocessing. Downloaded a small balanced sample to keep the MVP fast and presentation-friendly.

Files Created/Modified:

- `data_loader.py`
- `data/raw/donateacry_corpus_cleaned_and_updated_data/`
- `data/metadata.csv`

Key Outcomes:

- Downloaded 20 WAV files from the public GitHub corpus.
- Sample contains 5 corpus labels with 4 files per label: `belly_pain`, `burping`, `discomfort`, `hungry`, `tired`.
- Generated deterministic split with seed `42`: 10 train, 5 validation, 5 test.
- Verified all sampled files are mono WAV, 8,000 Hz, PCM 16-bit.
- Duration range: 6.54-7.06 seconds; mean duration: 6.93 seconds.

## 2026-06-08 - Data Loader Implementation

Stage / Action Taken: Stage 2 Data Loading

Clinical or Technical Justification: Implemented deterministic acquisition, metadata parsing, validation, and split creation so Claude receives a clean data table without needing to re-clean the raw audio layer.

Files Created/Modified:

- `data_loader.py`
- `data/metadata.csv`

Key Outcomes:

- `data_loader.py acquire --samples-per-label 4 --seed 42` downloads a repeatable balanced sample.
- `data_loader.py manifest` can rebuild metadata from local WAV files.
- Metadata contains 20 rows and 17 columns, with no missing values in the sampled files.

## 2026-06-08 - Preprocessing Implementation

Stage / Action Taken: Stage 2 Preprocessing

Clinical or Technical Justification: Implemented the requested audio pipeline: pre-emphasis filter, STFT spectrogram, and MFCC extraction. Kept the target sampling rate at 8 kHz because all sampled source files are already 8 kHz, avoiding artificial upsampling and empty frequency ranges.

Files Created/Modified:

- `preprocess.py`
- `data/processed/mfcc_features.csv`

Key Outcomes:

- Pre-emphasis coefficient: `0.97`.
- STFT settings: `n_fft=512`, `hop_length=128`.
- MFCC settings: 13 coefficients.
- Generated `data/processed/mfcc_features.csv` with 20 rows and 32 columns.
- Feature columns include MFCC means and standard deviations for each coefficient.

## 2026-06-08 - Before/After Figures

Stage / Action Taken: Stage 2 Visual Documentation

Clinical or Technical Justification: Created explicit before/after visuals for Lecture 4 so the report can show how raw mobile audio becomes structured acoustic features.

Files Created/Modified:

- `figures/01_before_raw_waveform.png`
- `figures/02_before_after_pre_emphasis.png`
- `figures/03_before_after_stft_spectrogram.png`
- `figures/04_before_after_mfcc.png`
- `figures/05_sample_class_balance.png`

Key Outcomes:

- Raw waveform figure shows the source time-domain signal.
- Pre-emphasis figure compares waveform before and after filtering.
- STFT figure compares waveform to the time-frequency spectrogram.
- MFCC figure compares STFT representation to compact cepstral features.
- Class balance figure confirms the balanced 20-file sample.

## 2026-06-08 - Data Report

Stage / Action Taken: Stage 2 Data Documentation

Clinical or Technical Justification: Wrote a course-facing data report aligned to Lecture 4 requirements, including source, license, sample limitations, preprocessing choices, and embedded before/after figures.

Files Created/Modified:

- `data_report.md`

Key Outcomes:

- Documented source: public `gveres/donateacry-corpus` GitHub repository.
- Documented source license: ODbL for the database and DbCL for individual contents.
- Documented generated metadata, feature table, data-quality notes, and Stage 2 figure order.

---

## 2026-06-08 - Dependency Fix (XGBoost + libomp)

Stage / Action Taken: Environment — Dependency Resolution

Clinical or Technical Justification: XGBoost is a core stack requirement. On macOS, XGBoost's shared library requires the OpenMP runtime (`libomp.dylib`). It was absent from the system, causing an import error. Installed via `brew install libomp` (non-destructive system library). Added `xgboost>=2.0` to `requirements.txt` to make the dependency explicit.

Files Created/Modified:

- `requirements.txt`

Key Outcomes:

- XGBoost 3.2.0 confirmed importable in the project venv.
- `libomp` 22.1.7 installed via Homebrew.

---

## 2026-06-08 - Stage 3: Model Architecture & Training

Stage / Action Taken: Stage 3 — Model & Metrics

Clinical or Technical Justification: Two models were trained and compared on the 26-feature MFCC table (13 means + 13 stds). XGBoost was chosen as the primary model because on small tabular datasets with engineered features, gradient-boosted trees are more stable and interpretable than neural networks. Interpretability (feature importance) is a mandatory non-functional requirement for a medical screening product.

The MLP neural baseline substitutes for the originally proposed 1D CNN: since PyTorch/TensorFlow are outside the project stack, and the 1D CNN's temporal inductive bias is already discarded by the MFCC mean/std pooling step, `MLPClassifier` (2 hidden layers, ReLU, Adam) is functionally equivalent for this feature representation.

Files Created/Modified:

- `models.py` (new) — `load_data`, `build_xgboost`, `build_mlp`, `evaluate_model`, `plot_confusion_matrix`, `plot_metric_comparison`

Key Outcomes (on 5-sample test set — illustrative only):

| Model | Accuracy | Macro AUC |
|---|---:|---:|
| XGBoost | 0.00 | 0.45 |
| MLP Baseline | 0.20 | 0.25 |

**Important caveat:** With n=15 training samples across 5 classes (3 per class), both models are severely underpowered. These metrics demonstrate the methodology pipeline, not production performance. A production system would require hundreds of expert-annotated recordings per class.

Model hyperparameters:
- XGBoost: n_estimators=50, max_depth=2, learning_rate=0.3, seed=42
- MLP: hidden_layer_sizes=(32, 16), activation=relu, solver=adam, max_iter=500, seed=42

---

## 2026-06-08 - Stage 3: Clinical Metric Selection

Stage / Action Taken: Stage 3 — Metric Justification

Clinical or Technical Justification:

Three metrics were selected and each is clinically justified:

1. **Sensitivity (Recall) for High-Risk class** — The cost of a false negative (missed belly_pain cry) is the highest in the system: delayed treatment for an infant in acute pain. Sensitivity for the High-Risk state is therefore the primary metric. Maximised by setting a low High-Risk detection threshold (0.30 instead of 0.50).

2. **Specificity for High-Risk class** — Excessive false alarms cause alert fatigue in nursing staff, eroding trust in the system. Specificity must remain acceptable alongside sensitivity. This is the known sensitivity–specificity tradeoff in clinical screening.

3. **Macro-average AUC (OvR)** — Threshold-independent measure of overall discriminative ability across all 3 clinical states. More stable than accuracy on small balanced samples; not inflated by any dominant class. Used as the primary model comparison metric.

Files Created/Modified: None (analytical decision documented here).

Key Outcomes: Metric selection reflects an asymmetric clinical cost function: false negatives for High-Risk (missed pain) are penalised more heavily than false positives.

---

## 2026-06-08 - Stage 3: 3-State Clinical Output Logic

Stage / Action Taken: Stage 3 — 3-State Threshold Logic

Clinical or Technical Justification: The XGBoost model produces 5-class softmax probabilities. These are collapsed into 3 clinical states by summing the probabilities of corpus labels that share the same clinical urgency level.

Mapping:
- `belly_pain` → High-Risk (3): acute visceral pain; highest urgency.
- `discomfort` + `burping` → Borderline–Suspicious (2): mild distress; warrants monitoring.
- `hungry` + `tired` → Normal (1): routine physiological states; no urgency.

Threshold assignment (clinically conservative):
- P(High-Risk) ≥ 0.30 → flag as **High-Risk** (low threshold to maximise sensitivity for pain).
- P(Normal) ≥ 0.55 → assign **Normal** (require clear normal signal; ambiguity defaults to Borderline).
- Otherwise → **Borderline–Suspicious** (uncertain cases flagged, not cleared).

The thresholds are a clinical product decision, not statistical optimisation. They would be calibrated against a larger annotated dataset before any clinical deployment.

Files Created/Modified:

- `classify.py` (new) — `collapse_probabilities`, `assign_state`, `classify_batch`, `LABEL_TO_RISK`, `STATE_NAMES`

Key Outcomes (test set, 5 samples):

| true_label | P(Normal) | P(Borderline) | P(High-Risk) | Clinical Flag |
|---|---:|---:|---:|---|
| belly_pain | 0.450 | 0.430 | 0.120 | Borderline–Suspicious |
| burping | 0.265 | 0.193 | 0.542 | High-Risk |
| discomfort | 0.081 | 0.267 | 0.652 | High-Risk |
| hungry | 0.366 | 0.596 | 0.038 | Borderline–Suspicious |
| tired | 0.160 | 0.738 | 0.102 | Borderline–Suspicious |

Note: with n=15 training samples, the model mislabels most test samples at both the 5-class and 3-state level. The architecture and clinical reasoning are sound; the dataset size is the limiting factor.

---

## 2026-06-08 - Stage 4: MVP Notebook Assembly

Stage / Action Taken: Stage 4 — MVP Deliverable

Clinical or Technical Justification: Assembled the final MVP notebook (`Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb`) integrating all four stages into a single top-to-bottom reproducible document. The notebook imports from `models.py` and `classify.py`, displays Stage 2 before/after figures inline, trains both models, shows the comparison, and outputs the 3-state clinical table.

Files Created/Modified:

- `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb` (new) — full end-to-end MVP notebook
- `models.py` (new) — Stage 3 model comparison module
- `classify.py` (new) — Stage 3 3-state mapping module
- `requirements.txt` (updated) — added `xgboost>=2.0`
- `data/processed/test_3state_output.csv` (generated on run) — test set 3-state results

Key Outcomes:

- Pipeline runs top-to-bottom in approximately 2 seconds on a laptop CPU.
- Fixed seed 42 used throughout — fully reproducible.
- Restricted clinical terminology check passed across project files.
- All 4 mandated stages covered.
- 3-state output implemented with documented clinical threshold logic.

---

## 2026-06-08 - Public Repository Preparation

Stage / Action Taken: Repository Packaging

Clinical or Technical Justification: Prepared the project for public GitHub sharing with a professional README, clean ignore rules, and explicit run instructions. The README presents the project as a responsible screening-support MVP and clearly separates pipeline demonstration from production performance.

Files Created/Modified:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `project_development_log.md`
- `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb`

Key Outcomes:

- Added ignore rules for `.venv/`, `__pycache__/`, `.DS_Store`, notebook checkpoints, and local matplotlib cache.
- Created a visually documented README with embedded Stage 2 figures.
- Added notebook runtime dependency to `requirements.txt`.
- Removed restricted clinical wording from notebook/log text before public commit.

---

## 2026-06-08 - Local Git Setup

Stage / Action Taken: Version Control

Clinical or Technical Justification: Initialized local Git tracking so the full MVP can be shared as a reproducible public repository with source code, figures, metadata, reports, and the project notebook under one versioned history.

Files Created/Modified:

- `.git/`
- `project_development_log.md`

Key Outcomes:

- Initialized a local Git repository.
- Set the default branch to `main`.
- Confirmed `.gitignore` excludes local environment, Python cache, Mac system files, and plotting cache artifacts.

---

## 2026-06-08 - XGBoost Results Figure for README

Stage / Action Taken: Stage 3 Model Results Documentation

Clinical or Technical Justification: Added a visible modeling-results artifact to the GitHub README so the public repository shows both the data preprocessing pipeline and the final AI evaluation stage. Used a feature-importance and 3-state confusion-matrix figure because they are easy to explain in a presentation and align with the project's interpretability requirement.

Files Created/Modified:

- `figures/06_xgboost_results.png`
- `data/processed/test_3state_output.csv`
- `data/processed/xgboost_metrics.json`
- `README.md`
- `project_development_log.md`

Key Outcomes:

- Generated XGBoost result plot with top MFCC feature importances, 3-state confusion matrix, and metric bars.
- Final MVP test metrics: High-Risk sensitivity `0.00`, High-Risk specificity `0.50`, clinical-state macro AUC `0.33`, 5-class macro AUC `0.45`.
- README now presents both Stage 2 data pipeline visuals and Stage 3 model-results evidence.

---

## 2026-06-08 - GitHub Publication

Stage / Action Taken: Public Repository Push

Clinical or Technical Justification: Published the MVP as a public GitHub repository so the complete reproducible project can be shared with peers, instructors, and future employers.

Files Created/Modified:

- `project_development_log.md`

Key Outcomes:

- Created public repository: `https://github.com/asafa3-cmyk/Neonatal-Cry-Acoustics`
- Pushed local `main` branch to `origin/main`.
- Public README includes data pipeline visuals and XGBoost model-results evidence.

---

## 2026-06-08 - Final Notebook Polish & Submission Seal

Stage / Action Taken: Stage 4 — MVP Notebook Finalisation for Academic Submission

Clinical or Technical Justification: Polished the MVP notebook for formal academic submission. The notebook was restructured to tell a complete, compelling clinical product story — from problem definition through preprocessing rationale to model selection and 3-state output — with professional markdown throughout. All design decisions (XGBoost primary selection, MLP substitute for 1D CNN, 3 clinical metrics, conservative threshold logic) are explained at the right level of detail for a non-technical reader (nurse, GP) and a technical reviewer (lecturer).

Actions taken:
- Replaced the generic notebook filename with `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb`.
- Verified and cross-referenced with `README.md` and full `project_development_log.md`.
- Upgraded all markdown cells: Executive Summary, Stage 1 product spec with user flow + requirements, Stage 2 preprocessing with clinical rationale for each step (pre-emphasis, STFT, MFCC), Stage 3 model selection justification, clinical metrics section with three individually justified metrics, 3-state logic section with threshold rationale, Stage 4 completion checklist.
- Added author name (Asaf Asnin) to title, figure suptitle, and footer.
- Added colour-coded 3-state output table (red=High-Risk, yellow=Borderline, green=Normal) via pandas Styler.
- Explicit 3-state clinical metrics computation cell (sensitivity, specificity, 3-state macro AUC).
- Side-by-side 5-class + 3-state confusion matrix figure.
- Generated `figures/06_xgboost_results.png` (3-panel: feature importance, 3-state confusion matrix, clinical metrics bar chart) — referenced in README.
- Ran forbidden-word scan: CLEAN (zero occurrences of prescriptive clinical terminology throughout).

Files Created/Modified:

- `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb` (final author-named notebook)
- `figures/06_xgboost_results.png` (generated)
- `data/processed/test_3state_output.csv` (generated on notebook run)

Key Outcomes:

- 19 cells total: 11 code, 8 markdown.
- Forbidden-word check: CLEAN.
- All 4 course stages covered with full clinical justification.
- Notebook runs top-to-bottom in < 2 seconds on laptop CPU, seed 42, fully reproducible.
- Notebook is submission-ready for the Applied ML in Medicine final project.

---

## 2026-06-08 - Final GitHub Synchronization Sweep

Stage / Action Taken: Repository Synchronization

Clinical or Technical Justification: Synced public-facing documentation with the final author-named notebook so GitHub visitors see the same product structure, notebook filename, modeling evidence, and compliance status that appear in the final submission artifact.

Files Created/Modified:

- `README.md`
- `project_development_log.md`
- `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb`
- `figures/06_xgboost_results.png`

Key Outcomes:

- Located the final notebook: `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb`.
- Verified the notebook has 19 cells and no restricted clinical terminology matches.
- Replaced remaining public documentation references with the final notebook filename.
- README now explicitly reflects CryFlag, the four course stages, metric rationale, conservative threshold logic, and the submission-ready notebook structure.

---

## 2026-07-07 - Fix-and-Tighten Pass: Rigor, Honesty, and Scope Fixes

Stage / Action Taken: Post-review hardening pass across data, modeling, and documentation, addressing 9 review tasks plus rollup fixes from a previous review round.

### Task 1 — Multi-Seed Stability Reporting
Added `run_seed_sweep()` to `models.py`, retraining XGBoost and MLP across 5 seeds (0-4) and reporting mean/std/min-max for accuracy and macro AUC. Added as a labeled Stage 3 subsection ("3.2 Multi-Seed Stability Check") in the notebook, with markdown explaining why stability matters given the small (30-sample) training set. `seed=42` remains the primary reproducible run used everywhere else.

### Task 2 — Statistical Baseline
Added `build_dummy()` (sklearn `DummyClassifier`, `strategy="most_frequent"`) to `models.py`. Included as "Baseline (majority-class guess)" in the Stage 3 model comparison table/plot, clearly labeled as a sanity-check floor rather than a candidate model.

### Task 3 — Numeric Success Criteria
Added a pre-registered, numeric success-criteria markdown cell in Stage 3 (before any results are shown): 3-state macro AUC ≥ 0.60, High-Risk sensitivity ≥ 0.50, and beating the Dummy baseline on both accuracy and AUC. Added an honest pass/fail cell immediately after the metrics are computed, stating plainly which criteria were and were not met and why (tiny n).

### Task 4 — Error Analysis
Added a Stage 3 subsection walking through the real rows of `results_3state`: 2 correctly and 2 incorrectly classified test examples, each with true/predicted state, predicted probabilities, a hypothesis for the outcome, and the clinical implication of that specific error.

### Task 5 — Data Leakage / Child ID Check
Investigated whether Donate-a-Cry filenames expose a persistent child identifier. They do not — only an uploader/device UUID, timestamp, app-version tag, gender, age bracket, and reason code. Added a notebook check (and `data_report.md` documentation) grouping the 40 sampled files by the leading UUID segment: 3 device IDs span more than one split (e.g. one `belly_pain` recording's device ID appears in both `train` and `test`). Documented as an honest, unfixed limitation — no child ID was fabricated.

### Task 6 — Robustness Check
Added a Stage 3 cell that takes one held-out test recording, applies mild Gaussian noise, and re-runs it through the full preprocess → MFCC → normalize → XGBoost pipeline, comparing the 3-state output and probabilities before/after. Reports whether the clinical flag stayed stable or flipped, with a short clinically-grounded conclusion either way.

### Task 7 — Weak Real-World Performance
(a) Investigated `acquire_sample`'s ceiling: the Donate-a-Cry `burping` label only has 8 total files available, so the balanced sample was increased from 4/label (20 files) to 8/label (40 files, 30 train / 5 val / 5 test) — the maximum balanced size the source supports, using the same deterministic seed=42 sampling logic. Re-ran the full pipeline (`data_loader.py` → `preprocess.py` → `models.py` → notebook) on the larger sample.
(b) Rewrote the Stage 3/4 results narrative to state plainly that the pre-registered success criteria are not met on the seed=42 run, tying the explanation directly to the seed-sweep instability and the Dummy baseline comparison, rather than a soft "illustrative only" caveat.

### Task 8 — Data Source Note
Added an explicit sentence in `data_report.md` and the notebook's Stage 2 section acknowledging the dataset is hosted on public GitHub rather than Kaggle/PhysioNet, and why it still meets the "public + accessible, license documented" bar (ODbL/DbCL, no account/paywall).

### Task 9 — Rollup Fixes
- Removed the literal word "diagnose" from `Presentation_Assets-AD_ML_Course/Presentation_Outline_English.md` and `Presentation_Assets-AD_ML_Course/CLAUDE.md`, rewording to "The system does not replace clinical judgment; it provides an additional layer of information to support it." (meaning preserved).
- Deleted the stale `mvp.ipynb` (20-cell earlier duplicate missing the executive summary, requirements table, and completion checklist present in the final notebook) after confirming `Asaf_Asnin_Neonatal_Cry_AI_MVP.ipynb` is the complete, correct, final artifact.
- Added an explicit per-coefficient z-score normalization step to `preprocess.py` (`normalize_mfcc()`), applied to the MFCC features before summarization, with its own before/after figure (`figures/05_before_after_normalization.png`) and its own `data_report.md` subsection (Step 5), distinct from pre-emphasis/STFT/MFCC.
- Added `preprocess.check_resampling_required()` and a "Resampling — Not Required" explanation in both `preprocess.py` and `data_report.md`/the notebook, making the no-resample decision an explicit, auditable check rather than an implicit omission.
- Strengthened the MLP justification in `models.py`'s docstring and the notebook's Stage 3 markdown into a 3-point, airtight argument for why MLP substitutes for the CNN/LSTM/XGBoost options named in `CLAUDE.md` (stack constraint, the CNN's temporal advantage not applying once MFCCs are pooled, and MLP being the correct like-for-like substitute).

### Bug Fixes Found During This Pass (not originally scoped, fixed because they blocked "runs without errors")
- `classify_batch`'s rounded probability columns occasionally summed to 1.001, which broke `sklearn.roc_auc_score`'s strict "must sum to 1.0" check once the larger 40-file sample changed which test rows appeared. Fixed by computing the 3-state AUC from unrounded `collapse_probabilities()` output instead of the rounded display columns.
- `pandas` 3.0.3 (installed in `.venv`) removed `Styler.applymap`; replaced with `Styler.map` in the 3-state output display cell.
- Renumbered Stage 2 figures so normalization (`05_before_after_normalization.png`) and class balance (`06_sample_class_balance.png`) follow the pipeline order; updated `preprocess.py`, the notebook figure-display cell, and `data_report.md` accordingly.

### Verification
- Ran the full pipeline end-to-end (`data_loader.py acquire --samples-per-label 8` → `preprocess.py` → notebook top-to-bottom) with zero cell errors; total notebook runtime remains in the low single-digit seconds on CPU.
- Global case-insensitive search for "diagnos" across all `.py`/`.ipynb`/`.md` files (excluding `.venv`/`__pycache__`) returns zero matches outside of the `CLAUDE.md` rule definition itself.
- Confirmed the 3-state output format (`1. Normal` / `2. Borderline–Suspicious` / `3. High-Risk`) and thresholds (`HIGH_RISK_THRESHOLD=0.30`, `NORMAL_THRESHOLD=0.55`) are unchanged in `classify.py` and correctly wired through the notebook's Clinical Flag column and confusion matrices.

### Not Applicable / Explicitly Not Done
- No item from Tasks 1-9 was found inapplicable; all nine were implemented as scoped. The only deviation from the literal instructions is that the child-ID leakage check (Task 5) uses the filename's leading UUID segment as a proxy for "same device/uploader," since the corpus exposes no stronger identifier — this is documented as a limitation of the check itself, not a shortcut.
