# Data Report: Neonatal Cry Acoustics

## Scope

This report covers the Codex-owned data layer only: data acquisition, loading, deterministic preprocessing, before/after visuals, and a ready-to-model feature table. Model choice, metric justification, thresholds, and the final 3-state clinical output remain Claude-owned.

## Dataset

Source: [gveres/donateacry-corpus](https://github.com/gveres/donateacry-corpus)

Subset used: `donateacry_corpus_cleaned_and_updated_data`

Access: public GitHub repository; the sample is downloaded directly from `raw.githubusercontent.com` by `data_loader.py`.

License noted by the source repository: Open Database License (ODbL) for the database and Database Contents License (DbCL) for individual contents.

**Source note:** this dataset is hosted on a public GitHub repository rather than Kaggle or PhysioNet (the examples named in the course brief). It still satisfies the "public + accessible, license documented" requirement: it is openly downloadable with no account or paywall, and it carries an explicit ODbL/DbCL license covering both the database structure and its individual audio files — the same accessibility bar Kaggle/PhysioNet datasets are held to, just on a different public platform.

Important data caveat: the repository describes the files as user-uploaded mobile-app audio with contributor-provided tags. For this course project, the data layer treats labels as corpus labels only and does not treat them as verified clinical truth.

## Local Sample

The current workspace contains a deterministic balanced sample, increased from an initial 20-file sample (4/label) to 40 files (8/label) — the maximum balanced size the source corpus supports, since `burping` only has 8 total files available:

| Item | Value |
|---|---:|
| WAV files | 40 |
| Labels | 5 |
| Files per label | 8 |
| Train files | 30 |
| Validation files | 5 |
| Test files | 5 |
| Source sample rate | 8,000 Hz |
| Channels | 1 mono |
| Audio subtype | PCM 16-bit |
| Duration range | 6.52-7.06 seconds |
| Mean duration | 6.92 seconds |

Class balance:

| Label | Files |
|---|---:|
| `belly_pain` | 8 |
| `burping` | 8 |
| `discomfort` | 8 |
| `hungry` | 8 |
| `tired` | 8 |

Split policy: fixed seed `42`, stratified by corpus label. With 8 files per label, each label contributes 6 train, 1 validation, and 1 test file.

Generated files:

- `data/metadata.csv`: one row per audio file with source path, local path, label, split, age tag, gender tag, sample rate, duration, format, and file size.
- `data/processed/mfcc_features.csv`: one row per audio file with 13 (z-score normalized) MFCC means and 13 MFCC standard deviations.
- `figures/`: before/after preprocessing visuals.

## Data Quality Notes

Missing values: none found in the generated metadata.

Duration outliers: none in the sampled files. All files are short neonatal cry clips between 6.52 and 7.06 seconds.

Format consistency: all sampled files are mono WAV, 8 kHz, PCM 16-bit. Because the source sample rate is consistent, preprocessing keeps 8 kHz rather than artificially upsampling.

Label limitation: the sample is intentionally balanced for a small classroom MVP. It should not be presented as the natural label distribution of the full public corpus.

**Known limitation — no persistent child ID.** Donate-a-Cry filenames (parsed by `data_loader.py`'s `parse_filename`) expose an uploader/device UUID, a Unix timestamp, an app-version tag, gender, age bracket, and a reason code — but no field identifies the same infant across recordings. A check across the 40 sampled files (grouping by the leading UUID segment of the filename, the closest available proxy for "same uploading device/household") found 3 device IDs whose recordings span more than one split (e.g. the same device ID appears in both `train` and `test` for one `belly_pain` recording). Because the source data does not expose a persistent child identifier, we cannot rule out the same infant appearing in both train and test — **this is flagged as a known limitation, not fixed**, because the source data doesn't support a proper fix without fabricating an identifier the corpus does not provide. See the corresponding check cell in the MVP notebook (Stage 2) for the full per-device breakdown.

## Lecture 4 Preprocessing Pipeline

### Step 1: Audio Loading and Sampling-Rate Check

Implementation: `librosa.load(..., sr=8000, mono=True)`.

Reasoning: the sampled source files are already 8 kHz mono. Keeping that rate preserves the real recorded bandwidth, keeps processing CPU-light, and avoids displaying empty high-frequency bands that would appear after unnecessary upsampling.

**Resampling — Not Required.** `preprocess.check_resampling_required()` explicitly checks every sampled file's source rate against the 8 kHz target. All 40 files pass at exactly 8,000 Hz, so no resampling step is applied — this is a verified, auditable decision (the check would fail loudly if a future corpus sample mixed sample rates) rather than a silently omitted step.

Visual:

![Before raw waveform](figures/01_before_raw_waveform.png)

### Step 2: Pre-Emphasis Filter

Implementation:

```text
y[t] = x[t] - 0.97 * x[t-1]
```

Reasoning: mobile recordings can contain low-frequency handling noise and large slow amplitude swings. Pre-emphasis highlights fast acoustic changes and cry harmonics while preserving the timing of cry bursts. This is useful before spectral analysis because it makes short high-frequency events more visible.

Before/After visual:

![Before and after pre-emphasis](figures/02_before_after_pre_emphasis.png)

### Step 3: STFT Spectrogram

Implementation: Hann-window STFT with `n_fft=512` and `hop_length=128`.

At 8 kHz this is a 64 ms analysis window with a 16 ms hop. That is short enough to show changing cry bursts while still providing a readable frequency structure for the report.

Reasoning: a waveform shows amplitude over time, but cry audio changes in both time and frequency. The STFT spectrogram converts the signal into a compact time-frequency representation that can be inspected visually and used as a clear intermediate data product.

Before/After visual:

![Before waveform and after STFT spectrogram](figures/03_before_after_stft_spectrogram.png)

### Step 4: MFCC Extraction

Implementation: 13 MFCC coefficients from the pre-emphasized signal using the same STFT window and hop.

Reasoning: MFCCs summarize the spectral envelope into a small feature vector. For a lightweight CPU-trainable project, MFCC mean and standard deviation features are much smaller than raw spectrogram pixels while still preserving useful acoustic structure.

Before/After visual:

![Before STFT and after MFCC map](figures/04_before_after_mfcc.png)

### Step 5: Per-Coefficient Normalization (z-score)

Implementation: `preprocess.normalize_mfcc()` — for each of the 13 MFCC coefficients, subtract that coefficient's own mean and divide by its own standard deviation across the time axis, before computing the mean/std summary features.

Reasoning: raw MFCC magnitude is sensitive to recording-level loudness (microphone gain, distance from the infant), which varies across the many different phones that contributed to Donate-a-Cry. This step is distinct from MFCC extraction itself: MFCC decides *which* acoustic information to keep, normalization decides *how it is scaled* before the model sees it. Removing loudness bias means the model compares cry shape, not recording volume — volume alone is not a clinically meaningful signal of pain or distress.

Before/After visual:

![Before and after normalization](figures/05_before_after_normalization.png)

## Ready-to-Model Output

The current feature table is:

```text
data/processed/mfcc_features.csv
```

Shape: 40 rows x 32 columns.

Columns include:

- `path`, `filename`, `label`, `split`
- `sample_rate`, `n_frames`
- `mfcc_01_mean` through `mfcc_13_mean` (post-normalization)
- `mfcc_01_std` through `mfcc_13_std` (post-normalization)

This table is deterministic and suitable for Claude's next stage: simple model comparison and downstream 3-state mapping.

## Presentation Figures

![Sample class balance](figures/06_sample_class_balance.png)

Recommended figure order for Stage 2 slides:

1. Raw waveform: `figures/01_before_raw_waveform.png`
2. Pre-emphasis before/after: `figures/02_before_after_pre_emphasis.png`
3. STFT before/after: `figures/03_before_after_stft_spectrogram.png`
4. MFCC before/after: `figures/04_before_after_mfcc.png`
5. Normalization before/after: `figures/05_before_after_normalization.png`
6. Sample class balance: `figures/06_sample_class_balance.png`
