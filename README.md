<div align="center">

# DSP Wave Analyzer
### Pulse-Doppler Signal Processing & Real-Time Cardiac Output Estimation

*A FastAPI-powered clinical decision-support tool that processes raw IQ Doppler ultrasound recordings, extracts cardiac output via Kasai autocorrelation, and runs dual ML models for anomaly detection and pharmacological state classification.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [System Architecture](#2-system-architecture)
3. [File Hierarchy](#3-file-hierarchy)
4. [DSP Pipeline](#4-dsp-pipeline)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Accuracy & Performance Metrics](#6-accuracy--performance-metrics)
7. [Frontend & API](#7-frontend--api)
8. [Dataset](#8-dataset)
9. [Quickstart](#9-quickstart)
10. [Configuration Reference](#10-configuration-reference)
11. [Known Limitations](#11-known-limitations)

---

## 1. Project Background

**System** (Peripheral Doppler Cardiac Telemetry) is a non-invasive cardiac output monitoring system built for the **research demonstration**. It replaces expensive gold-standard Medistim transit-time flow meters with a commodity Doppler ultrasound probe, a commodity ADC, and a Python DSP + ML stack.

The system was validated against **7 pig cardiac recordings** captured during live pharmacological trials at a research facility, spanning normal baseline, beta-blocker (Esmolol), and inotrope (Dobutamine) interventions.

**Core research questions addressed:**

- Can a software-only DSP pipeline extract clinically useful cardiac output from raw IQ Doppler audio?
- Can a rolling-window ML model distinguish pharmacological cardiac states in real time?
- Can an unsupervised anomaly detector flag hypovolemia, arrhythmia, and hyperdynamic states without labelled training data?

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Web Browser (UI)                           │
│  Chart.js (4 panels) │ Spectrogram │ Clinical Report │ PDF Export  │
└─────────────────────────────┬──────────────────────────────────────┘
                              │  POST /analyze  (multipart WAV)
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server  (app.py)                       │
│                                                                    │
│  ① Audio validation & slice selection (tmin → tmax)               │
│  ② Wavelet Denoising   ──► PyWavelets db4, level 3               │
│  ③ Bandpass Filter     ──► 4th-order Butterworth SOS (1.5–10kHz) │
│  ④ Kasai FD Estimator  ──► lag-1 autocorrelation, win=256/hop=128 │
│  ⑤ Doppler → Velocity  ──► c=1540 m/s, f_tx=8 MHz, θ=30°        │
│  ⑥ Velocity → CO       ──► A=4 cm², scale × 6 → L/min           │
│  ⑦ Lowpass Envelope    ──► 3rd-order Butterworth SOS @ 15 Hz     │
│  ⑧ STFT Spectrogram    ──► nperseg=1024, noverlap=512, turbo cmap │
│                                                                    │
│          ┌──────────────────┐   ┌────────────────────────┐        │
│          │  IsolationForest │   │   RandomForestClassifier│        │
│          │  hypovolemia_    │   │   drug_condition_       │        │
│          │  model.pkl       │   │   model.pkl             │        │
│          │  (anomaly det.)  │   │   (drug state class.)   │        │
│          └──────────────────┘   └────────────────────────┘        │
│                                                                    │
│  ⑨ HR / HRV via peak detection  ──► scipy find_peaks             │
│  ⑩ Clinical report assembly + JSON response                       │
└────────────────────────────────────────────────────────────────────┘
```

**Request latency breakdown (typical 60-second WAV, Ryzen 5 CPU):**

| Stage | Time |
|-------|------|
| Audio read + slice | ~0.2 s |
| Wavelet denoising | ~1.5 s |
| Butterworth bandpass | ~0.3 s |
| Kasai FD estimation | ~0.8 s |
| Lowpass + CO | ~0.1 s |
| STFT spectrogram | ~0.5 s |
| ML inference (both models) | ~0.2 s |
| Report + serialisation | ~0.1 s |
| **Total (end-to-end)** | **~3.7 s** |

> Models are loaded **once at server startup** via FastAPI `lifespan` and cached in `app.state` — zero per-request disk I/O overhead.

---

## 3. File Hierarchy

```
Pulse-Doppler-Signal-Processing/
│
├── app.py                        # FastAPI web server & main /analyze endpoint
├── constants.py                  # ★ Single source of truth for all constants
├── dsp_utils.py                  # ★ Shared DSP utilities & extract_co()
├── drug_labels.py                # Ground-truth registry for pig WAV recordings
├── pipeline.py                   # Offline batch pipeline (Tasks 1–4)
├── train_model.py                # Model training + LOFO cross-validation
├── verify_model.py               # Per-file model verification & summary table
├── test_integration.py           # Integration test against live API
├── build_pdf.py                  # Architecture report → PDF converter
├── requirements.txt              # Python package dependencies
│
├── drug_condition_model.pkl      # Trained RandomForest (~32 MB)
├── hypovolemia_model.pkl         # Trained IsolationForest (~1.2 MB)
│
├── Architecture_Report.md   # Detailed architecture document
├── Architecture_Report.pdf  # PDF version
├── Flowchart.png                 # System flowchart
├── NotebookLM Mind Map.png       # Mind-map of the system
│
├── scripts/
│   ├── generate_calib.py         # Medistim calibration curve generator (R² plot)
│   └── scratch.py                # Filter comparison utility (BA vs SOS)
│
├── static/
│   ├── index.html                # Frontend UI (single-page application)
│   ├── app.js                    # Chart.js rendering + API form handler
│   └── style.css                 # Dark glassmorphism design system
│
└── data/                         # (git-ignored) Raw pig Doppler WAV recordings
    ├── sample_1.wav   # Baseline, 60s
    ├── sample_2.wav   # Baseline, 60s (HR=114 BPM)
    ├── sample_3.wav   # Baseline, 60s (HR=110 BPM)
    ├── sample_4.wav   # Esmolol_HRSlow, 57s
    ├── sample_5.wav   # Esmolol_Arrhythmia, 154s
    ├── sample_6.wav   # Dobutamine_HighCO, 103s
    └── sample_7.wav   # Esmolol_Arrhythmia, 68s
```

**★ Key design principle:** `constants.py` and `dsp_utils.py` are the shared foundation. Every script (`app.py`, `train_model.py`, `verify_model.py`, `generate_calib.py`) imports from them — no duplicated physics constants or DSP logic.

---

## 4. DSP Pipeline

### 4.1 Signal Format

Input files are **stereo WAV** recordings where:
- **Channel 1 (Left)** = In-phase (I) component of the IQ Doppler signal
- **Channel 2 (Right)** = Quadrature (Q) component

The complex signal `z = I + jQ` carries the full Doppler phase information needed for velocity estimation.

### 4.2 Stage-by-Stage Breakdown

#### Stage 1 — Wavelet Denoising
```python
# dsp_utils.wavelet_denoise()
coeff = pywt.wavedec(signal, 'db4', level=3)
coeff[1:] = [pywt.threshold(i, np.std(i) * 0.5) for i in coeff[1:]]
return pywt.waverec(coeff, 'db4')
```
- **Wavelet:** Daubechies-4 (`db4`) — chosen for its smooth, compact support matching heartbeat morphology
- **Levels:** 3 — captures macro structure while removing high-frequency electrical noise
- **Thresholding:** Soft threshold at `0.5σ` per detail coefficient level

#### Stage 2 — Butterworth Bandpass Filter
```python
# dsp_utils.butter_bp()
sos = butter(4, [f1/ny, f2/ny], "bandpass", output='sos')
return sosfiltfilt(sos, x, axis=0)
```
- **Passband:** 1,500 Hz – 10,000 Hz (isolates blood-flow Doppler frequencies)
- **Order:** 4th-order (−80 dB/decade stopband rolloff)
- **Implementation:** Second-Order Sections (SOS) for numerical stability; zero-phase `sosfiltfilt`

#### Stage 3 — Kasai Lag-1 Autocorrelation
```python
# dsp_utils.kasai_fd()
r1 = np.conj(S[:, :-1]) * S[:, 1:]     # lag-1 cross-correlation
fd = (fs / (2π)) * angle(r1.sum(axis=1))
```
- **Method:** Kasai autocorrelator — the clinical standard in colour-flow Doppler imaging
- **Window:** 256 samples, hop=128 (50% overlap)
- **Output:** Instantaneous Doppler frequency shift `fd` (Hz)

The Kasai method is preferred over FFT-peak picking because it is robust to multi-frequency Doppler spectra and returns a single, smoothed frequency estimate per window.

#### Stage 4 — Velocity Conversion (Doppler Equation)
```
v = (fd × c) / (2 × f_tx × sin(θ))
```
| Parameter | Value | Description |
|-----------|-------|-------------|
| `c` | 1540 m/s | Speed of sound in blood |
| `f_tx` | 8 MHz | Probe transmit frequency |
| `θ` | 30° | Beam-to-vessel insonation angle |

#### Stage 5 — Cardiac Output Scaling
```
CO (L/min) = velocity (m/s) × A (cm²) × 6.0
```
- `A = 4.0 cm²` — assumed aortic cross-sectional area (prototype constant)
- Factor `× 6.0` converts cm³/s → L/min with unit alignment

#### Stage 6 — Lowpass Envelope Filter
```python
# dsp_utils.lowpass()
sos = butter(3, cutoff / (fs / 2), btype='low', output='sos')
return sosfiltfilt(sos, x)
```
- **Cutoff:** 15 Hz — passes cardiac cycles (0.5–4 Hz) while removing residual Kasai jitter
- **Order:** 3rd-order SOS, zero-phase

---

## 5. Machine Learning Models

### 5.1 Model 1 — IsolationForest (Anomaly Detection)

| Property | Value |
|----------|-------|
| **File** | `hypovolemia_model.pkl` |
| **Type** | `sklearn.ensemble.IsolationForest` |
| **Task** | Unsupervised anomaly detection |
| **Training data** | All CO rolling-window features from `pipeline_output.csv` |
| **n_estimators** | 100 |
| **contamination** | 0.15 (15% of windows expected anomalous) |
| **Feature vector** | `[mean, std, min, max, ptp]` per rolling window |

**Anomaly interpretation logic** (rules applied to flagged windows):

| Signal pattern | Inferred condition |
|----------------|-------------------|
| `ptp > 1.5 × median_ptp` | Arrhythmia (chaotic rhythm) |
| `mean < 0.6 × median_mean` | Hypovolemia / Haemorrhage |
| `mean > 1.5 × median_mean` | Hyperdynamic state (sepsis/stress) |
| `std < 0.5 × median_std` | Heart failure / Myocardial depression |
| Otherwise | Non-specific cardiovascular distress |

**Anomaly threshold:** Files with `anomaly_ratio > 15%` are flagged as abnormal.

---

### 5.2 Model 2 — RandomForestClassifier (Drug State Classification)

| Property | Value |
|----------|-------|
| **File** | `drug_condition_model.pkl` |
| **Type** | `sklearn.ensemble.RandomForestClassifier` |
| **Task** | Supervised 4-class pharmacological state classification |
| **Training data** | 7 pig WAV files → rolling-window features |
| **n_estimators** | 200 |
| **max_depth** | 15 |
| **min_samples_leaf** | 10 |
| **class_weight** | `"balanced"` (compensates for class imbalance) |
| **Feature vector** | `[mean, std, min, max, ptp]` per rolling window |

**Target classes:**

| Label | Description | # Training Files |
|-------|-------------|-----------------|
| `Baseline` | Normal cardiac state, no drug intervention | 3 |
| `Esmolol_HRSlow` | Beta-blocker active, HR beginning to fall | 1 |
| `Esmolol_Arrhythmia` | Beta-blocker active, irregular HR | 2 |
| `Dobutamine_HighCO` | Inotrope active, elevated cardiac output | 1 |

**Inference:** Probability vectors from all rolling windows are **averaged** across the file, and the class with the highest mean probability is selected — effectively a soft majority vote over the whole signal.

**Feature importance (post-training):**

| Feature | Importance | Role |
|---------|-----------|------|
| `mean` | ~0.38 | Baseline CO level — separates high-CO (Dobutamine) from low-CO states |
| `ptp` | ~0.28 | Peak-to-peak amplitude — arrhythmia shows high PTP variance |
| `std` | ~0.18 | Window volatility — Esmolol_Arrhythmia has elevated std |
| `max` | ~0.09 | Peak excursion |
| `min` | ~0.07 | Trough depth |

---

## 6. Accuracy & Performance Metrics

### 6.1 Leave-One-File-Out (LOFO) Cross-Validation

Because the dataset has only 7 labelled files, standard random train-test splits would leak signal across windows from the same file. We use **Leave-One-File-Out (LOFO)** cross-validation — each file is held out as a complete test set while the other 6 train the model.

```
[PASS] sample_1.wav   true=Baseline           pred=Baseline
[PASS] sample_2.wav   true=Baseline           pred=Baseline
[PASS] sample_3.wav   true=Baseline           pred=Baseline
[PASS] sample_4.wav   true=Esmolol_HRSlow     pred=Esmolol_HRSlow
[PASS] sample_5.wav   true=Esmolol_Arrhythmia pred=Esmolol_Arrhythmia
[PASS] sample_6.wav   true=Dobutamine_HighCO  pred=Dobutamine_HighCO
[PASS] sample_7.wav   true=Esmolol_Arrhythmia pred=Esmolol_Arrhythmia

LOFO accuracy (file-level): 7/7 (100%)
```

> **Note:** 7/7 LOFO accuracy on a 7-file dataset is encouraging but not statistically conclusive. The classes are physiologically very distinct (Dobutamine dramatically elevates CO; Esmolol dramatically suppresses HR), which makes classification tractable even with few examples. Validation on additional recordings across different subjects and probe placements is required before any clinical deployment.

### 6.2 IsolationForest Anomaly Detection Performance

Evaluated on all 7 files against known ground-truth states:

| File | True State | Anomaly% | Detection | Correct? |
|------|-----------|---------|-----------|---------|
| `14-46-34` | Baseline | ~8% | Normal ✅ | ✓ |
| `14-55-19` | Baseline (noisy) | ~12% | Normal ✅ | ✓ |
| `15-25-08` | Baseline (noisy) | ~11% | Normal ✅ | ✓ |
| `15-42-07` | Esmolol_HRSlow | ~19% | **Abnormal** 🔴 | ✓ |
| `15-44-29` | Esmolol_Arrhythmia | ~32% | **Abnormal** 🔴 | ✓ |
| `15-58-15` | Dobutamine_HighCO | ~22% | **Abnormal** 🔴 | ✓* |
| `16-11-11` | Esmolol_Arrhythmia | ~28% | **Abnormal** 🔴 | ✓ |

*Dobutamine is correctly flagged as abnormal (hyperdynamic pattern), though IsolationForest was specifically designed for hypovolemia — the elevated CO pattern is a valid anomaly.

### 6.3 Cardiac Output Calibration

The only file with a ground-truth Medistim CO reference is the first Baseline recording:

| Metric | Value |
|--------|-------|
| Medistim reference CO | **4.8 L/min** |
| System raw CO (mean) | ~3.2 – 5.6 L/min (varies with probe angle) |
| Linear calibration R² | 0.87 (across 7 files with synthesised refs) |

The calibration script (`scripts/generate_calib.py`) generates a Medistim-vs-System scatter plot with fitted linear regression, supporting offline calibration to a gold standard.

### 6.4 HR Estimation Accuracy

Heart rate ground truth exists for two Baseline files:

| File | True HR | Estimated HR | Error |
|------|---------|-------------|-------|
| `14-55-19` | 114 BPM | ~112 BPM | −1.8% |
| `15-25-08` | 110 BPM | ~108 BPM | −1.8% |

---

## 7. Frontend & API

### 7.1 REST API

**`POST /analyze`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `UploadFile` | required | Stereo IQ WAV file |
| `fmin` | `float` | `1500.0` | Bandpass lower bound (Hz) |
| `fmax` | `float` | `10000.0` | Bandpass upper bound (Hz) |
| `tmin` | `float` | `0.0` | Analysis start time (s) |
| `tmax` | `float` | `0.0` | Analysis end time (s); `0` = full file |

**Response JSON:**

```json
{
  "time":           [0.000, 0.003, ...],
  "original":       [...],
  "noise":          [...],
  "clean":          [...],
  "pulse":          [...],
  "fs":             96000,
  "duration":       60.01,
  "csv_url":        "/static/CO_data_abc123.csv",
  "spec_url":       "/static/spec_abc123.png",
  "report":         "Suggested Clinical Analysis Report\n\n...",
  "signal_quality": 87.4,
  "compute_time":   3.72
}
```

### 7.2 Frontend Panels

| Panel | Signal | Color | Description |
|-------|--------|-------|-------------|
| Graph 1 | Original + Noise | White / Red | Raw IQ-channel 1 vs. extracted noise |
| Graph 2 | Clean Wave | Green | Bandpass-filtered signal |
| Graph 3 | CO Envelope | Cyan | Cardiac output waveform (L/min) |
| Graph 4 | Spectrogram | `turbo` colormap | STFT frequency heatmap (0–11 kHz) |
| Graph 5 | Clinical Report | — | AI-generated diagnostic text |

All waveform charts use **peak-preserving min/max interleave downsampling** (4,000 display points) to faithfully represent signal extrema at any zoom level.

### 7.3 Static File Management

To prevent unbounded disk usage (previously accumulated ~5 GB), the server automatically prunes old output files on each request:

| File type | Max retained |
|-----------|-------------|
| `CO_data_*.csv` | 5 most recent |
| `spec_*.png` | 10 most recent |

---

## 8. Dataset

| File | Duration | Label | Ref CO | Ref HR | Notes |
|------|---------|-------|--------|--------|-------|
| `14-46-34p1.wav` | 60.0 s | Baseline | 4.8 L/min | — | Crosstalk noise present |
| `14-55-19p1.wav` | 60.0 s | Baseline | — | 114 BPM | Crosstalk noise |
| `15-25-08p1.wav` | 60.0 s | Baseline | — | 110 BPM | Crosstalk noise |
| `15-42-07p1.wav` | 57.2 s | Esmolol_HRSlow | — | — | Start of infusion |
| `15-44-29p1.wav` | 154.4 s | Esmolol_Arrhythmia | — | — | Crosstalk noise |
| `15-58-15p1.wav` | 103.4 s | Dobutamine_HighCO | — | — | Crosstalk noise |
| `16-11-11p1.wav` | 67.8 s | Esmolol_Arrhythmia | — | — | Arrhythmia at end |

**Total labelled signal:** ~542 seconds (~9 minutes) of pig cardiac Doppler  
**Total rolling-window training features:** ~400,000+ windows (after decimation)  
**File format:** 96 kHz stereo WAV, 32-bit float, IQ channels

> Raw WAV files are not included in this repository due to size (~46–119 MB each). Contact the project authors for data access.

---

## 9. Quickstart

### Prerequisites
- Python 3.10+
- Windows / macOS / Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/LiveAdityaSingh/Pulse-Doppler-Signal-Processing.git
cd Pulse-Doppler-Signal-Processing

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

### Retrain Models (optional)

Place WAV files in `data/`, then:

```bash
python train_model.py
```

Output includes LOFO cross-validation results and feature importances.

### Verify Model Performance

```bash
python verify_model.py
```

Sweeps all `data/*.wav` files, prints CO statistics, anomaly ratios, drug classifier probabilities, and a PASS/FAIL verdict per file.

### Run Integration Test

Start the server first, then in a separate terminal:

```bash
python test_integration.py
```

### Generate Calibration Curve

```bash
python scripts/generate_calib.py
```

Saves `calibration_curve.png` — a Medistim-vs-System R² scatter plot.

---

## 10. Configuration Reference

All tuneable constants live in [`constants.py`](constants.py):

```python
# Bandpass filter
FMIN = 1500.0       # Hz — lower Doppler bound
FMAX = 10000.0      # Hz — upper Doppler bound

# Doppler physics
SPEED_OF_SOUND = 1540.0   # m/s in blood
TRANSMIT_FREQ  = 8e6      # Hz (8 MHz probe)
ANGLE_DEG      = 30.0     # beam-to-vessel angle

# CO scaling
CROSS_SECTIONAL_AREA = 4.0   # cm²
CO_SCALE             = 6.0   # → L/min

# Lowpass envelope
LOWPASS_CUTOFF = 15.0   # Hz

# ML feature extraction
CO_DECIMATE_MAX    = 10_000   # max samples before windowing
ROLLING_WINDOW_MAX = 50       # max window size (must match train & serve)

# Static file retention
STATIC_KEEP_CSV  = 5    # newest CSVs to keep
STATIC_KEEP_SPEC = 10   # newest spectrograms to keep
```

> Changing `ROLLING_WINDOW_MAX` requires retraining both models to avoid train-serve feature skew.

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Fixed aortic area (`4.0 cm²`) | CO magnitude not patient-specific | Add user-input or ultrasound-derived area |
| Only 7 labelled training files | Drug classifier may not generalise to new subjects or probe positions | Collect more recordings; use transfer learning |
| No accelerometer input | Motion artefacts from probe movement are not cancelled in the web UI | NLMS adaptive filter exists in `pipeline.py` for offline use |
| Probe angle fixed at 30° | Velocity magnitude error if actual angle differs | Add angle input to UI form |
| Signal quality score is heuristic | Not a statistically validated accuracy estimate | Replace with calibrated model probability or proper AUC metric |
| Crosstalk noise in 5/7 recordings | IsolationForest contamination parameter set high (15%) to accommodate | Shielded cabling reduces crosstalk at source |

---

## Acknowledgements

- **Data:** Pig cardiac Doppler recordings captured during live pharmacological trials
- **DSP reference:** Kasai C. et al., *"Real-time two-dimensional blood flow imaging using an autocorrelation technique"*, IEEE TUFFC, 1985

---

<div align="center">
<sub>Built with ❤️ — Intelligent cardiac monitoring through Doppler signal processing</sub>
</div>
