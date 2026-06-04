import os
import io
import glob
import time
import uuid
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, stft          # Issue #9: top-level import
plt.switch_backend('Agg')

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from dsp_utils import butter_bp, kasai_fd, velocity_from_fd, wavelet_denoise, lowpass
from constants import (
    FMIN, FMAX,
    CROSS_SECTIONAL_AREA, CO_SCALE,
    STFT_NPERSEG, STFT_NOVERLAP,
    CO_DECIMATE_MAX, ROLLING_WINDOW_MAX,
    STATIC_KEEP_CSV, STATIC_KEEP_SPEC,
)

# Drug-condition label metadata (descriptions + icons used in report)
try:
    from drug_labels import LABEL_DESCRIPTIONS, LABEL_ICONS
except ImportError:
    LABEL_DESCRIPTIONS = {}
    LABEL_ICONS = {}


# ── Static file cleanup ────────────────────────────────────────────────────────
# Issue #1: Prevent unbounded accumulation of output files in static/

def cleanup_static(pattern: str, keep: int) -> None:
    """Delete oldest files matching *pattern* in static/, keeping only *keep* newest."""
    files = sorted(glob.glob(os.path.join("static", pattern)), key=os.path.getmtime)
    for f in files[:-keep] if keep > 0 else files:
        try:
            os.remove(f)
        except OSError:
            pass  # Best-effort; don't crash the request if removal fails


# ── Model caching via lifespan ─────────────────────────────────────────────────
# Issue #8: Load both models once at startup instead of on every request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load models into app.state so handlers can access them via request.app.state
    try:
        app.state.hypo_model = joblib.load("hypovolemia_model.pkl")
        print("[startup] hypovolemia_model.pkl loaded")
    except Exception as e:
        app.state.hypo_model = None
        print(f"[startup] WARNING: hypovolemia_model.pkl not found — {e}")

    try:
        app.state.drug_bundle = joblib.load("drug_condition_model.pkl")
        print("[startup] drug_condition_model.pkl loaded")
    except Exception as e:
        app.state.drug_bundle = None
        print(f"[startup] WARNING: drug_condition_model.pkl not found — {e}")

    yield  # Application runs here
    # (cleanup on shutdown if needed)


app = FastAPI(title="PerDeCT Wave Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


# ── Helpers ────────────────────────────────────────────────────────────────────

def downsample(data, max_points=2000):
    """Peak-preserving min/max interleave downsample for visualisation."""
    n = len(data)
    if n <= max_points:
        return data.tolist()
    chunk_size = n // (max_points // 2)
    usable_len = chunk_size * (n // chunk_size)
    reshaped = data[:usable_len].reshape(-1, chunk_size)
    maxs = np.max(reshaped, axis=1)
    mins = np.min(reshaped, axis=1)
    downsampled = np.empty((maxs.size + mins.size,), dtype=data.dtype)
    downsampled[0::2] = maxs
    downsampled[1::2] = mins
    return downsampled.tolist()


def build_rolling_features(co_signal, tmin, tmax):
    """
    Extract rolling-window features from a cardiac-output signal.

    Uses the same decimation and window-size formula as train_model.py
    (ROLLING_WINDOW_MAX cap from constants.py) to avoid train-serve skew.
    Issue #7 fix.
    """
    dec    = max(1, len(co_signal) // CO_DECIMATE_MAX)
    co_sub = co_signal[::dec]
    time_s = np.linspace(tmin, tmax, len(co_sub))

    df = pd.DataFrame({'co': co_sub, 'time': time_s})
    df['co'] = df['co'].ffill().fillna(0)

    wp      = max(10, min(ROLLING_WINDOW_MAX, len(co_sub) // 10))
    rolling = df['co'].rolling(window=wp, min_periods=max(1, wp // 2))

    feats = pd.DataFrame({
        'mean': rolling.mean(),
        'std':  rolling.std().fillna(0),
        'min':  rolling.min(),
        'max':  rolling.max(),
        'ptp':  rolling.max() - rolling.min(),
        'time': df['time'],
    }).dropna()

    return feats


def generate_clinical_report(co_signal, fs, tmin, tmax, hypo_model, drug_bundle):
    """
    Build a markdown-style clinical analysis report for the analyzed CO segment.

    Parameters
    ----------
    co_signal   : np.ndarray  Cardiac-output time-series (L/min)
    fs          : int         Sample rate (Hz)
    tmin, tmax  : float       Analysed interval bounds (seconds)
    hypo_model  : IsolationForest or None
    drug_bundle : dict {"model", "label_encoder"} or None

    Returns
    -------
    report          : str   Formatted report text
    signal_quality  : float Heuristic signal-quality score (0–100)
    """
    distance = int(fs * 0.33)
    peaks, properties = find_peaks(
        co_signal, distance=distance, prominence=np.max(co_signal) * 0.15
    )

    if len(peaks) < 2:
        return (
            "Insufficient data to generate a reliable clinical report. "
            "Please analyze a longer time interval or check signal quality.",
            0.0,
        )

    peak_times   = peaks / fs
    rr_intervals = np.diff(peak_times)

    avg_rr     = np.mean(rr_intervals)
    hr_bpm     = 60.0 / avg_rr
    hrv_ms     = np.std(rr_intervals) * 1000

    peak_velocity = np.max(co_signal) / (CROSS_SECTIONAL_AREA * CO_SCALE)

    velocity_signal = co_signal / (CROSS_SECTIONAL_AREA * CO_SCALE)
    vti_complete    = np.trapz(np.abs(velocity_signal), dx=1.0 / fs)
    duration        = tmax - tmin
    vti_co          = (vti_complete / duration) * CROSS_SECTIONAL_AREA * CO_SCALE if duration > 0 else 0

    avg_prom     = np.mean(properties['prominences']) if len(peaks) > 0 else 0
    max_amp      = max(np.max(co_signal), 1e-9)
    clarity_ratio = avg_prom / max_amp

    # Heuristic signal-quality score (clearly labelled as heuristic, Issue #4)
    base_quality      = min(99.9, clarity_ratio * 100 + 45.0)
    stability_penalty = min(15.0, hrv_ms / 12.0)

    # ── IsolationForest (hypovolemia / anomaly) ────────────────────────────────
    ml_report          = ""
    ml_quality_boost   = 0.0
    anomaly_ratio      = 0.0

    if hypo_model is not None:
        try:
            features = build_rolling_features(co_signal, tmin, tmax)
            if len(features) > 0:
                feat_vals            = features[['mean', 'std', 'min', 'max', 'ptp']].values
                preds                = hypo_model.predict(feat_vals)
                features['anomaly'] = preds
                anomalies            = features[features['anomaly'] == -1]
                anomaly_ratio        = len(anomalies) / len(features)

                if anomaly_ratio > 0.15:
                    max_ptp_row    = anomalies.loc[anomalies['ptp'].idxmax()]
                    time_of_anomaly = int(max_ptp_row['time'])
                    val_ptp  = max_ptp_row['ptp']
                    val_mean = max_ptp_row['mean']
                    med_ptp  = features['ptp'].median()
                    med_mean = features['mean'].median()

                    if val_ptp > med_ptp * 1.5:
                        condition_guess = "arrythmia"
                        description     = "more difference between peaks"
                        fuel_text       = "a heart experiencing chaotic rhythm"
                    elif val_mean < med_mean * 0.6:
                        condition_guess = "hypovolemia or hemorrhage"
                        description     = "a massive drop in baseline pressure"
                        fuel_text       = "a heart trying to pump with not enough fuel"
                    elif val_mean > med_mean * 1.5:
                        condition_guess = "hyperdynamic state (e.g., severe stress or sepsis)"
                        description     = "an unusually elevated cardiac output"
                        fuel_text       = "a heart overworking aggressively"
                    elif max_ptp_row['std'] < features['std'].median() * 0.5:
                        condition_guess = "heart failure or severe myocardial depression"
                        description     = "extremely weak contraction pulses"
                        fuel_text       = "a heart muscle failing to generate adequate force"
                    else:
                        condition_guess = "non-specific cardiovascular distress"
                        description     = "abnormal and erratic waveform morphology"
                        fuel_text       = "a heart pumping irregularly"

                    ml_report       = (
                        f"- 🔴 **ML Analysis:** The AI detected an abnormal CO profile "
                        f"(anomaly ratio {anomaly_ratio * 100:.1f}%). This looks like "
                        f"{fuel_text}. This condition may be {condition_guess}. "
                        f"Inferred from CO at {time_of_anomaly}s where there is {description}.\n"
                    )
                    ml_quality_boost = min(15.0, anomaly_ratio * 40.0)
                else:
                    ml_report       = (
                        "- 🟢 **ML Analysis:** The AI found the waveform CO pattern stable, "
                        "with no severe hypovolemic signatures detected.\n"
                    )
                    ml_quality_boost = 10.0
        except Exception as e:
            ml_report = f"- ⚪ ML Analysis unavailable ({e})\n"
    else:
        ml_report = "- ⚪ ML Analysis unavailable (model not loaded)\n"

    # ── Drug-condition classifier ──────────────────────────────────────────────
    drug_report = ""
    if drug_bundle is not None:
        try:
            drug_clf = drug_bundle["model"]
            le       = drug_bundle["label_encoder"]

            features = build_rolling_features(co_signal, tmin, tmax)
            if len(features) > 0:
                probs      = drug_clf.predict_proba(features[['mean', 'std', 'min', 'max', 'ptp']].values)
                avg_probs  = probs.mean(axis=0)
                pred_idx   = int(np.argmax(avg_probs))
                pred_label = le.inverse_transform([pred_idx])[0]
                confidence = avg_probs[pred_idx] * 100
                icon       = LABEL_ICONS.get(pred_label, "💊")
                desc       = LABEL_DESCRIPTIONS.get(pred_label, "Drug influence pattern detected.")

                drug_report = (
                    f"- {icon} **Drug Influence ({confidence:.0f}% confidence):** "
                    f"Predicted state — **{pred_label.replace('_', ' ')}**. {desc}\n"
                )
        except Exception as e:
            drug_report = f"- ⚪ Drug classifier unavailable ({e})\n"
    else:
        drug_report = "- ⚪ Drug classifier unavailable (model not loaded)\n"

    # Issue #4: Heuristic score clearly labelled — NOT presented as measured accuracy
    signal_quality = min(99.9, base_quality + ml_quality_boost - stability_penalty)

    # ── Format report ──────────────────────────────────────────────────────────
    report  = "Suggested Clinical Analysis Report\n\n"
    report += f"**Analyzed Interval:** {tmin:.1f}s to {tmax:.1f}s\n"
    report += f"**Peak Velocity Observed:** {peak_velocity:.3f} m/s\n"
    report += f"**Complete VTI of the Wave:** {vti_complete:.3f} meters\n"
    report += f"**VTI-Derived Cardiac Output:** {vti_co:.2f} L/min\n"
    report += f"**Detected Pulses:** {len(peaks)} beats\n"
    report += f"**Estimated Pulse Rate:** {hr_bpm:.1f} BPM\n"
    report += f"**Heart Rate Variability (HRV):** {hrv_ms:.1f} ms\n"
    report += f"**Heuristic Signal Quality Score:** {signal_quality:.1f} / 100\n\n"
    report += "**Diagnostic Observations:**\n"

    if hr_bpm > 100:
        report += (
            "- 🔴 **Tachycardia Indication:** The resting heart rate is elevated above 100 BPM. "
            "Consider assessing patient state for exertion, stress, or underlying conditions.\n"
        )
    elif hr_bpm < 60:
        report += (
            "- 🟡 **Bradycardia Indication:** The resting pulse is below 60 BPM. "
            "This can be normal in athletic individuals but may require context.\n"
        )
    else:
        report += (
            "- 🟢 **Normal Rate:** The estimated pulse rate falls within the standard "
            "resting physiological range (60–100 BPM).\n"
        )

    if hrv_ms > 150:
        report += (
            "- 🟡 **High Variability (Arrhythmia Risk):** Beat-to-beat intervals show "
            "significant variance. Extreme or chaotic variance combined with visual "
            "irregularities might indicate Atrial Fibrillation or ectopic beats.\n"
        )
    elif hrv_ms < 20:
        report += (
            "- 🟡 **Low Variability:** The heart rhythm is extremely rigid. "
            "Chronically low HRV can correlate with physiological stress or fatigue.\n"
        )
    else:
        report += (
            "- 🟢 **Stable Rhythm:** The heart rhythm appears regular with normal, "
            "healthy physiological variability.\n"
        )

    report += ml_report
    report += drug_report
    return report, signal_quality


# ── Main endpoint ──────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_wave(
    request: Request,
    file: UploadFile = File(...),
    fmin: float = Form(FMIN),
    fmax: float = Form(FMAX),
    # Issue #2: noise_factor removed — fft_noise_reduction is not used in this pipeline.
    tmin: float = Form(0.0),
    tmax: float = Form(0.0),
):
    # Issue #1: Clean up oldest static output files before writing new ones
    cleanup_static("CO_data_*.csv",  STATIC_KEEP_CSV)
    cleanup_static("spec_*.png",     STATIC_KEEP_SPEC)

    start_t = time.time()

    # Issue #10: Validate the uploaded file is actually readable audio
    content = await file.read()
    try:
        data, fs = sf.read(io.BytesIO(content), always_2d=True, dtype="float32")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read audio file '{file.filename}': {e}. "
                   "Please upload a valid WAV file."
        )

    if data.shape[1] < 2:
        raise HTTPException(
            status_code=422,
            detail="Audio file must have at least 2 channels (I and Q). "
                   f"File '{file.filename}' has only {data.shape[1]} channel(s)."
        )

    iq_data = data[:, :2]

    total_duration = len(iq_data) / fs
    if tmax <= 0 or tmax > total_duration:
        tmax = total_duration
    tmin = max(0.0, tmin)
    if tmin >= tmax:
        tmin = max(0.0, tmax - 1.0)

    start_idx = int(tmin * fs)
    end_idx   = int(tmax * fs)
    iq_data   = iq_data[start_idx:end_idx, :]

    iq_clean_wavelet = np.zeros_like(iq_data)
    for ch in range(iq_data.shape[1]):
        denoised = wavelet_denoise(iq_data[:, ch])
        iq_clean_wavelet[:, ch] = denoised[:len(iq_data)]

    iq_clean_bp = butter_bp(iq_clean_wavelet, fs, fmin, fmax)
    noise       = iq_data - iq_clean_bp

    fd_kasai        = kasai_fd(iq_clean_bp, fs)
    velocity        = velocity_from_fd(fd_kasai)
    raw_cardiac_out = velocity * CROSS_SECTIONAL_AREA * CO_SCALE
    cardiac_output  = lowpass(raw_cardiac_out, fs)

    timestamps = np.linspace(tmin, tmax, len(cardiac_output))
    df = pd.DataFrame({'Time_s': timestamps, 'Raw_PerDeCT_CO': cardiac_output})

    csv_filename = f"CO_data_{uuid.uuid4().hex[:8]}.csv"
    csv_path     = os.path.join("static", csv_filename)
    df.to_csv(csv_path, index=False)

    # Spectrogram (STFT) — Issue #9: stft already imported at top of file
    f_stft, t_stft, Zxx = stft(
        iq_clean_bp[:, 0], fs=fs, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP
    )
    t_stft  += tmin
    Zxx_db   = 20 * np.log10(np.abs(Zxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t_stft, f_stft, Zxx_db, cmap='turbo', shading='gouraud')
    ax.set_ylim([0, min(fmax + 1000, fs / 2)])
    ax.set_ylabel('Frequency (Hz)', color='#94a3b8')
    ax.set_xlabel('Time (s)', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')

    spec_filename = f"spec_{uuid.uuid4().hex[:8]}.png"
    spec_path     = os.path.join("static", spec_filename)
    plt.savefig(spec_path, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Issue #8: Use pre-loaded models from app.state
    hypo_model  = request.app.state.hypo_model
    drug_bundle = request.app.state.drug_bundle

    clinical_report, signal_quality = generate_clinical_report(
        cardiac_output, fs, tmin, tmax, hypo_model, drug_bundle
    )

    max_pts      = 4000
    viz_original = downsample(iq_data[:, 0], max_pts)
    viz_noise    = downsample(noise[:, 0], max_pts)
    viz_clean    = downsample(iq_clean_bp[:, 0], max_pts)
    viz_pulse    = downsample(cardiac_output, max_pts)
    t_axis       = np.linspace(tmin, tmax, len(viz_original)).tolist()

    return JSONResponse({
        "time":          t_axis,
        "original":      viz_original,
        "noise":         viz_noise,
        "clean":         viz_clean,
        "pulse":         viz_pulse,
        "fs":            fs,
        "duration":      tmax,
        "csv_url":       f"/static/{csv_filename}",
        "spec_url":      f"/static/{spec_filename}",
        "report":        clinical_report,
        "signal_quality": signal_quality,  # renamed from "confidence"/"accuracy"
        "compute_time":  time.time() - start_t,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
