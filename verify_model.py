"""
verify_model.py
---------------
Verifies both trained models against all pig WAV files in data/.

For each file:
  - Prints CO statistics (mean, median, std, min, max, range, ASCII trend)
  - Runs IsolationForest anomaly detector
  - Runs Drug-condition RandomForest classifier
  - Compares prediction vs. ground-truth label (Pass/Fail)

Prints a final summary table.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
# Issue #6: Use shared extract_co — no local duplicate
from dsp_utils import extract_co
from drug_labels import DRUG_LABELS, DRUG_CLASSES, LABEL_DESCRIPTIONS, LABEL_ICONS
from constants import FMIN, FMAX, CO_DECIMATE_MAX, ROLLING_WINDOW_MAX

WAV_DIR    = os.path.join(os.path.dirname(__file__), "data")
IF_MODEL   = os.path.join(os.path.dirname(__file__), "hypovolemia_model.pkl")
DRUG_MODEL = os.path.join(os.path.dirname(__file__), "drug_condition_model.pkl")
SEP        = "-" * 70


# extract_co is imported from dsp_utils (shared with app.py and train_model.py)


def extract_features(co_signal, fs=48000):
    """Rolling-window features using the same constants as app.py and train_model.py."""
    dec    = max(1, len(co_signal) // CO_DECIMATE_MAX)
    co_sub = co_signal[::dec]
    time_s = np.linspace(0, len(co_signal) / fs, len(co_sub))
    df = pd.DataFrame({'co': co_sub, 'time': time_s})
    df['co'] = df['co'].ffill().fillna(0)
    wp      = max(10, min(ROLLING_WINDOW_MAX, len(co_sub) // 10))
    rolling = df['co'].rolling(window=wp, min_periods=max(1, wp // 2))
    feats   = pd.DataFrame({
        'mean': rolling.mean(),
        'std' : rolling.std().fillna(0),
        'min' : rolling.min(),
        'max' : rolling.max(),
        'ptp' : rolling.max() - rolling.min(),
        'time': df['time'],
    }).dropna()
    return feats


def hr_from_co(co, fs):
    peaks, _ = find_peaks(co, distance=int(fs * 0.33),
                          prominence=np.max(co) * 0.15)
    if len(peaks) < 2:
        return None, None
    rr = np.diff(peaks / fs)
    return 60.0 / np.mean(rr), np.std(rr) * 1000


def infer_if_condition(anomalies, features):
    if len(anomalies) == 0:
        return "No anomalies"
    row      = anomalies.loc[anomalies['ptp'].idxmax()]
    med_ptp  = features['ptp'].median()
    med_mean = features['mean'].median()
    t        = int(row['time'])
    if row['ptp'] > med_ptp * 1.5:
        c = "Arrhythmia (chaotic rhythm)"
    elif row['mean'] < med_mean * 0.6:
        c = "Hypovolemia / Hemorrhage"
    elif row['mean'] > med_mean * 1.5:
        c = "Hyperdynamic state (stress/sepsis)"
    elif row['std'] < features['std'].median() * 0.5:
        c = "Heart Failure / Myocardial Depression"
    else:
        c = "Non-specific cardiovascular distress"
    return f"{c}  [worst anomaly ~{t}s]"


def ascii_trend(co, buckets=50):
    chunk = max(1, len(co) // buckets)
    means = [np.mean(co[i*chunk:(i+1)*chunk]) for i in range(buckets)]
    lo, hi = min(means), max(means)
    chars  = " _.-~^*#@"
    if hi > lo:
        bars = "".join(chars[int((v-lo)/(hi-lo)*(len(chars)-1))] for v in means)
    else:
        bars = "-" * buckets
    return bars, lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  PerDeCT ML Verification -- Pig Health Data")
    print(SEP)

    # --- Load models ---------------------------------------------------------
    if not os.path.exists(IF_MODEL):
        print(f"[ERROR] IsolationForest model not found: {IF_MODEL}")
        sys.exit(1)
    if not os.path.exists(DRUG_MODEL):
        print(f"[ERROR] Drug-condition model not found: {DRUG_MODEL}")
        print("  Run:  .venv\\Scripts\\python.exe train_model.py")
        sys.exit(1)

    if_model    = joblib.load(IF_MODEL)
    drug_bundle = joblib.load(DRUG_MODEL)
    drug_clf    = drug_bundle["model"]
    le          = drug_bundle["label_encoder"]
    print(f"[OK] IsolationForest loaded       : {os.path.basename(IF_MODEL)}")
    print(f"[OK] Drug classifier loaded       : {os.path.basename(DRUG_MODEL)}")
    print(f"     Classes                       : {list(le.classes_)}")
    print()

    # --- Ground-truth table --------------------------------------------------
    print(SEP)
    print("GROUND TRUTH REGISTRY")
    print(SEP)
    gt_rows = []
    for fname, meta in DRUG_LABELS.items():
        gt_rows.append({
            "File"        : fname,
            "Start"       : meta["start_time"],
            "Dur(s)"      : meta["duration_s"],
            "True Label"  : meta["label"],
            "Ref CO"      : meta["reference_co"] if meta["reference_co"] else "N/A",
            "Ref HR"      : meta["reference_hr"] if meta["reference_hr"] else "N/A",
            "Noise"       : "Yes" if meta["has_noise"] else "No",
        })
    df_gt = pd.DataFrame(gt_rows)
    print(df_gt.to_string(index=False))
    print()

    # --- Per-file analysis ---------------------------------------------------
    wav_files = sorted(
        os.path.join(WAV_DIR, f)
        for f in os.listdir(WAV_DIR) if f.lower().endswith(".wav")
    )
    if not wav_files:
        print(f"[ERROR] No WAV files found in: {WAV_DIR}")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV file(s)\n")
    summary_rows = []

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        meta  = DRUG_LABELS.get(fname, {})
        true_label = meta.get("label", "Unknown")

        print(SEP)
        print(f"File        : {fname}")
        print(f"True Label  : {true_label}")
        if meta.get("notes"):
            print(f"Notes       : {meta['notes']}")
        print(SEP)

        try:
            co, fs = extract_co(wav_path, fmin=FMIN, fmax=FMAX)

            # CO statistics
            duration = len(co) / fs
            co_mean  = float(np.mean(co))
            co_std   = float(np.std(co))
            co_min   = float(np.min(co))
            co_max   = float(np.max(co))
            co_med   = float(np.median(co))

            print(f"  Duration          : {duration:.1f} s  ({len(co):,} samples @ {fs} Hz)")
            print()
            print("  Cardiac Output (CO) statistics:")
            print(f"    Mean   : {co_mean:+.4f} L/min")
            print(f"    Median : {co_med:+.4f} L/min")
            print(f"    Std    : {co_std:.4f} L/min")
            print(f"    Min    : {co_min:+.4f} L/min")
            print(f"    Max    : {co_max:+.4f} L/min")
            print(f"    Range  : {co_max - co_min:.4f} L/min")

            if meta.get("reference_co"):
                print(f"    Ref CO : {meta['reference_co']:.1f} L/min  (Medistim gold standard)")
            if meta.get("reference_hr"):
                print(f"    Ref HR : {meta['reference_hr']} BPM")

            bars, lo, hi = ascii_trend(co)
            print()
            print(f"  CO Trend (ASCII)  : |{bars}|")
            print(f"                      {lo:+.2f} -> {hi:+.2f} L/min")
            print()

            # Feature extraction
            features = extract_features(co, fs=fs)
            if len(features) == 0:
                print("  [WARN] Feature matrix empty -- skipping ML.")
                continue

            feat_vals = features[['mean','std','min','max','ptp']].values

            # -- IsolationForest result
            if_preds       = if_model.predict(feat_vals)
            features['if'] = if_preds
            anomalies      = features[features['if'] == -1]
            anomaly_ratio  = len(anomalies) / len(features)
            if_status      = "[!!] ABNORMAL" if anomaly_ratio > 0.15 else "[OK] NORMAL"
            if_cond        = (infer_if_condition(anomalies, features)
                              if anomaly_ratio > 0.15
                              else "Stable CO -- no severe hypovolemic signature")

            print(f"  IsolationForest:")
            print(f"    Anomaly ratio   : {anomaly_ratio*100:.1f}%  ({if_status})")
            print(f"    Inferred cond.  : {if_cond}")

            # -- Drug classifier result
            drug_probs  = drug_clf.predict_proba(feat_vals)
            avg_probs   = drug_probs.mean(axis=0)
            pred_idx    = int(np.argmax(avg_probs))
            pred_label  = le.inverse_transform([pred_idx])[0]
            confidence  = avg_probs[pred_idx] * 100
            correct     = (pred_label == true_label)
            verdict     = "PASS" if correct else "FAIL"

            print()
            print(f"  Drug Classifier:")
            for i, cls in enumerate(le.classes_):
                bar  = "#" * int(avg_probs[i] * 40)
                mark = " <-- predicted" if i == pred_idx else ""
                print(f"    {cls:25s}: {avg_probs[i]*100:5.1f}%  {bar}{mark}")
            print(f"    Predicted Label : {pred_label}  ({confidence:.1f}% confidence)")
            print(f"    True Label      : {true_label}")
            print(f"    Verdict         : [{verdict}]")
            print(f"    Description     : {LABEL_DESCRIPTIONS[pred_label]}")

            # HR
            hr_bpm, hrv_ms = hr_from_co(co, fs)
            print()
            if hr_bpm:
                print(f"  Heart Rate        : {hr_bpm:.1f} BPM   HRV: {hrv_ms:.1f} ms")
            else:
                print(f"  Heart Rate        : insufficient peaks")

            summary_rows.append({
                "File"         : fname,
                "True Label"   : true_label,
                "Pred Label"   : pred_label,
                "Confidence"   : f"{confidence:.1f}%",
                "Anomaly%"     : f"{anomaly_ratio*100:.1f}%",
                "CO Mean"      : f"{co_mean:+.2f}",
                "HR(BPM)"      : f"{hr_bpm:.1f}" if hr_bpm else "N/A",
                "Verdict"      : verdict,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
        print()

    # --- Summary table -------------------------------------------------------
    print(SEP)
    print("SUMMARY TABLE")
    print(SEP)
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        print(df_sum.to_string(index=False))
        passed = sum(1 for r in summary_rows if r["Verdict"] == "PASS")
        total  = len(summary_rows)
        print(f"\n  Overall accuracy : {passed}/{total} ({passed/total*100:.0f}%)")

    print()
    print("Verification complete.")


if __name__ == "__main__":
    main()
