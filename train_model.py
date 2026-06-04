"""
train_model.py
--------------
Trains two models:

  1. hypovolemia_model.pkl  — IsolationForest (unsupervised anomaly detector)
     Trained on all CO data from pipeline_output.csv.

  2. drug_condition_model.pkl — RandomForestClassifier (supervised)
     Trained on rolling-window features extracted from every WAV file in data/,
     with each window labelled by its file's known drug/condition ground truth
     (see drug_labels.py).
     Also runs Leave-One-File-Out (LOFO) cross-validation so out-of-sample
     performance is estimated honestly — not just in-sample accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
# Issue #6: Use shared extract_co from dsp_utils (no more duplication)
from dsp_utils import extract_co
from drug_labels import DRUG_LABELS, DRUG_CLASSES
from constants import (
    CO_DECIMATE_MAX, ROLLING_WINDOW_MAX,
)

WAV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Shared feature extraction ─────────────────────────────────────────────────

def rolling_features(co_signal, label_str):
    """
    Extract rolling-window features identical to app.py (Issue #7 fix:
    window cap is now ROLLING_WINDOW_MAX = 50 in both train and serve).

    Returns a DataFrame with columns: mean, std, min, max, ptp, label
    """
    dec    = max(1, len(co_signal) // CO_DECIMATE_MAX)
    co_sub = co_signal[::dec]

    df = pd.DataFrame({'co': co_sub})
    df['co'] = df['co'].ffill().fillna(0)

    # Issue #7: was min(100, ...) in training vs min(50, ...) in app.py.
    # Now both use ROLLING_WINDOW_MAX = 50 from constants.py.
    wp      = max(10, min(ROLLING_WINDOW_MAX, len(co_sub) // 10))
    rolling = df['co'].rolling(window=wp, min_periods=max(1, wp // 2))

    feats = pd.DataFrame({
        'mean' : rolling.mean(),
        'std'  : rolling.std().fillna(0),
        'min'  : rolling.min(),
        'max'  : rolling.max(),
        'ptp'  : rolling.max() - rolling.min(),
    }).dropna()

    feats['label'] = label_str
    return feats


# ── Model 1: IsolationForest ──────────────────────────────────────────────────

def train_isolation_forest():
    print("\n[1/2] Training IsolationForest (hypovolemia anomaly detector)...")
    try:
        df = pd.read_csv('pipeline_output.csv')
    except Exception as e:
        print(f"  [WARN] pipeline_output.csv not found ({e}). "
              "Skipping IsolationForest retrain — existing model kept.")
        return

    if 'Raw_System_CO' not in df.columns:
        print("  [WARN] 'Raw_System_CO' column missing. Skipping.")
        return

    co_data = df['Raw_System_CO'].values
    dec     = max(1, len(co_data) // 100_000)
    co_sub  = co_data[::dec]
    wp      = max(10, 500 // dec)

    co_df       = pd.DataFrame({'co': co_sub})
    co_df['co'] = co_df['co'].ffill().fillna(0)
    roll        = co_df['co'].rolling(window=wp, min_periods=max(1, wp // 10))
    feats       = pd.DataFrame({
        'mean': roll.mean(),
        'std' : roll.std().fillna(0),
        'min' : roll.min(),
        'max' : roll.max(),
        'ptp' : roll.max() - roll.min(),
    }).dropna()

    print(f"  Feature matrix: {feats.shape}")
    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    model.fit(feats.values)
    joblib.dump(model, "hypovolemia_model.pkl")
    print("  Saved -> hypovolemia_model.pkl")


# ── Model 2: Supervised Drug-Condition Classifier ─────────────────────────────

def train_drug_classifier():
    print("\n[2/2] Training RandomForest drug-condition classifier...")

    wav_files = sorted(
        f for f in os.listdir(WAV_DIR) if f.lower().endswith(".wav")
    )
    if not wav_files:
        print(f"  [ERROR] No WAV files in {WAV_DIR}")
        return

    # ── Collect per-file feature frames ───────────────────────────────────────
    file_frames: dict[str, pd.DataFrame] = {}
    missing = []

    for fname in wav_files:
        if fname not in DRUG_LABELS:
            missing.append(fname)
            continue

        label = DRUG_LABELS[fname]["label"]
        path  = os.path.join(WAV_DIR, fname)

        print(f"  Processing [{label:22s}] {fname}")
        try:
            # Issue #6: use shared extract_co from dsp_utils
            co, _   = extract_co(path)
            frame   = rolling_features(co, label)
            file_frames[fname] = frame
            print(f"    -> {len(frame):,} feature windows extracted")
        except Exception as e:
            print(f"    [ERROR] {e}")

    if missing:
        print(f"\n  [WARN] No label entry for: {missing}")

    if not file_frames:
        print("  [ERROR] No data to train on.")
        return

    combined = pd.concat(list(file_frames.values()), ignore_index=True)
    print(f"\n  Total feature windows : {len(combined):,}")
    print("  Class distribution:")
    for lbl, cnt in combined['label'].value_counts().items():
        print(f"    {lbl:25s}: {cnt:6,}  ({cnt/len(combined)*100:.1f}%)")

    X_all = combined[['mean', 'std', 'min', 'max', 'ptp']].values
    y_all = combined['label'].values

    le = LabelEncoder()
    le.fit(DRUG_CLASSES)   # Fix encoding order to DRUG_CLASSES
    y_enc_all = le.transform(y_all)

    # ── Issue #5: Leave-One-File-Out Cross-Validation ─────────────────────────
    print("\n  Running Leave-One-File-Out cross-validation...")
    lofo_preds  = []
    lofo_truths = []

    file_keys = list(file_frames.keys())
    for held_out in file_keys:
        train_frames = [frame for k, frame in file_frames.items() if k != held_out]
        if not train_frames:
            continue
        train_df = pd.concat(train_frames, ignore_index=True)
        test_df  = file_frames[held_out]

        X_tr = train_df[['mean', 'std', 'min', 'max', 'ptp']].values
        y_tr = le.transform(train_df['label'].values)
        X_te = test_df[['mean', 'std', 'min', 'max', 'ptp']].values
        y_te = le.transform(test_df['label'].values)

        clf_cv = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf_cv.fit(X_tr, y_tr)

        # Use per-window majority vote (same as inference) for the held-out file
        probs_te   = clf_cv.predict_proba(X_te).mean(axis=0)
        pred_label = le.inverse_transform([int(np.argmax(probs_te))])[0]
        true_label = DRUG_LABELS[held_out]["label"]

        verdict = "PASS" if pred_label == true_label else "FAIL"
        print(f"    [{verdict}] {held_out[:30]:30s}  true={true_label:22s}  pred={pred_label}")

        lofo_preds.append(pred_label)
        lofo_truths.append(true_label)

    if lofo_preds:
        lofo_acc = accuracy_score(lofo_truths, lofo_preds)
        print(f"\n  LOFO accuracy (file-level): {lofo_acc * 100:.0f}%  "
              f"({sum(p == t for p, t in zip(lofo_preds, lofo_truths))}/{len(lofo_preds)} files correct)")
        print("\n  Per-class breakdown:")
        print(classification_report(lofo_truths, lofo_preds, zero_division=0))

    # ── Train final model on ALL data ──────────────────────────────────────────
    print("  Training final RandomForestClassifier on all data (200 trees)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_all, y_enc_all)

    bundle = {"model": clf, "label_encoder": le, "classes": DRUG_CLASSES}
    joblib.dump(bundle, "drug_condition_model.pkl")
    print("  Saved -> drug_condition_model.pkl")

    # In-sample sanity check (not a substitute for the LOFO above)
    acc = clf.score(X_all, y_enc_all) * 100
    print(f"\n  In-sample accuracy (sanity check only): {acc:.1f}%")
    print("  Refer to LOFO accuracy above for an honest out-of-sample estimate.")

    # Feature importances
    fi = pd.Series(clf.feature_importances_,
                   index=['mean', 'std', 'min', 'max', 'ptp'])
    print("\n  Feature importances:")
    for feat, imp in fi.sort_values(ascending=False).items():
        bar = "#" * int(imp * 40)
        print(f"    {feat:6s}: {imp:.3f}  {bar}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  System Model Training")
    print("=" * 60)
    train_isolation_forest()
    train_drug_classifier()
    print("\nAll models trained successfully.")
