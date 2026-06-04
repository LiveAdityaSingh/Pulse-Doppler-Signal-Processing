"""
constants.py
------------
Single source of truth for all shared physics and DSP constants used
across app.py, pipeline.py, train_model.py, verify_model.py, and
generate_calib.py.  Change a value here and it propagates everywhere.
"""

# ── Bandpass filter bounds (Hz) ───────────────────────────────────────────────
FMIN: float = 1500.0   # Lower Doppler frequency bound
FMAX: float = 10000.0  # Upper Doppler frequency bound

# ── Velocity / Doppler conversion ─────────────────────────────────────────────
SPEED_OF_SOUND: float = 1540.0        # Speed of sound in blood (m/s)
TRANSMIT_FREQ: float  = 8_000_000.0   # Probe transmit frequency (Hz) — 8 MHz
ANGLE_DEG: float      = 30.0          # Beam-to-vessel angle (degrees)

# ── Cardiac output scaling ─────────────────────────────────────────────────────
CROSS_SECTIONAL_AREA: float = 4.0  # Assumed aortic cross-section (cm²)
CO_SCALE: float             = 6.0  # Converts cm³/s → L/min (× area × 6)

# ── Lowpass filter ────────────────────────────────────────────────────────────
LOWPASS_CUTOFF: float = 15.0  # Cardiac output envelope cutoff (Hz)

# ── ML feature extraction ─────────────────────────────────────────────────────
# Maximum points to keep after decimation before rolling-window feature extraction.
# MUST be the same value at training time and inference time to avoid
# train-serve feature skew (previously 10 000 in both).
CO_DECIMATE_MAX: int = 10_000

# Maximum rolling-window size cap for feature extraction.
# MUST match between train_model.py and app.py.
ROLLING_WINDOW_MAX: int = 50

# ── Spectrogram ───────────────────────────────────────────────────────────────
STFT_NPERSEG: int  = 1024
STFT_NOVERLAP: int = 512

# ── Static file retention ─────────────────────────────────────────────────────
# How many recent CSV / spectrogram files to keep in static/ after each request.
STATIC_KEEP_CSV: int  = 5
STATIC_KEEP_SPEC: int = 10
