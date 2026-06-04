import os
import sys
# Allow importing from the parent project directory when run from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import soundfile as sf
import numpy as np
from dsp_utils import butter_bp
from scipy.signal import butter, sosfiltfilt, filtfilt

data, fs = sf.read('data/2026_02_10_14-46-34p1.wav', always_2d=True, dtype="float32")
x = data[:, 0]
f1, f2 = 50.0, 8000.0

# original filter
y_ba = butter_bp(x, fs, f1, f2)
noise_ba = x - y_ba

# sos filter
ny = fs * 0.5
sos = butter(4, [f1/ny, f2/ny], "bandpass", output='sos')
y_sos = sosfiltfilt(sos, x, axis=0)
noise_sos = x - y_sos

print("Original max:", np.max(np.abs(x)))
print("BA filter max output:", np.max(np.abs(y_ba)))
print("BA filter max noise:", np.max(np.abs(noise_ba)))
print("SOS filter max output:", np.max(np.abs(y_sos)))
print("SOS filter max noise:", np.max(np.abs(noise_sos)))