"""
Core: Digital Signal Processing Utilities
Includes core frequency extraction and Adaptive Noise Cancellation (ANC).
"""
import soundfile as sf
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt, filtfilt, savgol_filter, stft, istft

from constants import (
    FMIN, FMAX,
    SPEED_OF_SOUND, TRANSMIT_FREQ, ANGLE_DEG,
    CROSS_SECTIONAL_AREA, CO_SCALE,
    LOWPASS_CUTOFF,
)


def butter_bp(x, fs, f1, f2):
    """Bandpass filter to isolate relevant Doppler shifts."""
    ny = fs * 0.5
    f1, f2 = max(1.0, float(f1)), min(ny * 0.95, float(f2))
    sos = butter(4, [f1 / ny, f2 / ny], "bandpass", output='sos')
    return sosfiltfilt(sos, x, axis=0)


def inst_freq_hilbert(iq, fs):
    """Extract instantaneous frequency using Hilbert transform."""
    z = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    phase = np.unwrap(np.angle(z))
    dphi = np.diff(phase, prepend=phase[0])
    return (fs / (2 * np.pi)) * dphi


def kasai_fd(iq, fs, win=256, hop=128):
    """Extract frequency using Kasai lag-1 autocorrelation."""
    z = iq[:, 0].astype(np.complex128) + 1j * iq[:, 1].astype(np.complex128)
    n = z.size
    import numpy.lib.stride_tricks as st
    nwin = 1 + (n - win) // hop
    S = st.as_strided(z, shape=(nwin, win), strides=(z.strides[0] * hop, z.strides[0]))
    r1 = np.conj(S[:, :-1]) * S[:, 1:]
    fd = (fs / (2 * np.pi)) * np.angle(r1.sum(axis=1))
    out = np.repeat(fd, hop)
    return np.pad(out, (0, max(0, n - out.size)), mode='edge')[:n]


def velocity_from_fd(fd, c=SPEED_OF_SOUND, f_tx=TRANSMIT_FREQ, angle_deg=ANGLE_DEG):
    """Convert Doppler frequency shift to blood velocity."""
    sin_a = max(float(np.sin(np.deg2rad(angle_deg))), 1e-6)
    return (fd * c) / (2.0 * f_tx * sin_a)


def nlms_adaptive_filter(primary_signal, reference_noise, mu=0.01, filter_order=32):
    """
    Normalized Least Mean Squares (NLMS) filter for Task 4.
    Cancels probe motion artifacts using accelerometer data.
    """
    n = len(primary_signal)
    w = np.zeros(filter_order)
    clean_signal = np.zeros(n)

    # Pad reference noise for initial buffer
    ref_padded = np.pad(reference_noise, (filter_order - 1, 0), 'constant')

    for i in range(n):
        x_in = ref_padded[i: i + filter_order][::-1]

        # Estimate the noise component
        y_hat = np.dot(w, x_in)

        # Error is the primary signal minus the estimated noise (clean blood flow)
        error = primary_signal[i] - y_hat
        clean_signal[i] = error

        # Weight update with normalization to prevent instability
        norm = np.dot(x_in, x_in) + 1e-8
        w = w + (mu / norm) * error * x_in

    return clean_signal


def fft_noise_reduction(data, fs, nperseg=512, noise_factor=1.5):
    """
    FFT-based Spectral Subtraction (Spectral Gating) to remove noise
    without destroying original wave patterns.
    """
    clean_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = data[:, ch]
        f, t, Zxx = stft(x, fs=fs, nperseg=nperseg)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise profile using the median of the magnitude spectrum over time
        noise_profile = np.median(mag, axis=-1, keepdims=True)

        # Spectral subtraction: remove noise profile
        mag_clean = mag - (noise_profile * noise_factor)
        mag_clean = np.maximum(mag_clean, 0.0)

        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, x_rec = istft(Zxx_clean, fs=fs, nperseg=nperseg)

        if len(x_rec) > len(x):
            x_rec = x_rec[:len(x)]
        elif len(x_rec) < len(x):
            x_rec = np.pad(x_rec, (0, len(x) - len(x_rec)))

        clean_data[:, ch] = x_rec

    return clean_data


def wavelet_denoise(signal):
    """Wavelet denoising using Daubechies-4 at 3 levels with soft thresholding."""
    coeff = pywt.wavedec(signal, 'db4', level=3)
    coeff[1:] = [pywt.threshold(i, np.std(i) * 0.5) for i in coeff[1:]]
    return pywt.waverec(coeff, 'db4')


def lowpass(x, fs, cutoff=LOWPASS_CUTOFF):
    """3rd-order Butterworth low-pass filter using numerically stable SOS form."""
    sos = butter(3, cutoff / (fs / 2), btype='low', output='sos')
    return sosfiltfilt(sos, x)


def extract_co(wav_path, fmin=FMIN, fmax=FMAX):
    """
    Shared DSP pipeline: WAV file → cardiac output signal.

    Applies (in order):
      1. Wavelet denoising (db4, level 3)
      2. Butterworth bandpass filter
      3. Kasai lag-1 autocorrelation for Doppler frequency
      4. Doppler frequency → blood velocity
      5. Velocity → cardiac output (L/min)
      6. Low-pass envelope filter

    Returns
    -------
    co : np.ndarray  Cardiac output time-series (L/min)
    fs : int         Sample rate (Hz)
    """
    data, fs = sf.read(wav_path, always_2d=True, dtype="float32")
    iq = data[:, :2]

    iq_w = np.zeros_like(iq)
    for ch in range(iq.shape[1]):
        d = wavelet_denoise(iq[:, ch])
        iq_w[:, ch] = d[:len(iq)]

    iq_bp = butter_bp(iq_w, fs, fmin, fmax)
    fd    = kasai_fd(iq_bp, fs)
    vel   = velocity_from_fd(fd)
    co    = vel * CROSS_SECTIONAL_AREA * CO_SCALE
    co    = lowpass(co, fs)
    return co, fs
