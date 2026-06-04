"""
Core: Main Pipeline
Executes Tasks 1-4: Processing, Alignment, Calibration, and Motion Cancellation.
"""
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LinearRegression
from dsp_utils import butter_bp, inst_freq_hilbert, kasai_fd, velocity_from_fd, nlms_adaptive_filter

class System_Pipeline:
    def __init__(self, fs=48000, fmin=1500, fmax=10000):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        
    def process_audio(self, wav_path, accel_data=None):
        """Task 1 & Task 4: Extract velocity and handle motion cancellation."""
        print(f"Loading {wav_path}...")
        data, fs = sf.read(wav_path, always_2d=True, dtype="float32")
        iq_data = data[:, :2] # Assuming stereo I/Q
        
        # Task 4: Motion Cancellation (if accelerometer data provided)
        if accel_data is not None:
            print("Applying NLMS adaptive filter for motion cancellation...")
            # Apply independently to I and Q channels
            iq_data[:, 0] = nlms_adaptive_filter(iq_data[:, 0], accel_data)
            iq_data[:, 1] = nlms_adaptive_filter(iq_data[:, 1], accel_data)

        # Pre-processing
        iq_clean = butter_bp(iq_data, self.fs, self.fmin, self.fmax)
        
        # Task 1: Frequency & Velocity mapping
        ###Using Kasai for robust baseline, Hilbert for high-res
        fd_kasai = kasai_fd(iq_clean, self.fs)
        velocity = velocity_from_fd(fd_kasai)
        
        # Create timestamps
        timestamps = np.arange(len(velocity)) / self.fs
        
        # Calculate Flow (simplified Area assumption for prototype)
        cross_sectional_area = 4.0 # cm^2 (example)
        cardiac_output = velocity * cross_sectional_area * 6.0
        
        return pd.DataFrame({'Time_s': timestamps, 'Raw_System_CO': cardiac_output})

    def temporal_alignment(self, high_res_df, reference_df):
        """Task 2: Align 100ms outputs with 10s reference averages."""
        print("Aligning time-series resolutions...")
        # Convert seconds to Timedelta for pandas resampling
        high_res_df['Time'] = pd.to_timedelta(high_res_df['Time_s'], unit='s')
        high_res_df.set_index('Time', inplace=True)
        
        # Downsample System data to 10-second averages to match reference
        aligned_df = high_res_df.resample('10S').mean()
        
        # Merge with reference (assuming reference_df has 10S index)
        # combined_df = aligned_df.join(reference_df, how='inner')
        return aligned_df

    def calibrate_to_gold_standard(self, aligned_df, medistim_col='Medistim_CO'):
        """Task 3: Linear regression calibration against Medistim."""
        print("Calibrating against Gold Standard...")
        # Drop NaNs where reference data is missing
        clean_df = aligned_df.dropna(subset=['Raw_System_CO', medistim_col])
        
        X = clean_df[['Raw_System_CO']].values
        y = clean_df[medistim_col].values
        
        reg = LinearRegression().fit(X, y)
        m, b = reg.coef_[0], reg.intercept_
        print(f"Calibration Formula: True_CO = {m:.4f} * System_CO + {b:.4f}")
        
        aligned_df['Calibrated_CO'] = (aligned_df['Raw_System_CO'] * m) + b
        return aligned_df, reg

if __name__ == "__main__":
    print("Initialize System Pipeline...")
    # Example Execution Flow:
    pipeline = System_Pipeline()
    df = pipeline.process_audio(r'data\sample_1.wav')
    print("Saving the results to 'pipeline_output.csv'...")
    df.to_csv('pipeline_output.csv', index=False)
    print("Saved!")
    print(df.head())
