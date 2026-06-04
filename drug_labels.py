"""
drug_labels.py
--------------
Ground-truth metadata registry for all 7 pig Doppler WAV recordings.

Labels
------
  Baseline            : Normal cardiac state, no drug intervention
  Esmolol_HRSlow      : Esmolol (beta-blocker) started, HR beginning to fall
  Esmolol_Arrhythmia  : Esmolol active, HR arrhythmia present
  Dobutamine_HighCO   : Dobutamine (inotrope) active, CO elevated

Each entry maps the WAV filename (without directory) to its metadata.
"""

# Valid class labels in training order
DRUG_CLASSES = [
    "Baseline",
    "Esmolol_HRSlow",
    "Esmolol_Arrhythmia",
    "Dobutamine_HighCO",
]

# Ground-truth registry
# Keys must exactly match the filenames in the data/ folder
DRUG_LABELS = {
    "2026_02_10_14-46-34p1.wav": {
        "start_time"   : "14:46:34",
        "duration_s"   : 60.01,
        "label"        : "Baseline",
        "reference_co" : 4.8,          # L/min (Medistim gold standard)
        "reference_hr" : None,
        "peak_velocity": 150,           # cm/s measured by Medistim
        "notes"        : "Cardiac Output 4.8 L/min. ~150 cm/s peak velocity "
                         "measured by Medistim. Noise (crosstalk from other probes).",
        "has_noise"    : True,
    },
    "2026_02_10_14-55-19p1.wav": {
        "start_time"   : "14:55:19",
        "duration_s"   : 60.01,
        "label"        : "Baseline",
        "reference_co" : None,
        "reference_hr" : 114,           # BPM
        "peak_velocity": None,
        "notes"        : "Heart rate 114 BPM. "
                         "Noise (crosstalk from other probes).",
        "has_noise"    : True,
    },
    "2026_02_10_15-25-08p1.wav": {
        "start_time"   : "15:25:08",
        "duration_s"   : 60.01,
        "label"        : "Baseline",
        "reference_co" : None,
        "reference_hr" : 110,
        "peak_velocity": None,
        "notes"        : "Heart rate 110 BPM. "
                         "Noise (crosstalk from other probes).",
        "has_noise"    : True,
    },
    "2026_02_10_15-42-07p1.wav": {
        "start_time"   : "15:42:07",
        "duration_s"   : 57.194,
        "label"        : "Esmolol_HRSlow",
        "reference_co" : None,
        "reference_hr" : None,
        "peak_velocity": None,
        "notes"        : "Start of Esmolol infusion. HR beginning to slow.",
        "has_noise"    : False,
    },
    "2026_02_10_15-44-29p1.wav": {
        "start_time"   : "15:44:29",
        "duration_s"   : 154.431,
        "label"        : "Esmolol_Arrhythmia",
        "reference_co" : None,
        "reference_hr" : None,
        "peak_velocity": None,
        "notes"        : "Esmolol slowing HR. HR has arrhythmia. "
                         "Noise (crosstalk from other probes).",
        "has_noise"    : True,
    },
    "2026_02_10_15-58-15p1.wav": {
        "start_time"   : "15:58:15",
        "duration_s"   : 103.445,
        "label"        : "Dobutamine_HighCO",
        "reference_co" : None,
        "reference_hr" : None,
        "peak_velocity": None,
        "notes"        : "Dobutamine raising cardiac output. "
                         "Noise (crosstalk from other probes).",
        "has_noise"    : True,
    },
    "2026_02_10_16-11-11p1.wav": {
        "start_time"   : "16:11:11",
        "duration_s"   : 67.805,
        "label"        : "Esmolol_Arrhythmia",
        "reference_co" : None,
        "reference_hr" : None,
        "peak_velocity": None,
        "notes"        : "Esmolol slowing HR. Arrhythmia appears towards "
                         "the end of the clip.",
        "has_noise"    : False,
    },
}

# Human-readable descriptions for report generation
LABEL_DESCRIPTIONS = {
    "Baseline": (
        "No pharmacological intervention detected. "
        "Cardiac rhythm and output appear at natural baseline levels."
    ),
    "Esmolol_HRSlow": (
        "Pattern consistent with Esmolol (beta-1 adrenergic blocker) effect. "
        "Characteristic progressive HR suppression detected without severe arrhythmia."
    ),
    "Esmolol_Arrhythmia": (
        "Pattern consistent with Esmolol-induced bradyarrhythmia. "
        "Irregular beat-to-beat intervals and HR suppression are hallmarks of "
        "beta-blocker overdose or high dosing."
    ),
    "Dobutamine_HighCO": (
        "Pattern consistent with Dobutamine (inotrope/chronotrope) effect. "
        "Elevated cardiac output and hyperdynamic waveform morphology detected, "
        "consistent with catecholamine stimulation."
    ),
}

# Icon for each label used in clinical reports
LABEL_ICONS = {
    "Baseline"           : "🟢",
    "Esmolol_HRSlow"     : "🔵",
    "Esmolol_Arrhythmia" : "🟡",
    "Dobutamine_HighCO"  : "🔴",
}
