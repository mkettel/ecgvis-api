# --- ECG Generation Constants ---
FS = 250
BASELINE_MV = 0.0
MIN_REFRACTORY_PERIOD_SEC = 0.200

# --- Beat Morphology Definitions ---
SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
    "q_amplitude": -0.2, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.25,
}
PVC_PARAMS = { # Also used as a base for VT beats
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "q_amplitude": -0.05, "r_amplitude": 1.2, # Typical PVC/VT morphology
    "s_amplitude": -0.2, "st_duration": 0.10, "t_duration": 0.22,
    "t_amplitude": -0.6, # Often discordant T-wave
}
PAC_PARAMS = {
    "p_duration": 0.08, "p_amplitude": 0.12, "pr_interval": 0.14,
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}
JUNCTIONAL_ESCAPE_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.05, "r_amplitude": 0.8, "s_amplitude": -0.2, "t_amplitude": 0.25,
}
VENTRICULAR_ESCAPE_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "st_duration": 0.10, "t_duration": 0.18,
    "q_amplitude": -0.15, "r_amplitude": 0.7, "s_amplitude": -0.5, "t_amplitude": -0.35,
}
AFIB_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy()
AFIB_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.001})

FLUTTER_WAVE_PARAMS = {
    "p_duration": 0.10, "p_amplitude": -0.2,
    "pr_interval": 0.0, "qrs_duration": 0.0, "st_duration": 0.0, "t_duration": 0.0,
    "q_amplitude":0.0, "r_amplitude":0.0, "s_amplitude":0.0, "t_amplitude":0.0,
}
FLUTTER_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy()
FLUTTER_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.14})

SVT_BEAT_PARAMS = SINUS_PARAMS.copy()
SVT_BEAT_PARAMS.update({
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.001,
})

VT_BEAT_PARAMS = PVC_PARAMS.copy() # Monomorphic VT beats often resemble PVCs

BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS,
    "afib_conducted": AFIB_CONDUCTED_QRS_PARAMS,
    "flutter_wave": FLUTTER_WAVE_PARAMS,
    "flutter_conducted_qrs": FLUTTER_CONDUCTED_QRS_PARAMS,
    "svt_beat": SVT_BEAT_PARAMS,
    "vt_beat": VT_BEAT_PARAMS,
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70

#----------------------------------------------------------------
# --- 3D Cardiac Vector Directions ------------------------------
import numpy as np # Make sure numpy is imported

# --- New Constants for 3D Cardiac Vector Directions ---
# These are VERY rough placeholders for mean electrical axes of P, QRS, T.
# These need to be normalized and are part of your R&D for Phase 2 & 3.
# Format: [X_component, Y_component, Z_component] based on defined coordinate system
# (X: R->L, Y: Sup->Inf, Z: Post->Ant)

# For Sinus Rhythm
SINUS_P_WAVE_DIRECTION = np.array([0.2, 0.8, 0.3])  # Example: Leftward, Inferior, Anterior
SINUS_QRS_COMPLEX_DIRECTION = np.array([0.4, 0.7, 0.2]) # Example: Leftward, Inferior, Anterior (mean QRS axis)
SINUS_T_WAVE_DIRECTION = np.array([0.3, 0.6, 0.15]) # Example: Generally follows QRS

# For PVCs (example, will vary by origin)
PVC_QRS_COMPLEX_DIRECTION = np.array([-0.6, 0.3, -0.5]) # Example: Rightward, Inferior, Posterior (e.g., LV origin)
PVC_T_WAVE_DIRECTION = np.array([0.5, -0.2, 0.4])      # Example: Discordant T-wave

# For PACs (P-wave direction will change based on focus)
PAC_P_WAVE_DIRECTION_EXAMPLE = np.array([0.1, 0.5, 0.6]) # Example: Low atrial PAC - more superior P vector

# Normalize these direction vectors (do this once, e.g., on module load or here)
def _normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

SINUS_P_WAVE_DIRECTION = _normalize(SINUS_P_WAVE_DIRECTION)
SINUS_QRS_COMPLEX_DIRECTION = _normalize(SINUS_QRS_COMPLEX_DIRECTION)
SINUS_T_WAVE_DIRECTION = _normalize(SINUS_T_WAVE_DIRECTION)
PVC_QRS_COMPLEX_DIRECTION = _normalize(PVC_QRS_COMPLEX_DIRECTION)
PVC_T_WAVE_DIRECTION = _normalize(PVC_T_WAVE_DIRECTION)
PAC_P_WAVE_DIRECTION_EXAMPLE = _normalize(PAC_P_WAVE_DIRECTION_EXAMPLE)

# You might want a more structured way to store these, e.g., in BEAT_MORPHOLOGIES
# For now, separate constants are fine for starting.
BEAT_3D_DIRECTIONS = {
    "sinus": {
        "P": SINUS_P_WAVE_DIRECTION,
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "pac": { # QRS/T often same as sinus if normally conducted
        "P": PAC_P_WAVE_DIRECTION_EXAMPLE, # This would ideally change based on PAC origin
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "pvc": {
        "P": np.array([0.0, 0.0, 0.0]), # No P-wave for PVC itself
        "QRS": PVC_QRS_COMPLEX_DIRECTION,
        "T": PVC_T_WAVE_DIRECTION,
    },
    "junctional_escape": {
        "P": np.array([0.0, 0.0, 0.0]),
        "QRS": SINUS_QRS_COMPLEX_DIRECTION, # Narrow QRS, so similar direction to sinus
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "ventricular_escape": {
        "P": np.array([0.0, 0.0, 0.0]),
        "QRS": PVC_QRS_COMPLEX_DIRECTION, # Wide QRS, similar to a PVC
        "T": PVC_T_WAVE_DIRECTION,
    },
    "svt_beat": { # Narrow complex, likely similar to sinus QRS/T directions
        "P": np.array([0.0, 0.0, 0.0]), # P-wave often hidden or retrograde
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "vt_beat": { # Wide complex, like PVCs
        "P": np.array([0.0, 0.0, 0.0]),
        "QRS": PVC_QRS_COMPLEX_DIRECTION, # Assuming monomorphic VT, direction depends on origin
        "T": PVC_T_WAVE_DIRECTION,
    },
    # AFib/AFlutter conducted beats would use their respective QRS/T, P directions would be zero.
    # The atrial activity itself for AFib/AFlutter needs a separate 3D model.
    "afib_conducted": {
        "P": np.array([0.0,0.0,0.0]),
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "flutter_conducted_qrs": {
        "P": np.array([0.0,0.0,0.0]),
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,
        "T": SINUS_T_WAVE_DIRECTION,
    },
}