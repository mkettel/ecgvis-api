# --- ECG Generation Constants ---
import numpy as np  # Import numpy at the top
FS = 250
BASELINE_MV = 0.0
MIN_REFRACTORY_PERIOD_SEC = 0.200

# --- QT Interval Correction Constants ---
# Normal QTc range: 350-450ms (men), 360-460ms (women)
NORMAL_QTC_MS = 400.0  # Target corrected QT interval in milliseconds
NORMAL_RR_INTERVAL_SEC = 1.0  # RR interval at 60 bpm (used as reference)

def calculate_qt_from_heart_rate(heart_rate_bpm: float, target_qtc_ms: float = NORMAL_QTC_MS) -> float:
    """
    Calculate QT interval using Bazett's formula: QTc = QT / √(RR)
    Rearranged to: QT = QTc × √(RR)
    
    Args:
        heart_rate_bpm: Heart rate in beats per minute
        target_qtc_ms: Target corrected QT interval in milliseconds
        
    Returns:
        QT interval in seconds
    """
    if heart_rate_bpm <= 0:
        return NORMAL_QTC_MS / 1000.0  # Default fallback
        
    rr_interval_sec = 60.0 / heart_rate_bpm
    qt_interval_sec = (target_qtc_ms / 1000.0) * (rr_interval_sec ** 0.5)
    
    # Safety bounds: QT should be 20-60% of RR interval
    min_qt = rr_interval_sec * 0.20
    max_qt = rr_interval_sec * 0.60
    
    return max(min_qt, min(max_qt, qt_interval_sec))

def get_rate_corrected_intervals(heart_rate_bpm: float, base_params: dict, target_qtc_ms: float = NORMAL_QTC_MS) -> dict:
    """
    Get rate-corrected interval durations for ECG generation.
    
    Args:
        heart_rate_bpm: Heart rate in beats per minute
        base_params: Base parameter dictionary (e.g., SINUS_PARAMS)
        
    Returns:
        Dictionary with rate-corrected timing intervals
    """
    corrected_params = base_params.copy()
    
    # Calculate corrected QT interval using specified target QTc
    total_qt_sec = calculate_qt_from_heart_rate(heart_rate_bpm, target_qtc_ms)
    
    # Distribute QT interval: QRS (25%), ST (30%), T-wave (45%)
    qrs_duration = corrected_params.get('qrs_duration', 0.10)  # Keep QRS relatively stable
    
    # Adjust QRS slightly for rate (wide complex at very high rates)
    if heart_rate_bpm > 150:
        qrs_duration *= 1.1  # Slight widening at very high rates
    elif heart_rate_bpm < 40:
        qrs_duration *= 0.95  # Slight narrowing at very low rates
        
    # Remaining time for ST + T-wave
    remaining_qt = total_qt_sec - qrs_duration
    
    # Distribute remaining time: ST gets ~30%, T-wave gets ~70%
    st_duration = remaining_qt * 0.30
    t_duration = remaining_qt * 0.70
    
    # Update the corrected parameters
    corrected_params.update({
        'qrs_duration': max(0.06, min(0.16, qrs_duration)),  # Bounds: 60-160ms
        'st_duration': max(0.05, min(0.20, st_duration)),    # Bounds: 50-200ms  
        't_duration': max(0.08, min(0.30, t_duration)),      # Bounds: 80-300ms
    })
    
    return corrected_params

# --- Beat Morphology Definitions ---
SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.25,
    "q_amplitude": -0.08, "r_amplitude": 1.4, "s_amplitude": -0.6,   # Physiological amplitudes for proper V1 rS pattern
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

# Torsades de Pointes parameters
TORSADES_BEAT_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.14, "st_duration": 0.08, "t_duration": 0.18,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.3, 
    "t_amplitude": -0.4, # Often discordant T-wave
}

# Bundle Branch Block Parameters
RBBB_SINUS_PARAMS = SINUS_PARAMS.copy()
RBBB_SINUS_PARAMS.update({
    "qrs_duration": 0.12,  # Wide QRS (>120ms)
    "r_amplitude": 0.8,    # Reduced R wave
    "s_amplitude": -0.8,   # Deep S wave in lateral leads
})

LBBB_SINUS_PARAMS = SINUS_PARAMS.copy() 
LBBB_SINUS_PARAMS.update({
    "qrs_duration": 0.13,  # Wide QRS (>120ms)
    "q_amplitude": 0.0,    # Absent Q waves
    "r_amplitude": 1.6,    # Tall, notched R wave
    "s_amplitude": -0.1,   # Small or absent S wave
})

# VFib Parameters  
VFIB_PARAMS = {
    "base_amplitude": 0.5,     # Base amplitude for VFib waves
    "amplitude_variation": 0.8, # How much amplitude varies
    "frequency_range": (4, 10), # Frequency range in Hz
    "chaos_factor": 0.9,       # How chaotic the rhythm is
}

BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    # Different PAC types - same morphology parameters, different vector directions
    "pac_high_ra": PAC_PARAMS,
    "pac_low_atrial": PAC_PARAMS,
    "pac_left_atrial": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS,
    "afib_conducted": AFIB_CONDUCTED_QRS_PARAMS,
    "flutter_wave": FLUTTER_WAVE_PARAMS,
    "flutter_conducted_qrs": FLUTTER_CONDUCTED_QRS_PARAMS,
    "svt_beat": SVT_BEAT_PARAMS,
    "vt_beat": VT_BEAT_PARAMS,
    "torsades_beat": TORSADES_BEAT_PARAMS,
    "rbbb_sinus": RBBB_SINUS_PARAMS,
    "lbbb_sinus": LBBB_SINUS_PARAMS,
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70

# --- Torsades de Pointes Constants ---
TORSADES_TRIGGER_QTC_MS = 500.0  # QTc threshold for Torsades risk
TORSADES_SENSITIVE_QTC_MS = 480.0  # QTc threshold for sensitive patients
TORSADES_MIN_DURATION_SEC = 3.0   # Minimum episode duration
TORSADES_MAX_DURATION_SEC = 30.0  # Maximum episode duration (usually self-terminates)
TORSADES_RATE_BPM = 220           # Typical Torsades rate (200-250 bpm)
TORSADES_AXIS_ROTATION_PERIOD_BEATS = 5  # How many beats per 180° rotation (faster twisting)

def rotate_vector_around_z_axis(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate a 3D vector around the Z-axis (anterior-posterior axis).
    This simulates the "twisting" characteristic of Torsades de Pointes.
    
    Args:
        vector: 3D vector [x, y, z]
        angle_degrees: Rotation angle in degrees
    
    Returns:
        Rotated 3D vector
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around Z-axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0], 
        [0,      0,     1]
    ])
    
    return rotation_matrix @ vector

def calculate_torsades_qrs_direction(beat_number: int, base_direction: np.ndarray) -> np.ndarray:
    """
    Calculate the QRS direction for a Torsades beat with characteristic axis rotation.
    
    Args:
        beat_number: Sequential beat number in the Torsades episode
        base_direction: Base QRS direction vector
        
    Returns:
        Rotated QRS direction vector for this beat
    """
    # Calculate rotation angle: 180° over TORSADES_AXIS_ROTATION_PERIOD_BEATS
    rotation_per_beat = 180.0 / TORSADES_AXIS_ROTATION_PERIOD_BEATS
    current_angle = (beat_number * rotation_per_beat) % 360.0
    
    # Add some variability to make it more realistic
    angle_variation = np.random.normal(0, 25)  # ±25° random variation (increased for more dramatic changes)
    total_angle = current_angle + angle_variation
    
    return rotate_vector_around_z_axis(base_direction, total_angle)

def calculate_qrs_vector_from_axis(target_axis_degrees: float) -> np.ndarray:
    """
    Calculate a 3D QRS vector that will produce the desired electrical axis.
    
    The electrical axis is primarily determined by the frontal plane (X-Y) components.
    For Lead I (pure X direction): projection = X_component
    For Lead aVF (pure Y direction): projection = Y_component
    Axis = arctan2(Y_component, X_component)
    
    Args:
        target_axis_degrees: Desired electrical axis in degrees (-180 to +180)
        
    Returns:
        Normalized 3D QRS vector [X, Y, Z] that produces the target axis
    """
    # Convert degrees to radians
    axis_radians = np.deg2rad(target_axis_degrees)
    
    # Calculate X and Y components (frontal plane)
    # Use unit magnitude in frontal plane, then add small Z component
    x_component = np.cos(axis_radians)  # Lead I amplitude
    y_component = np.sin(axis_radians)  # Lead aVF amplitude
    
    # Add realistic anterior component for proper precordial progression
    z_component = 0.6
    
    # Create and normalize the vector
    qrs_vector = np.array([x_component, y_component, z_component])
    norm = np.linalg.norm(qrs_vector)
    
    return qrs_vector / norm if norm > 0 else qrs_vector

#----------------------------------------------------------------
# --- 3D Cardiac Vector Directions ------------------------------

# --- New Constants for 3D Cardiac Vector Directions ---
# These are VERY rough placeholders for mean electrical axes of P, QRS, T.
# These need to be normalized and are part of your R&D for Phase 2 & 3.
# Format: [X_component, Y_component, Z_component] based on defined coordinate system
# (X: R->L, Y: Sup->Inf, Z: Post->Ant)

# For Sinus Rhythm - Dual-Phase P-wave Implementation
SINUS_P_WAVE_PHASE1_DIRECTION = np.array([-0.3, 0.5, 0.8])  # Right atrial depolarization: rightward, anterior
SINUS_P_WAVE_PHASE2_DIRECTION = np.array([0.6, 0.7, 0.2])   # Left atrial depolarization: leftward, posterior

# Multi-Phase QRS Implementation - Physiologically Accurate Sequential Depolarization
# Corrected vectors for proper V1 rS pattern and lateral Q waves
SINUS_QRS_SEPTAL_DIRECTION = np.array([1.0, 0.0, 0.2])      # Strong leftward septal: creates Q in lateral, r in V1
SINUS_QRS_FREE_WALL_DIRECTION = np.array([0.6, 0.8, 0.4])   # LV free wall: leftward/inferior, dominant R waves
SINUS_QRS_BASAL_DIRECTION = np.array([-0.6, 0.2, -0.4])     # Strong rightward/posterior: creates deep S waves in V1

# Legacy single QRS direction (for backward compatibility)
SINUS_QRS_COMPLEX_DIRECTION = np.array([0.7, 0.6, 0.35]) # Enhanced: Further increased anterior component for realistic precordial progression
SINUS_T_WAVE_DIRECTION = np.array([0.3, 0.6, 0.15]) # Example: Generally follows QRS

# For PVCs (example, will vary by origin)
# PVC_QRS_COMPLEX_DIRECTION = np.array([-0.8, 0.4, -0.3]) # Enhanced: More rightward for better V1 visibility (LV origin PVC)
# PVC_QRS_COMPLEX_DIRECTION = np.array([0.7, -0.5, -0.5]) # Left (X=0.7), Superior (Y=-0.5), Posterior (Z=-0.5)
PVC_QRS_COMPLEX_DIRECTION = np.array([0.6, 0.8, -0.3])
PVC_T_WAVE_DIRECTION = np.array([0.5, -0.2, 0.4])      # Example: Discordant T-wave

# For PACs - Different vectors based on ectopic focus location
# High Right Atrial PAC (near SVC junction)
HIGH_RA_PAC_PHASE1 = np.array([-0.5, 0.2, 0.9])  # More rightward and anterior than sinus
HIGH_RA_PAC_PHASE2 = np.array([0.3, 0.4, 0.8])   # LA component less dominant

# Low Atrial PAC (near coronary sinus/AV junction) 
# Spreads UPWARD from low atrium - creates inverted P-waves in inferior leads
LOW_ATRIAL_PAC_PHASE1 = np.array([0.1, -0.7, 0.2])  # Superior vector (negative Y = upward)
LOW_ATRIAL_PAC_PHASE2 = np.array([0.2, -0.8, 0.1])  # Both phases spread upward from ectopic focus

# Left Atrial PAC (from left atrial appendage or pulmonary veins)
LEFT_ATRIAL_PAC_PHASE1 = np.array([0.7, 0.5, 0.1])  # RA fires normally first
LEFT_ATRIAL_PAC_PHASE2 = np.array([0.9, 0.3, -0.2]) # Strong leftward, posterior LA focus

# Normalize these direction vectors (do this once, e.g., on module load or here)
def _normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

SINUS_P_WAVE_PHASE1_DIRECTION = _normalize(SINUS_P_WAVE_PHASE1_DIRECTION)
SINUS_P_WAVE_PHASE2_DIRECTION = _normalize(SINUS_P_WAVE_PHASE2_DIRECTION)

# Normalize multi-phase QRS vectors
SINUS_QRS_SEPTAL_DIRECTION = _normalize(SINUS_QRS_SEPTAL_DIRECTION)
SINUS_QRS_FREE_WALL_DIRECTION = _normalize(SINUS_QRS_FREE_WALL_DIRECTION)
SINUS_QRS_BASAL_DIRECTION = _normalize(SINUS_QRS_BASAL_DIRECTION)

# Legacy single QRS direction (normalized)
SINUS_QRS_COMPLEX_DIRECTION = _normalize(SINUS_QRS_COMPLEX_DIRECTION)
SINUS_T_WAVE_DIRECTION = _normalize(SINUS_T_WAVE_DIRECTION)
PVC_QRS_COMPLEX_DIRECTION = _normalize(PVC_QRS_COMPLEX_DIRECTION)
PVC_T_WAVE_DIRECTION = _normalize(PVC_T_WAVE_DIRECTION)

# Normalize PAC vectors
HIGH_RA_PAC_PHASE1 = _normalize(HIGH_RA_PAC_PHASE1)
HIGH_RA_PAC_PHASE2 = _normalize(HIGH_RA_PAC_PHASE2)
LOW_ATRIAL_PAC_PHASE1 = _normalize(LOW_ATRIAL_PAC_PHASE1)
LOW_ATRIAL_PAC_PHASE2 = _normalize(LOW_ATRIAL_PAC_PHASE2)
LEFT_ATRIAL_PAC_PHASE1 = _normalize(LEFT_ATRIAL_PAC_PHASE1)
LEFT_ATRIAL_PAC_PHASE2 = _normalize(LEFT_ATRIAL_PAC_PHASE2)

# You might want a more structured way to store these, e.g., in BEAT_MORPHOLOGIES
# For now, separate constants are fine for starting.
BEAT_3D_DIRECTIONS = {
    "sinus": {
        "P_PHASE1": SINUS_P_WAVE_PHASE1_DIRECTION,  # Right atrial depolarization
        "P_PHASE2": SINUS_P_WAVE_PHASE2_DIRECTION,  # Left atrial depolarization  
        "QRS_SEPTAL": SINUS_QRS_SEPTAL_DIRECTION,    # Septal depolarization (0-20ms)
        "QRS_FREE_WALL": SINUS_QRS_FREE_WALL_DIRECTION,  # Free wall depolarization (20-60ms)
        "QRS_BASAL": SINUS_QRS_BASAL_DIRECTION,      # Basal depolarization (60-100ms)
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,          # Legacy single QRS direction
        "T": SINUS_T_WAVE_DIRECTION,
    },
    # PACs - Different types based on ectopic focus location
    "pac_high_ra": {  # High Right Atrial PAC (near SVC)
        "P_PHASE1": HIGH_RA_PAC_PHASE1, 
        "P_PHASE2": HIGH_RA_PAC_PHASE2, 
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,  # Normally conducted
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "pac_low_atrial": {  # Low Atrial PAC (near AV junction)
        "P_PHASE1": LOW_ATRIAL_PAC_PHASE1,
        "P_PHASE2": LOW_ATRIAL_PAC_PHASE2, 
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,  # Normally conducted
        "T": SINUS_T_WAVE_DIRECTION,
    },
    "pac_left_atrial": {  # Left Atrial PAC (LA appendage/PV)
        "P_PHASE1": LEFT_ATRIAL_PAC_PHASE1,
        "P_PHASE2": LEFT_ATRIAL_PAC_PHASE2,
        "QRS": SINUS_QRS_COMPLEX_DIRECTION,  # Normally conducted  
        "T": SINUS_T_WAVE_DIRECTION,
    },
    # Legacy PAC (for backward compatibility)
    "pac": {  # Default to high RA PAC
        "P_PHASE1": HIGH_RA_PAC_PHASE1, 
        "P_PHASE2": HIGH_RA_PAC_PHASE2, 
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
    "torsades_beat": { # Polymorphic VT - direction will be dynamically rotated
        "P": np.array([0.0, 0.0, 0.0]),
        "QRS": np.array([-0.9, 0.4, -0.2]),  # More distinct base direction (rightward, posterior)
        "T": np.array([0.6, -0.8, 0.1]), # Strongly discordant T-wave direction
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