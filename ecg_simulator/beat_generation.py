# // backend/ecg-simulator/beat_generation.py
import numpy as np
from typing import Dict, Tuple, Optional
from .constants import FS, BASELINE_MV, BEAT_3D_DIRECTIONS
from .waveform_primitives import gaussian_wave, fourier_p_wave, fourier_qrs_component

def generate_biphasic_p_wave_vectors(
    t_axis: np.ndarray,
    p_center: float,
    p_amplitude: float,
    p_duration: float,
    phase1_direction: np.ndarray,
    phase2_direction: np.ndarray
) -> np.ndarray:
    """
    Generate dual-phase P-wave vectors representing right and left atrial depolarization.
    Creates smooth, blended RA/LA components that merge physiologically without visible separation.
    
    Args:
        t_axis: Time axis for the beat
        p_center: Center time of P-wave
        p_amplitude: Peak amplitude of P-wave
        p_duration: Total P-wave duration
        phase1_direction: 3D direction vector for right atrial depolarization
        phase2_direction: 3D direction vector for left atrial depolarization
    
    Returns:
        cardiac_vectors: Array of shape (len(t_axis), 3) representing P-wave vectors
    """
    cardiac_vectors = np.zeros((len(t_axis), 3))
    
    # Create overlapping phases with minimal temporal separation for smooth blending
    # Phase 1: Right atrial depolarization (starts early, overlaps with LA)
    phase1_duration = p_duration * 0.7  # Longer duration for smooth overlap
    phase1_center = p_center - p_duration * 0.08  # Minimal offset from center
    phase1_amplitude = p_amplitude * 0.6  # Reduced to prevent over-amplitude when combined
    
    phase1_scalar = fourier_p_wave(t_axis, phase1_center, phase1_amplitude, phase1_duration)
    
    # Phase 2: Left atrial depolarization (starts slightly later, larger contribution)
    phase2_duration = p_duration * 0.8  # Longer duration for smooth overlap  
    phase2_center = p_center + p_duration * 0.05  # Minimal offset for subtle timing difference
    phase2_amplitude = p_amplitude * 0.7  # LA is typically dominant but not overwhelming
    
    phase2_scalar = fourier_p_wave(t_axis, phase2_center, phase2_amplitude, phase2_duration)
    
    # Combine the phases with vector addition
    cardiac_vectors += np.outer(phase1_scalar, phase1_direction)
    cardiac_vectors += np.outer(phase2_scalar, phase2_direction)
    
    # Calculate and log cumulative atrial vector axis for debugging
    # Find the peak of combined P-wave activity
    total_magnitude = np.linalg.norm(cardiac_vectors, axis=1)
    peak_idx = np.argmax(total_magnitude)
    if peak_idx < len(cardiac_vectors):
        peak_vector = cardiac_vectors[peak_idx]
        
        # Calculate axis in frontal plane (x-y plane) - degrees from positive x-axis
        frontal_axis_radians = np.arctan2(peak_vector[1], peak_vector[0])
        frontal_axis_degrees = np.degrees(frontal_axis_radians)
        
        # Normalize to 0-360 degrees
        if frontal_axis_degrees < 0:
            frontal_axis_degrees += 360
            
        # Also calculate magnitude for reference
        peak_magnitude = np.linalg.norm(peak_vector)
        
        # P-wave analysis for debugging (removed for production)
    
    # Apply smoothing to the combined result to eliminate any residual jagedness
    # Smooth each vector component separately
    for i in range(3):  # x, y, z components
        vector_component = cardiac_vectors[:, i]
        # Find non-zero region to apply smoothing
        non_zero_mask = np.abs(vector_component) > 0.01 * np.max(np.abs(vector_component))
        if np.any(non_zero_mask):
            # Apply gentle 5-point smoothing to eliminate sharp transitions
            smoothed_component = np.copy(vector_component)
            for j in range(2, len(vector_component) - 2):
                if non_zero_mask[j]:
                    smoothed_component[j] = (0.1 * vector_component[j-2] + 
                                           0.2 * vector_component[j-1] + 
                                           0.4 * vector_component[j] + 
                                           0.2 * vector_component[j+1] + 
                                           0.1 * vector_component[j+2])
            cardiac_vectors[:, i] = smoothed_component
    
    return cardiac_vectors

def generate_physiological_qrs_vectors(
    t_axis: np.ndarray,
    qrs_onset_time: float,
    qrs_duration: float,
    qrs_amplitudes: Dict[str, float],
    target_axis_degrees: float = None
) -> np.ndarray:
    """
    Generate physiologically accurate QRS vectors based on real cardiac electrophysiology.
    
    This creates QRS vectors that are properly placed in the time axis at the correct QRS location.
    
    Args:
        t_axis: Time axis for the beat
        qrs_onset_time: Time when QRS complex begins
        qrs_duration: Total QRS complex duration
        qrs_amplitudes: Dictionary with Q, R, S amplitudes (used for scaling)
        target_axis_degrees: Optional electrical axis override in degrees
    
    Returns:
        cardiac_vectors: Array of shape (len(t_axis), 3) with QRS vectors at correct times
    """
    cardiac_vectors = np.zeros((len(t_axis), 3))
    
    # Define the QRS time window
    qrs_end_time = qrs_onset_time + qrs_duration
    qrs_mask = (t_axis >= qrs_onset_time) & (t_axis <= qrs_end_time)
    
    if not np.any(qrs_mask):
        return cardiac_vectors
    
    # Get indices and time points within QRS
    qrs_indices = np.where(qrs_mask)[0]
    qrs_time_points = t_axis[qrs_mask]
    qrs_progress = (qrs_time_points - qrs_onset_time) / qrs_duration  # 0 to 1
    
    # QRS region processing
    
    # Calculate instantaneous cardiac vector for each QRS time point
    for i, (idx, progress) in enumerate(zip(qrs_indices, qrs_progress)):
        vector_direction, vector_magnitude = calculate_instantaneous_qrs_vector(progress, qrs_amplitudes, target_axis_degrees)
        cardiac_vectors[idx] = vector_direction * vector_magnitude
        
        # Vector calculation complete
    
    return cardiac_vectors


def calculate_instantaneous_qrs_vector(progress: float, qrs_amplitudes: Dict[str, float], target_axis_degrees: float = None) -> tuple:
    """
    Calculate the instantaneous cardiac vector direction and magnitude during QRS.
    
    Based on physiological ventricular depolarization sequence with proper amplitude scaling.
    
    Args:
        progress: QRS progress from 0.0 (onset) to 1.0 (end)
        qrs_amplitudes: Amplitude scaling factors
        
    Returns:
        (vector_direction, magnitude): 3D direction vector and scalar magnitude
    """
    # Clamp progress to valid range
    progress = max(0.0, min(1.0, progress))
    
    # Calculate overlapping phase contributions with proper scaling
    septal_contrib = 0.0
    free_wall_contrib = 0.0
    basal_contrib = 0.0
    
    # Calculate axis-specific phase contributions
    septal_factor, free_wall_factor, basal_factor = calculate_phase_contributions(target_axis_degrees)
    
    # Septal contribution (0-35% of QRS, peaks at 15%)
    if progress <= 0.35:
        septal_envelope = np.exp(-((progress - 0.15) / 0.10) ** 2)  
        septal_contrib = abs(qrs_amplitudes.get('q_amplitude', -0.08)) * septal_envelope * septal_factor
    
    # Free wall contribution (10-70% of QRS, peaks at 40%)
    if 0.10 <= progress <= 0.70:
        free_wall_envelope = np.exp(-((progress - 0.40) / 0.18) ** 2)  
        free_wall_contrib = abs(qrs_amplitudes.get('r_amplitude', 1.4)) * free_wall_envelope * free_wall_factor
    
    # Basal contribution (65-100% of QRS, peaks at 85%)
    if progress >= 0.65:
        basal_envelope = np.exp(-((progress - 0.85) / 0.12) ** 2)  
        basal_contrib = abs(qrs_amplitudes.get('s_amplitude', -0.25)) * basal_envelope * basal_factor
    
    # Phase-dependent vector directions (base physiological directions)
    base_septal_direction = np.array([-0.7, 0.1, 0.3])    # Rightward, anterior (Q in lateral, r in V1)
    base_free_wall_direction = np.array([0.6, 0.7, 0.4])  # Left, inferior, anterior (dominant R)
    base_basal_direction = np.array([0.1, 0.1, -1.0])     # Strongly posterior (negative Z for deep S in V1)
    
    # Apply axis-specific pathophysiology if specified
    if target_axis_degrees is not None:
        septal_direction, free_wall_direction, basal_direction = calculate_pathophysiological_vectors(
            target_axis_degrees, base_septal_direction, base_free_wall_direction, base_basal_direction
        )
        # Using pathophysiological vectors for axis override
    else:
        # Use base physiological directions
        septal_direction = base_septal_direction
        free_wall_direction = base_free_wall_direction  
        basal_direction = base_basal_direction
    
    # Normalize directions
    septal_direction = septal_direction / np.linalg.norm(septal_direction)
    free_wall_direction = free_wall_direction / np.linalg.norm(free_wall_direction)
    basal_direction = basal_direction / np.linalg.norm(basal_direction)
    
    # Calculate weighted vector sum
    total_vector = (septal_contrib * septal_direction + 
                   free_wall_contrib * free_wall_direction + 
                   basal_contrib * basal_direction)
    
    # Get magnitude and normalize direction
    magnitude = np.linalg.norm(total_vector)
    if magnitude > 1e-6:
        direction = total_vector / magnitude
    else:
        direction = np.array([1.0, 0.0, 0.0])  # Default leftward
        magnitude = 0.0
    
    # QRS vector calculation complete
    
    return direction, magnitude

def rotate_vector_in_frontal_plane(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate a 3D vector in the frontal plane (X-Y plane) by the specified angle.
    Preserves the Z component for proper precordial lead morphology.
    
    Args:
        vector: 3D vector to rotate
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated 3D vector
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix for frontal plane (X-Y plane)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]  # Preserve Z component
    ])
    
    return rotation_matrix @ vector

def calculate_pathophysiological_vectors(target_axis_degrees: float, base_septal: np.ndarray, 
                                       base_free_wall: np.ndarray, base_basal: np.ndarray) -> tuple:
    """
    Calculate pathophysiologically accurate vectors based on what causes different electrical axes.
    
    Args:
        target_axis_degrees: Target electrical axis
        base_septal, base_free_wall, base_basal: Base physiological directions
        
    Returns:
        (septal_direction, free_wall_direction, basal_direction): Modified vectors for the axis
    """
    
    if -30 <= target_axis_degrees <= 90:
        # Normal axis range - use base vectors with minor adjustments
        rotation_angle = target_axis_degrees - 49.4  # Adjust from base ~50° axis
        septal_direction = rotate_vector_in_frontal_plane(base_septal, rotation_angle)
        free_wall_direction = rotate_vector_in_frontal_plane(base_free_wall, rotation_angle)
        basal_direction = rotate_vector_in_frontal_plane(base_basal, rotation_angle)
        
    elif target_axis_degrees > 90:
        # Right axis deviation - RV hypertrophy/dominance
        # Enhance rightward forces, particularly in free wall and basal phases
        
        # Septal: slightly more rightward
        septal_direction = np.array([-0.8, 0.0, 0.4])  # More rightward, more anterior
        
        # Free wall: shift toward RV dominance  
        rv_dominance_factor = min((target_axis_degrees - 90) / 90, 1.0)  # 0 to 1
        free_wall_direction = np.array([
            0.6 - 1.2 * rv_dominance_factor,  # Shift from left to right
            0.7,  # Maintain inferior
            0.4 + 0.3 * rv_dominance_factor   # More anterior with RV dominance
        ])
        
        # Basal: strong rightward forces (enhanced RV)
        basal_direction = np.array([
            -0.6 - 0.4 * rv_dominance_factor,  # Even more rightward
            0.1,
            -0.8 + 0.6 * rv_dominance_factor   # Less posterior, more anterior
        ])
        
        # Right axis deviation pathophysiology applied
        
    else:  # target_axis_degrees < -30
        # Left axis deviation - LV hypertrophy or left anterior fascicular block
        # Enhance leftward forces, reduce or eliminate rightward components
        
        lv_dominance_factor = min(abs(target_axis_degrees + 30) / 60, 1.0)  # 0 to 1
        
        # Septal: may be reduced or reversed in fascicular block
        septal_direction = np.array([
            -0.7 + 1.4 * lv_dominance_factor,  # Shift toward leftward
            0.1 - 0.6 * lv_dominance_factor,   # Shift toward superior  
            0.3
        ])
        
        # Free wall: enhanced LV forces
        free_wall_direction = np.array([
            0.6 + 0.4 * lv_dominance_factor,   # Even more leftward
            0.7 - 1.2 * lv_dominance_factor,   # Shift toward superior
            0.4
        ])
        
        # Basal: strong leftward, may be minimal in extreme LAD
        basal_direction = np.array([
            0.1 + 0.7 * lv_dominance_factor,   # Shift leftward
            0.1 - 0.6 * lv_dominance_factor,   # Shift superior
            -1.0 - 0.5 * lv_dominance_factor   # Even more posterior
        ])
        
        # Left axis deviation pathophysiology applied
    
    return septal_direction, free_wall_direction, basal_direction

def calculate_phase_contributions(target_axis_degrees: float) -> tuple:
    """
    Calculate how much each phase contributes based on the pathophysiology causing the axis.
    
    Args:
        target_axis_degrees: Target electrical axis
        
    Returns:
        (septal_factor, free_wall_factor, basal_factor): Multiplication factors for each phase
    """
    if target_axis_degrees is None:
        # Normal contributions
        return 1.0, 1.0, 1.0
    
    if -30 <= target_axis_degrees <= 90:
        # Normal axis range - standard contributions
        return 1.0, 1.0, 1.0
        
    elif target_axis_degrees > 90:
        # Right axis deviation - RV hypertrophy/dominance
        # Increase basal (RV) contribution, slightly reduce septal
        rv_dominance_factor = min((target_axis_degrees - 90) / 90, 1.0)
        
        septal_factor = 1.0 - 0.3 * rv_dominance_factor      # Reduce septal slightly
        free_wall_factor = 1.0 + 0.2 * rv_dominance_factor   # Slight increase in LV
        basal_factor = 1.0 + 1.5 * rv_dominance_factor       # Major increase in RV
        
        # Right axis phase contributions calculated
        
    else:  # target_axis_degrees < -30
        # Left axis deviation - LV hypertrophy or fascicular block
        # Increase free wall (LV), reduce or eliminate septal if fascicular block
        lv_dominance_factor = min(abs(target_axis_degrees + 30) / 60, 1.0)
        
        septal_factor = 1.0 - 0.8 * lv_dominance_factor      # Major reduction (fascicular block effect)
        free_wall_factor = 1.0 + 1.0 * lv_dominance_factor   # Major increase in LV
        basal_factor = 1.0 - 0.2 * lv_dominance_factor       # Slight reduction in terminal forces
        
        # Left axis phase contributions calculated
    
    return septal_factor, free_wall_factor, basal_factor

def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS, draw_only_p: bool = False, is_flutter_wave_itself: bool = False):
    if is_flutter_wave_itself:
        wave_duration = params.get('p_duration', 0.10)
        wave_amplitude = params.get('p_amplitude', -0.2)
        num_samples = int(wave_duration * fs)
        if num_samples <= 0: return np.array([]), np.array([]), 0.0
        t_relative = np.linspace(0, wave_duration, num_samples, endpoint=False)
        waveform = np.zeros_like(t_relative)
        peak_time_ratio = 0.7
        nadir_idx = int(num_samples * peak_time_ratio)
        if nadir_idx < num_samples -1 and nadir_idx > 0 :
            waveform[:nadir_idx] = np.linspace(0, wave_amplitude, nadir_idx)
            waveform[nadir_idx:] = np.linspace(wave_amplitude, wave_amplitude * 0.2 , num_samples - nadir_idx)
        elif num_samples > 0 : waveform[:] = wave_amplitude / 2
        return t_relative, waveform, 0.0

    p_wave_total_offset = params.get('pr_interval', 0) if params.get('p_amplitude',0) !=0 else 0
    p_duration = params.get('p_duration', 0) if params.get('p_amplitude',0) !=0 else 0
    qrs_duration = 0.0 if draw_only_p else params.get('qrs_duration', 0.1)
    st_duration = 0.0 if draw_only_p else params.get('st_duration', 0.1)
    t_duration = 0.0 if draw_only_p else params.get('t_duration', 0.1)
    duration_from_p_onset_to_qrs_onset = p_wave_total_offset
    total_complex_duration = duration_from_p_onset_to_qrs_onset + \
                             qrs_duration + st_duration + t_duration + 0.05 # Small buffer
    if draw_only_p: total_complex_duration = p_duration + 0.05 if p_duration > 0 else 0.05

    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: return np.array([]), np.array([]), 0.0
    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)

    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2
        # Use Fourier-based P-wave for more physiological morphology
        beat_waveform += fourier_p_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_duration)

    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        if qrs_duration > 0:
            if params.get('q_amplitude', 0) != 0:
                q_center = qrs_onset_in_array_time + qrs_duration * 0.15
                q_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
            if params.get('r_amplitude', 0) != 0:
                r_center = qrs_onset_in_array_time + qrs_duration * 0.4
                r_width_std_dev = qrs_duration / 6 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
            if params.get('s_amplitude', 0) != 0:
                s_center = qrs_onset_in_array_time + qrs_duration * 0.75
                s_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)

        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            t_center = t_onset_in_array_time + t_duration / 2
            t_width_std_dev = t_duration / 4 if t_duration > 0 else 1e-3
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)

    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset


def generate_single_beat_3d_vectors(
    params: Dict[str, float], 
    beat_type: str,
    fs: int = FS, 
    draw_only_p: bool = False, 
    is_flutter_wave_itself: bool = False,
    torsades_beat_number: Optional[int] = None,
    enable_axis_override: bool = False,
    target_axis_degrees: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate 3D cardiac vectors for a single beat.
    
    Returns:
        - t_relative: Time axis relative to beat onset
        - cardiac_vectors: Array of shape (num_samples, 3) representing [Vx, Vy, Vz] over time
        - qrs_offset: Time offset from start of array to QRS onset
    """
    if is_flutter_wave_itself:
        wave_duration = params.get('p_duration', 0.10)
        wave_amplitude = params.get('p_amplitude', -0.2)
        num_samples = int(wave_duration * fs)
        if num_samples <= 0: 
            return np.array([]), np.zeros((0, 3)), 0.0
            
        t_relative = np.linspace(0, wave_duration, num_samples, endpoint=False)
        
        # Flutter waves are atrial activity, use a generic atrial direction
        flutter_direction = np.array([0.1, 0.9, 0.2])  # Mostly inferior direction
        flutter_direction = flutter_direction / np.linalg.norm(flutter_direction)
        
        # Create scalar waveform
        waveform_scalar = np.zeros_like(t_relative)
        peak_time_ratio = 0.7
        nadir_idx = int(num_samples * peak_time_ratio)
        if nadir_idx < num_samples - 1 and nadir_idx > 0:
            waveform_scalar[:nadir_idx] = np.linspace(0, wave_amplitude, nadir_idx)
            waveform_scalar[nadir_idx:] = np.linspace(wave_amplitude, wave_amplitude * 0.2, num_samples - nadir_idx)
        elif num_samples > 0:
            waveform_scalar[:] = wave_amplitude / 2
            
        # Convert to 3D vectors
        cardiac_vectors = np.outer(waveform_scalar, flutter_direction)
        return t_relative, cardiac_vectors, 0.0

    # Get 3D direction vectors for this beat type
    if beat_type not in BEAT_3D_DIRECTIONS:
        # Fallback to sinus directions if beat type not found
        beat_directions = BEAT_3D_DIRECTIONS["sinus"]
    else:
        beat_directions = BEAT_3D_DIRECTIONS[beat_type]
    
    # Check if this beat type supports dual-phase P-waves
    has_dual_phase_p = "P_PHASE1" in beat_directions and "P_PHASE2" in beat_directions
    if has_dual_phase_p:
        p_phase1_direction = beat_directions["P_PHASE1"]
        p_phase2_direction = beat_directions["P_PHASE2"]
        p_direction = None  # Will use dual-phase generation
    else:
        p_direction = beat_directions.get("P", np.array([0.0, 0.0, 0.0]))
        p_phase1_direction = None
        p_phase2_direction = None
    
    qrs_direction = beat_directions["QRS"] 
    
    # Override QRS direction if axis override is enabled
    if enable_axis_override and beat_type in ["sinus", "pac", "pac_high_ra", "pac_low_atrial", "pac_left_atrial", "afib_conducted", "flutter_conducted_qrs", "svt_beat"]:
        from .constants import calculate_qrs_vector_from_axis
        qrs_direction = calculate_qrs_vector_from_axis(target_axis_degrees)
    
    # Special handling for Torsades de Pointes - rotate QRS axis
    if beat_type == "torsades_beat" and torsades_beat_number is not None:
        from .constants import calculate_torsades_qrs_direction
        qrs_direction = calculate_torsades_qrs_direction(torsades_beat_number, qrs_direction)
    
    t_direction = beat_directions["T"]

    # Calculate timing parameters
    p_wave_total_offset = params.get('pr_interval', 0) if params.get('p_amplitude', 0) != 0 else 0
    p_duration = params.get('p_duration', 0) if params.get('p_amplitude', 0) != 0 else 0
    qrs_duration = 0.0 if draw_only_p else params.get('qrs_duration', 0.1)
    st_duration = 0.0 if draw_only_p else params.get('st_duration', 0.1)
    t_duration = 0.0 if draw_only_p else params.get('t_duration', 0.1)
    duration_from_p_onset_to_qrs_onset = p_wave_total_offset
    total_complex_duration = duration_from_p_onset_to_qrs_onset + \
                             qrs_duration + st_duration + t_duration + 0.05
    if draw_only_p: 
        total_complex_duration = p_duration + 0.05 if p_duration > 0 else 0.05

    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: 
        return np.array([]), np.zeros((0, 3)), 0.0
        
    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    cardiac_vectors = np.zeros((num_samples, 3))

    # Generate P wave vectors  
    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2
        
        if has_dual_phase_p and p_phase1_direction is not None and p_phase2_direction is not None:
            # Use dual-phase P-wave generation for realistic biphasic morphology
            p_wave_vectors = generate_biphasic_p_wave_vectors(
                t_relative_to_p_onset, 
                p_center, 
                params['p_amplitude'], 
                p_duration,
                p_phase1_direction, 
                p_phase2_direction
            )
            cardiac_vectors += p_wave_vectors
        else:
            # Use Fourier-based P-wave for single-phase generation
            p_scalar_waveform = fourier_p_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_duration)
            cardiac_vectors += np.outer(p_scalar_waveform, p_direction)

    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        
        # Generate QRS complex vectors
        if qrs_duration > 0:
            # Check if this beat type supports multi-phase QRS
            has_multi_phase_qrs = ("QRS_SEPTAL" in beat_directions and 
                                 "QRS_FREE_WALL" in beat_directions and 
                                 "QRS_BASAL" in beat_directions)
            
            if has_multi_phase_qrs:
                # Use physiological QRS generation based on real cardiac electrophysiology
                qrs_amplitudes = {
                    'q_amplitude': params.get('q_amplitude', 0),
                    'r_amplitude': params.get('r_amplitude', 0),
                    's_amplitude': params.get('s_amplitude', 0)
                }
                
                # Pass axis override if enabled
                axis_degrees = None
                if enable_axis_override:
                    axis_degrees = target_axis_degrees
                
                qrs_vectors = generate_physiological_qrs_vectors(
                    t_relative_to_p_onset,
                    qrs_onset_in_array_time,
                    qrs_duration,
                    qrs_amplitudes,
                    axis_degrees
                )
                cardiac_vectors += qrs_vectors
                
            else:
                # Fallback to legacy single-vector QRS generation
                qrs_scalar_waveform = np.zeros_like(t_relative_to_p_onset)
                
                if params.get('q_amplitude', 0) != 0:
                    q_center = qrs_onset_in_array_time + qrs_duration * 0.15
                    q_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                    qrs_scalar_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
                if params.get('r_amplitude', 0) != 0:
                    r_center = qrs_onset_in_array_time + qrs_duration * 0.4
                    r_width_std_dev = qrs_duration / 6 if qrs_duration > 0 else 1e-3
                    qrs_scalar_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
                if params.get('s_amplitude', 0) != 0:
                    s_center = qrs_onset_in_array_time + qrs_duration * 0.75
                    s_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                    qrs_scalar_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)
                
                cardiac_vectors += np.outer(qrs_scalar_waveform, qrs_direction)

        # Generate T wave vectors
        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            t_center = t_onset_in_array_time + t_duration / 2
            t_width_std_dev = t_duration / 4 if t_duration > 0 else 1e-3
            t_scalar_waveform = gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)
            cardiac_vectors += np.outer(t_scalar_waveform, t_direction)

    return t_relative_to_p_onset, cardiac_vectors, p_wave_total_offset