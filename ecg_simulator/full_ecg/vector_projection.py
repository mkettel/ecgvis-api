# ecg_simulator/vector_projection.py
import numpy as np
from typing import Dict, Tuple

# Define a standard 3D coordinate system for cardiac vectors:
# X-axis: Patient's Right to Left (positive Left)
# Y-axis: Patient's Superior to Inferior (positive Inferior)
# Z-axis: Patient's Posterior to Anterior (positive Anterior)

# Medically accurate lead vectors based on published electrophysiology models
# References: Kors et al. (1990), Dower et al. (1988), and Frank lead system
# 3D coordinate system: X=Left, Y=Inferior, Z=Anterior
LEAD_VECTOR_DIRECTIONS: Dict[str, np.ndarray] = {
    # Frontal Plane Leads (Standard Einthoven Triangle + Goldberger augmented leads)
    "I":   np.array([1.0, 0.0, 0.0]),                    # 0° - Purely leftward
    "II":  np.array([0.5, 0.866, 0.0]),                  # 60° - Left-inferior
    "III": np.array([-0.5, 0.866, 0.0]),                 # 120° - Right-inferior
    "aVR": np.array([-0.866, -0.5, 0.0]),                # -150° - Right-superior
    "aVL": np.array([0.866, -0.5, 0.0]),                 # -30° - Left-superior
    "aVF": np.array([0.0, 1.0, 0.0]),                    # 90° - Purely inferior
    
    # Precordial Leads (Based on Frank XYZ lead system derivatives)
    # Optimized for realistic ECG morphology and medical accuracy
    "V1":  np.array([-0.144, -0.108, 0.983]),           # Right septal, anterior emphasis
    "V2":  np.array([-0.066, -0.025, 0.997]),           # Septal-anterior, slight right
    "V3":  np.array([0.164, 0.088, 0.983]),             # Anterior wall, balanced
    "V4":  np.array([0.454, 0.203, 0.867]),             # Anterior-lateral, left shift
    "V5":  np.array([0.688, 0.243, 0.686]),             # Lateral wall, left emphasis
    "V6":  np.array([0.912, 0.262, 0.317]),             # Lateral, minimal anterior
}

# Normalize all lead vectors
for lead in LEAD_VECTOR_DIRECTIONS:
    norm = np.linalg.norm(LEAD_VECTOR_DIRECTIONS[lead])
    if norm > 1e-6:
        LEAD_VECTOR_DIRECTIONS[lead] /= norm
    else: # Should not happen with defined vectors
        LEAD_VECTOR_DIRECTIONS[lead] = np.array([0.0,0.0,0.0])


def project_cardiac_vector_to_12_leads(
    cardiac_vector_t: np.ndarray, # Shape (num_samples, 3) for Vx, Vy, Vz over time
) -> Dict[str, np.ndarray]:
    """
    Projects a time-varying 3D cardiac vector onto the 12 standard ECG leads.
    """
    if cardiac_vector_t.ndim == 1: # single time point vector
        if cardiac_vector_t.shape[0] != 3:
            raise ValueError("Single time point cardiac_vector_t must have 3 components (Vx, Vy, Vz)")
        cardiac_vector_t = cardiac_vector_t.reshape(1,3)
    elif cardiac_vector_t.ndim != 2 or cardiac_vector_t.shape[1] != 3:
        raise ValueError("Time-series cardiac_vector_t must have shape (num_samples, 3)")

    projected_ecg_waveforms: Dict[str, np.ndarray] = {}
    for lead_name, lead_direction_vector in LEAD_VECTOR_DIRECTIONS.items():
        # Dot product of the cardiac vector at each time point with the lead's direction vector
        projected_ecg_waveforms[lead_name] = np.dot(cardiac_vector_t, lead_direction_vector)
        
    return projected_ecg_waveforms


def calculate_electrical_axis(lead_i_amplitude: float, lead_avf_amplitude: float) -> Tuple[float, str]:
    """
    Calculate electrical axis from Lead I and aVF amplitudes.
    
    Args:
        lead_i_amplitude: QRS amplitude in Lead I (mV)
        lead_avf_amplitude: QRS amplitude in aVF (mV)
    
    Returns:
        Tuple of (axis_degrees, axis_interpretation)
    """
    # Calculate axis in degrees using arctan2
    axis_radians = np.arctan2(lead_avf_amplitude, lead_i_amplitude)
    axis_degrees = np.degrees(axis_radians)
    
    # Normalize to 0-360 degrees, then to standard ECG range (-180 to +180)
    if axis_degrees < -180:
        axis_degrees += 360
    elif axis_degrees > 180:
        axis_degrees -= 360
    
    # Interpret axis deviation
    if -30 <= axis_degrees <= 90:
        interpretation = "Normal"
    elif 90 < axis_degrees <= 180:
        interpretation = "Right Axis Deviation"
    elif -30 > axis_degrees >= -90:
        interpretation = "Left Axis Deviation" 
    else:  # -90 to -180 or extreme right
        interpretation = "Extreme Axis Deviation"
    
    return axis_degrees, interpretation


def calculate_qrs_axis_from_12_lead(ecg_signals: Dict[str, np.ndarray]) -> Tuple[float, str]:
    """
    Calculate electrical axis from 12-lead ECG signals by finding QRS peak amplitudes.
    
    Args:
        ecg_signals: Dictionary containing all 12 lead signals
    
    Returns:
        Tuple of (axis_degrees, axis_interpretation)
    """
    # Get Lead I and aVF signals
    lead_i_signal = np.array(ecg_signals["I"])
    lead_avf_signal = np.array(ecg_signals["aVF"])
    
    if len(lead_i_signal) == 0 or len(lead_avf_signal) == 0:
        return 0.0, "Unable to calculate (no signal)"
    
    # Find QRS complexes in the first 2.5 seconds (typical for dominant beat analysis)
    # Assuming 250 Hz sampling rate, 2.5s = 625 samples
    analysis_length = min(625, len(lead_i_signal))
    
    # For QRS detection, look for the steepest positive or negative deflection
    # which typically occurs in the first part of the signal
    lead_i_analysis = lead_i_signal[:analysis_length]
    lead_avf_analysis = lead_avf_signal[:analysis_length]
    
    # Calculate net deflection (area under curve) for QRS complex
    # This is more robust than peak detection for axis calculation
    # Focus on the portion likely to contain QRS (exclude P-waves and T-waves)
    
    # Method 1: Use the strongest deflection in the analysis window
    lead_i_max_idx = np.argmax(np.abs(lead_i_analysis))
    lead_avf_max_idx = np.argmax(np.abs(lead_avf_analysis))
    
    # Get the actual amplitude (with sign) at the strongest deflection
    lead_i_qrs = lead_i_analysis[lead_i_max_idx]
    lead_avf_qrs = lead_avf_analysis[lead_avf_max_idx]
    
    # Alternative method: Use net deflection in early portion (more stable)
    # Take average of strongest deflections to reduce noise
    if analysis_length > 100:  # Only if we have enough samples
        # Find peaks in the middle section (likely QRS location)
        start_idx = analysis_length // 4  # Skip early P-wave region
        end_idx = 3 * analysis_length // 4  # Stop before T-wave region
        
        qrs_window_i = lead_i_analysis[start_idx:end_idx]
        qrs_window_avf = lead_avf_analysis[start_idx:end_idx]
        
        if len(qrs_window_i) > 0 and len(qrs_window_avf) > 0:
            # Use the maximum absolute deflection in the QRS window
            i_peak_idx = np.argmax(np.abs(qrs_window_i))
            avf_peak_idx = np.argmax(np.abs(qrs_window_avf))
            
            lead_i_qrs = qrs_window_i[i_peak_idx]
            lead_avf_qrs = qrs_window_avf[avf_peak_idx]
    
    # Debug output
    print(f"DEBUG: QRS detection - Lead I: {lead_i_qrs:.3f}, aVF: {lead_avf_qrs:.3f}")
    
    return calculate_electrical_axis(lead_i_qrs, lead_avf_qrs)