# ecg_simulator/vector_projection.py
import numpy as np
from typing import Dict, Tuple

# Define a standard 3D coordinate system for cardiac vectors:
# X-axis: Patient's Right to Left (positive Left)
# Y-axis: Patient's Superior to Inferior (positive Inferior)
# Z-axis: Patient's Posterior to Anterior (positive Anterior)

# These are textbook approximations for lead vectors in 3D space.
# Accurate determination of these vectors can be complex and is part of your R&D.
# Values should be normalized unit vectors.
# TODO: Refine these vectors based on physiological models and desired accuracy.
LEAD_VECTOR_DIRECTIONS: Dict[str, np.ndarray] = {
    # Limb Leads (Frontal Plane primarily, simplified as 2D here, Z=0)
    "I":   np.array([1.0, 0.0, 0.0]),           # Points Left
    "II":  np.array([0.5, np.sqrt(3)/2, 0.0]),  # Points Left-Inferior (60 deg)
    "III": np.array([-0.5, np.sqrt(3)/2, 0.0]), # Points Right-Inferior (120 deg) # Corrected: Should be (LL-LA), so points more inferior and slightly less left than II, or even right if LA is more positive than RA
                                                # Textbook: Lead III = Lead II - Lead I. So, [-0.5, sqrt(3)/2, 0] is for (LA-RA) + (LL-RA) - (LA-RA). Let's use Einthoven's triangle properly.
                                                # If I is (LA-RA) and II is (LL-RA), then III = II-I = (LL-LA)
                                                # Vector for I: LA - RA
                                                # Vector for II: LL - RA
                                                # Vector for III: LL - LA
                                                # Assume RA @ (0,0,0), LA @ (1,0,0), LL @ (0.5, sqrt(3)/2, 0) for simplicity in frontal plane
                                                # I_dir = LA-RA = [1,0,0]
                                                # II_dir = LL-RA = [0.5, sqrt(3)/2, 0]
                                                # III_dir = LL-LA = [-0.5, sqrt(3)/2, 0] This seems consistent.

    "aVR": np.array([-np.sqrt(3)/2, -0.5, 0.0]), # Approx -150 deg. Or from -(I+II)/2, points to RA.
                                                # -(I_dir + II_dir)/2 = -([1,0,0] + [0.5, sqrt(3)/2, 0])/2 = -[1.5, sqrt(3)/2,0]/2 = [-0.75, -sqrt(3)/4, 0] -> normalize
    "aVL": np.array([np.sqrt(3)/2, -0.5, 0.0]), # Approx -30 deg. Or (I-III)/2, points to LA.
                                                # (I_dir - III_dir)/2 = ([1,0,0] - [-0.5, sqrt(3)/2,0])/2 = [1.5, -sqrt(3)/2, 0]/2 = [0.75, -sqrt(3)/4, 0] -> normalize (This needs correction for aVL's standard angle)
                                                # aVL is typically at -30 degrees. cos(-30), sin(-30) => [sqrt(3)/2, -0.5, 0]
    "aVF": np.array([0.0, 1.0, 0.0]),            # Points Inferior (90 deg). Or (II+III)/2, points to LL.
                                                # (II_dir + III_dir)/2 = ([0.5, sqrt(3)/2,0] + [-0.5, sqrt(3)/2,0])/2 = [0, sqrt(3), 0]/2 = [0, sqrt(3)/2, 0] -> normalize to [0,1,0]

    # Precordial Leads (Transverse Plane primarily) - These are rough estimates
    # TODO: These need significant R&D for accurate Z-axis components and orientations.
    "V1":  np.array([-0.2, -0.1, 0.9]), # Right sternal border, 4th ICS; points mostly anterior, slightly right & superior
    "V2":  np.array([0.0, -0.05, 0.95]),# Left sternal border, 4th ICS; points mostly anterior, slightly superior
    "V3":  np.array([0.3, 0.1, 0.8]),  # Midway V2-V4; anterior, slightly left & inferior
    "V4":  np.array([0.6, 0.2, 0.6]),  # Midclavicular line, 5th ICS; anterior-left-inferior
    "V5":  np.array([0.8, 0.1, 0.3]),  # Anterior axillary line, 5th ICS; left-anterior, slightly inferior
    "V6":  np.array([0.9, 0.0, 0.1]),  # Midaxillary line, 5th ICS; points mostly left, slightly anterior
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