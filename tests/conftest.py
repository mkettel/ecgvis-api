"""
Pytest configuration and shared fixtures for ECG simulator tests.
"""
import pytest
import numpy as np
from typing import Dict, Any
from ecg_simulator.api_models import AdvancedECGParams

@pytest.fixture
def basic_sinus_params():
    """Basic sinus rhythm parameters for testing."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=10.0,
        target_qtc_ms=400.0,
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False,
        enable_atrial_flutter=False,
        enable_vt=False,
        enable_torsades_risk=False
    )

@pytest.fixture
def fast_sinus_params():
    """Fast sinus rhythm for QT testing."""
    return AdvancedECGParams(
        heart_rate_bpm=120,
        duration_sec=5.0,
        target_qtc_ms=400.0,
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False,
        enable_atrial_flutter=False,
        enable_vt=False,
        enable_torsades_risk=False
    )

@pytest.fixture
def afib_params():
    """Atrial fibrillation parameters."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=10.0,
        enable_atrial_fibrillation=True,
        afib_average_ventricular_rate_bpm=80,
        afib_irregularity_factor=0.3,
        afib_fibrillation_wave_amplitude_mv=0.1,
        enable_pvc=False,
        enable_pac=False,
        enable_vt=False
    )

@pytest.fixture
def vt_params():
    """Ventricular tachycardia parameters."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=20.0,
        enable_vt=True,
        vt_start_time_sec=5.0,
        vt_duration_sec=8.0,
        vt_rate_bpm=180,
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False
    )

@pytest.fixture
def torsades_params():
    """Torsades de Pointes risk parameters."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=30.0,
        target_qtc_ms=520.0,  # Prolonged QTc
        enable_torsades_risk=True,
        torsades_probability_per_beat=0.01,  # Higher probability for testing
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False,
        enable_vt=False
    )

@pytest.fixture
def third_degree_av_block_params():
    """Complete AV block with escape rhythm."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=15.0,
        enable_third_degree_av_block=True,
        third_degree_escape_rhythm_origin="junctional",
        third_degree_escape_rate_bpm=45,
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False,
        enable_vt=False
    )

@pytest.fixture
def mobitz_ii_params():
    """Mobitz II AV block parameters."""
    return AdvancedECGParams(
        heart_rate_bpm=60,
        duration_sec=10.0,
        enable_mobitz_ii_av_block=True,
        mobitz_ii_p_waves_per_qrs=3,  # 3:1 AV block
        enable_pvc=False,
        enable_pac=False,
        enable_atrial_fibrillation=False,
        enable_vt=False
    )

@pytest.fixture
def sample_12_lead_vectors():
    """Sample 3D cardiac vectors for 12-lead testing."""
    # Create a simple QRS complex vector
    num_samples = 100
    time_axis = np.linspace(0, 0.4, num_samples)  # 400ms
    
    # Simple QRS vector pointing left, inferior, anterior
    qrs_direction = np.array([0.7, 0.6, 0.4])
    qrs_direction = qrs_direction / np.linalg.norm(qrs_direction)
    
    # Create a Gaussian QRS complex
    qrs_center = 0.2  # 200ms
    qrs_amplitude = 1.0
    qrs_width = 0.05  # 50ms width
    
    qrs_scalar = qrs_amplitude * np.exp(-((time_axis - qrs_center)**2) / (2 * qrs_width**2))
    cardiac_vectors = np.outer(qrs_scalar, qrs_direction)
    
    return time_axis, cardiac_vectors

@pytest.fixture
def tolerance_config():
    """Standard tolerance values for numerical comparisons."""
    return {
        'timing_tolerance_sec': 0.004,  # 4ms (1/250 Hz sampling)
        'amplitude_tolerance_mv': 0.01,  # 0.01 mV
        'qt_tolerance_ms': 10.0,  # 10ms QT tolerance
        'axis_tolerance_degrees': 5.0,  # 5 degree axis tolerance
        'rate_tolerance_bpm': 2.0  # 2 bpm tolerance
    }

@pytest.fixture
def medical_reference_values():
    """Reference values for medical validation."""
    return {
        'normal_qtc_range_ms': (350, 450),
        'normal_pr_interval_ms': (120, 200),
        'normal_qrs_duration_ms': (80, 120),
        'normal_axis_range_degrees': (-30, 90),
        'sinus_rate_range_bpm': (60, 100),
        'bazett_qtc_at_60bpm_ms': 400,  # Expected QTc at 60 bpm for 400ms target
        'bazett_qtc_at_120bpm_ms': 566,  # Expected QTc at 120 bpm for 400ms target (400 * sqrt(2))
    }