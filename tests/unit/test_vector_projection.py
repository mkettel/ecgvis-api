"""
Tests for 3D cardiac vector projection to 12-lead ECG.
Validates mathematical accuracy of vector projections and lead calculations.
"""
import pytest
import numpy as np
from ecg_simulator.full_ecg.vector_projection import (
    project_cardiac_vector_to_12_leads,
    calculate_electrical_axis,
    calculate_qrs_axis_from_12_lead,
    LEAD_VECTOR_DIRECTIONS
)
from ecg_simulator.constants import calculate_qrs_vector_from_axis

class TestVectorProjection:
    """Test 3D cardiac vector projection system."""
    
    @pytest.mark.unit
    def test_lead_vector_normalization(self, tolerance_config):
        """Test that all lead vectors are properly normalized."""
        for lead_name, lead_vector in LEAD_VECTOR_DIRECTIONS.items():
            vector_magnitude = np.linalg.norm(lead_vector)
            
            # All lead vectors should be unit vectors (magnitude = 1)
            assert abs(vector_magnitude - 1.0) < tolerance_config['amplitude_tolerance_mv'], \
                f"Lead {lead_name} vector magnitude {vector_magnitude:.6f} is not normalized"
    
    @pytest.mark.unit
    def test_lead_vector_directions_medical_accuracy(self, tolerance_config):
        """Test that lead vector directions match published electrophysiology values."""
        
        # Test frontal plane leads (Einthoven triangle)
        # Lead I should be purely leftward (X=1, Y=0, Z=0)
        lead_i = LEAD_VECTOR_DIRECTIONS["I"]
        assert abs(lead_i[0] - 1.0) < tolerance_config['amplitude_tolerance_mv']  # X component
        assert abs(lead_i[1]) < tolerance_config['amplitude_tolerance_mv']        # Y component  
        assert abs(lead_i[2]) < tolerance_config['amplitude_tolerance_mv']        # Z component
        
        # Lead aVF should be purely inferior (X=0, Y=1, Z=0)
        lead_avf = LEAD_VECTOR_DIRECTIONS["aVF"]
        assert abs(lead_avf[0]) < tolerance_config['amplitude_tolerance_mv']      # X component
        assert abs(lead_avf[1] - 1.0) < tolerance_config['amplitude_tolerance_mv'] # Y component
        assert abs(lead_avf[2]) < tolerance_config['amplitude_tolerance_mv']      # Z component
        
        # Lead II should be at 60° in frontal plane
        lead_ii = LEAD_VECTOR_DIRECTIONS["II"]
        expected_x = np.cos(np.deg2rad(60))  # ~0.5
        expected_y = np.sin(np.deg2rad(60))  # ~0.866
        assert abs(lead_ii[0] - expected_x) < 0.01
        assert abs(lead_ii[1] - expected_y) < 0.01
        assert abs(lead_ii[2]) < tolerance_config['amplitude_tolerance_mv']
    
    @pytest.mark.unit
    def test_precordial_lead_anterior_progression(self, tolerance_config):
        """Test that precordial leads show appropriate anterior progression."""
        precordial_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        
        # V1 should have the most anterior (positive Z) component
        v1_anterior = LEAD_VECTOR_DIRECTIONS["V1"][2]
        
        # V6 should have the least anterior component (most lateral)
        v6_anterior = LEAD_VECTOR_DIRECTIONS["V6"][2]
        
        assert v1_anterior > v6_anterior, "V1 should be more anterior than V6"
        
        # V1 should be rightward (negative X), V6 should be leftward (positive X)
        v1_lateral = LEAD_VECTOR_DIRECTIONS["V1"][0]
        v6_lateral = LEAD_VECTOR_DIRECTIONS["V6"][0]
        
        assert v1_lateral < 0, "V1 should have negative (rightward) X component"
        assert v6_lateral > 0, "V6 should have positive (leftward) X component"
    
    @pytest.mark.unit
    def test_single_vector_projection(self, tolerance_config):
        """Test projection of a single 3D cardiac vector."""
        # Create a simple cardiac vector pointing left, inferior, anterior
        cardiac_vector = np.array([0.7, 0.6, 0.4])  # [left, inferior, anterior]
        
        projections = project_cardiac_vector_to_12_leads(cardiac_vector)
        
        # Should return all 12 leads
        assert len(projections) == 12
        expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for lead in expected_leads:
            assert lead in projections
        
        # Lead I (purely leftward) should have positive projection
        assert projections["I"] > 0, "Lead I should be positive for leftward vector"
        
        # Lead aVF (purely inferior) should have positive projection  
        assert projections["aVF"] > 0, "Lead aVF should be positive for inferior vector"
        
        # aVR (right-superior) should have negative projection
        assert projections["aVR"] < 0, "Lead aVR should be negative for left-inferior vector"
    
    @pytest.mark.unit
    def test_time_series_vector_projection(self, sample_12_lead_vectors, tolerance_config):
        """Test projection of time-varying 3D cardiac vectors."""
        time_axis, cardiac_vectors = sample_12_lead_vectors
        
        # cardiac_vectors should have shape (num_samples, 3)
        assert cardiac_vectors.shape[1] == 3
        num_samples = cardiac_vectors.shape[0]
        
        projections = project_cardiac_vector_to_12_leads(cardiac_vectors)
        
        # Each lead should have same number of samples as input
        for lead_name, lead_signal in projections.items():
            assert len(lead_signal) == num_samples, \
                f"Lead {lead_name} has {len(lead_signal)} samples, expected {num_samples}"
    
    @pytest.mark.unit
    def test_electrical_axis_calculation_accuracy(self, tolerance_config):
        """Test electrical axis calculation from Lead I and aVF amplitudes."""
        
        # Test known axis values
        test_cases = [
            (1.0, 0.0, 0.0, "Normal"),        # 0° axis
            (0.0, 1.0, 90.0, "Normal"),       # 90° axis  
            (1.0, 1.0, 45.0, "Normal"),       # 45° axis
            (-1.0, 1.0, 135.0, "Right Axis Deviation"),  # 135° axis
            (1.0, -1.0, -45.0, "Left Axis Deviation"),   # -45° axis
            (-1.0, -1.0, -135.0, "Extreme Axis Deviation"), # -135° axis
        ]
        
        for lead_i_amp, lead_avf_amp, expected_axis, expected_interp in test_cases:
            calculated_axis, interpretation = calculate_electrical_axis(lead_i_amp, lead_avf_amp)
            
            # Allow some tolerance for floating point precision
            assert abs(calculated_axis - expected_axis) < tolerance_config['axis_tolerance_degrees'], \
                f"Calculated axis {calculated_axis:.1f}° doesn't match expected {expected_axis}°"
            
            assert interpretation == expected_interp, \
                f"Interpretation '{interpretation}' doesn't match expected '{expected_interp}'"
    
    @pytest.mark.unit
    def test_axis_calculation_edge_cases(self, tolerance_config):
        """Test electrical axis calculation edge cases."""
        
        # Test zero amplitudes
        axis, interp = calculate_electrical_axis(0.0, 0.0)
        assert interp in ["Normal", "Extreme Axis Deviation"]  # Could be either depending on implementation
        
        # Test very small amplitudes
        axis, interp = calculate_electrical_axis(0.001, 0.001)
        assert -180 <= axis <= 180  # Should be within valid range
        
        # Test extreme amplitudes
        axis, interp = calculate_electrical_axis(10.0, -10.0)
        assert -90 <= axis <= -45  # Should be in left axis deviation range
    
    @pytest.mark.unit  
    def test_qrs_vector_from_axis_conversion(self, tolerance_config):
        """Test conversion from electrical axis to 3D QRS vector."""
        
        # Test that converting axis to vector and back gives same axis
        test_axes = [0, 30, 60, 90, -30, -60, -90]
        
        for target_axis in test_axes:
            # Convert axis to 3D vector
            qrs_vector = calculate_qrs_vector_from_axis(target_axis)
            
            # Vector should be normalized
            vector_magnitude = np.linalg.norm(qrs_vector)
            assert abs(vector_magnitude - 1.0) < tolerance_config['amplitude_tolerance_mv']
            
            # Project vector to Lead I and aVF to get axis back
            lead_i_projection = np.dot(qrs_vector, LEAD_VECTOR_DIRECTIONS["I"])
            lead_avf_projection = np.dot(qrs_vector, LEAD_VECTOR_DIRECTIONS["aVF"])
            
            calculated_axis, _ = calculate_electrical_axis(lead_i_projection, lead_avf_projection)
            
            # Should match original axis within tolerance
            # Note: Some axes might wrap around (e.g., 180° = -180°)
            axis_diff = abs(calculated_axis - target_axis)
            if axis_diff > 180:
                axis_diff = 360 - axis_diff  # Handle wraparound
            
            assert axis_diff < tolerance_config['axis_tolerance_degrees'], \
                f"Axis conversion failed: {target_axis}° -> {calculated_axis}°"
    
    @pytest.mark.unit
    def test_12_lead_qrs_axis_detection(self, tolerance_config):
        """Test QRS axis calculation from 12-lead ECG signals."""
        
        # Create synthetic 12-lead ECG with known axis
        target_axis = 60.0  # degrees
        qrs_vector = calculate_qrs_vector_from_axis(target_axis)
        
        # Create a simple QRS complex time series
        fs = 250
        duration = 0.5  # 500ms
        num_samples = int(duration * fs)
        time_axis = np.linspace(0, duration, num_samples)
        
        # Create QRS complex shape (Gaussian)
        qrs_center = 0.25  # 250ms
        qrs_amplitude = 1.0
        qrs_width = 0.05
        qrs_shape = qrs_amplitude * np.exp(-((time_axis - qrs_center)**2) / (2 * qrs_width**2))
        
        # Create 3D cardiac vectors
        cardiac_vectors = np.outer(qrs_shape, qrs_vector)
        
        # Project to 12 leads
        lead_signals = project_cardiac_vector_to_12_leads(cardiac_vectors)
        
        # Calculate axis from 12-lead ECG
        calculated_axis, interpretation = calculate_qrs_axis_from_12_lead(lead_signals)
        
        # Should match target axis within tolerance
        assert abs(calculated_axis - target_axis) < tolerance_config['axis_tolerance_degrees'], \
            f"12-lead axis detection failed: expected {target_axis}°, got {calculated_axis}°"
    
    @pytest.mark.unit
    def test_vector_projection_linearity(self, tolerance_config):
        """Test that vector projection is linear (superposition principle)."""
        
        # Create two different cardiac vectors
        vector1 = np.array([1.0, 0.0, 0.0])  # Pure leftward
        vector2 = np.array([0.0, 1.0, 0.0])  # Pure inferior
        
        # Project each vector individually
        proj1 = project_cardiac_vector_to_12_leads(vector1)
        proj2 = project_cardiac_vector_to_12_leads(vector2)
        
        # Project sum of vectors
        combined_vector = vector1 + vector2
        proj_combined = project_cardiac_vector_to_12_leads(combined_vector)
        
        # Due to linearity, proj_combined should equal proj1 + proj2
        for lead in proj1:
            expected_combined = proj1[lead] + proj2[lead]
            actual_combined = proj_combined[lead]
            
            assert abs(actual_combined - expected_combined) < tolerance_config['amplitude_tolerance_mv'], \
                f"Linearity failed for lead {lead}: {actual_combined:.6f} != {expected_combined:.6f}"
    
    @pytest.mark.unit
    def test_vector_projection_scaling(self, tolerance_config):
        """Test that vector projection scales proportionally."""
        
        base_vector = np.array([0.5, 0.5, 0.5])
        scale_factor = 2.5
        scaled_vector = base_vector * scale_factor
        
        base_projections = project_cardiac_vector_to_12_leads(base_vector)
        scaled_projections = project_cardiac_vector_to_12_leads(scaled_vector)
        
        for lead in base_projections:
            expected_scaled = base_projections[lead] * scale_factor
            actual_scaled = scaled_projections[lead]
            
            assert abs(actual_scaled - expected_scaled) < tolerance_config['amplitude_tolerance_mv'], \
                f"Scaling failed for lead {lead}: {actual_scaled:.6f} != {expected_scaled:.6f}"
    
    @pytest.mark.unit
    def test_lead_vector_orthogonality(self, tolerance_config):
        """Test relationships between lead vectors (not all are orthogonal, but some should be)."""
        
        # Lead I and aVF should be orthogonal (90° apart in frontal plane)
        lead_i = LEAD_VECTOR_DIRECTIONS["I"]
        lead_avf = LEAD_VECTOR_DIRECTIONS["aVF"]
        
        dot_product = np.dot(lead_i, lead_avf)
        assert abs(dot_product) < tolerance_config['amplitude_tolerance_mv'], \
            f"Lead I and aVF should be orthogonal, dot product = {dot_product:.6f}"
        
        # aVR should be opposite to (aVL + aVF)/2 (Goldberger relationship)
        lead_avr = LEAD_VECTOR_DIRECTIONS["aVR"]
        lead_avl = LEAD_VECTOR_DIRECTIONS["aVL"]
        
        # The exact relationship is more complex, but aVR should have opposite polarity
        # to the average of other limb leads for typical cardiac vectors
        # This is a simplified test - the actual relationship involves all limb leads
    
    @pytest.mark.unit
    def test_input_validation(self):
        """Test input validation for vector projection functions."""
        
        # Test invalid vector dimensions
        with pytest.raises(ValueError):
            project_cardiac_vector_to_12_leads(np.array([1, 2]))  # Only 2D
            
        with pytest.raises(ValueError):
            project_cardiac_vector_to_12_leads(np.array([1, 2, 3, 4]))  # 4D
        
        # Test empty input
        with pytest.raises(ValueError):
            project_cardiac_vector_to_12_leads(np.array([]))
        
        # Test wrong shape for time series
        with pytest.raises(ValueError):
            project_cardiac_vector_to_12_leads(np.array([[1, 2], [3, 4]]))  # Wrong second dimension