"""
Tests for individual beat generation and morphology.
Validates beat timing, amplitude, and waveform characteristics.
"""
import pytest
import numpy as np
from ecg_simulator.beat_generation import (
    generate_single_beat_morphology,
    generate_single_beat_3d_vectors,
    generate_biphasic_p_wave_vectors
)
from ecg_simulator.constants import BEAT_MORPHOLOGIES, BEAT_3D_DIRECTIONS

class TestBeatGeneration:
    """Test individual beat generation functions."""
    
    @pytest.mark.unit
    def test_sinus_beat_morphology_timing(self, tolerance_config):
        """Test sinus beat has correct timing intervals."""
        sinus_params = BEAT_MORPHOLOGIES["sinus"].copy()
        fs = 250  # 250 Hz sampling
        
        time_axis, waveform, qrs_offset = generate_single_beat_morphology(sinus_params, fs)
        
        # Should generate non-empty waveform
        assert len(waveform) > 0
        assert len(time_axis) == len(waveform)
        
        # QRS offset should match PR interval
        expected_qrs_offset = sinus_params["pr_interval"]
        assert abs(qrs_offset - expected_qrs_offset) < tolerance_config['timing_tolerance_sec']
        
        # Total duration should be reasonable (< 1 second for single beat)
        total_duration = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
        assert total_duration < 1.0, f"Beat duration {total_duration:.3f}s too long"
    
    @pytest.mark.unit
    def test_sinus_beat_amplitude_ranges(self, tolerance_config):
        """Test sinus beat amplitudes are within expected physiological ranges."""
        sinus_params = BEAT_MORPHOLOGIES["sinus"].copy()
        
        time_axis, waveform, _ = generate_single_beat_morphology(sinus_params, 250)
        
        # Check that waveform contains expected components
        max_amplitude = np.max(waveform)
        min_amplitude = np.min(waveform)
        
        # R wave should be dominant positive deflection
        assert max_amplitude > 0.5, f"R wave amplitude {max_amplitude:.3f} too small"
        assert max_amplitude < 3.0, f"R wave amplitude {max_amplitude:.3f} too large"
        
        # Should have some negative deflections (Q, S waves)
        assert min_amplitude < -0.1, f"No significant negative deflections found"
    
    @pytest.mark.unit
    def test_pvc_beat_characteristics(self, tolerance_config):
        """Test PVC beat has wide QRS and appropriate morphology."""
        pvc_params = BEAT_MORPHOLOGIES["pvc"].copy()
        
        time_axis, waveform, qrs_offset = generate_single_beat_morphology(pvc_params, 250)
        
        # PVC should have no P wave (PR interval = 0)
        assert abs(qrs_offset) < tolerance_config['timing_tolerance_sec']
        
        # PVC should have wide QRS duration
        qrs_duration = pvc_params["qrs_duration"]
        assert qrs_duration >= 0.12, f"PVC QRS duration {qrs_duration:.3f}s too narrow"
        
        # Should have content
        assert len(waveform) > 0
        assert not np.all(waveform == 0)
    
    @pytest.mark.unit
    def test_p_wave_only_generation(self, tolerance_config):
        """Test generation of P wave only (for AV blocks)."""
        sinus_params = BEAT_MORPHOLOGIES["sinus"].copy()
        
        time_axis, p_wave_only, qrs_offset = generate_single_beat_morphology(
            sinus_params, 250, draw_only_p=True
        )
        
        # Should generate shorter waveform (P wave only)
        assert len(p_wave_only) > 0
        
        # Duration should be much shorter than full beat
        p_wave_duration = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
        assert p_wave_duration < 0.2, f"P wave duration {p_wave_duration:.3f}s too long"
        
        # Should have P wave amplitude
        max_p_amplitude = np.max(p_wave_only)
        assert max_p_amplitude > 0.1, "P wave amplitude too small"
    
    @pytest.mark.unit
    def test_flutter_wave_generation(self, tolerance_config):
        """Test flutter wave generation has correct characteristics."""
        flutter_params = BEAT_MORPHOLOGIES["flutter_wave"].copy()
        flutter_params["p_duration"] = 0.2  # 200ms flutter wave
        flutter_params["p_amplitude"] = -0.3  # Negative flutter waves
        
        time_axis, flutter_wave, _ = generate_single_beat_morphology(
            flutter_params, 250, is_flutter_wave_itself=True
        )
        
        # Should generate waveform with specified duration
        assert len(flutter_wave) > 0
        
        # Should be predominantly negative
        min_amplitude = np.min(flutter_wave)
        assert min_amplitude < -0.2, f"Flutter wave not sufficiently negative: {min_amplitude:.3f}"
        
        # Duration should approximately match specified
        wave_duration = len(flutter_wave) / 250.0  # Convert samples to seconds
        expected_duration = flutter_params["p_duration"]
        assert abs(wave_duration - expected_duration) < 0.05, \
            f"Flutter wave duration {wave_duration:.3f}s != expected {expected_duration:.3f}s"
    
    @pytest.mark.unit
    def test_3d_vector_generation_sinus(self, tolerance_config):
        """Test 3D vector generation for sinus beats."""
        sinus_params = BEAT_MORPHOLOGIES["sinus"].copy()
        
        time_axis, cardiac_vectors, qrs_offset = generate_single_beat_3d_vectors(
            sinus_params, "sinus", 250
        )
        
        # Should return proper shape
        assert cardiac_vectors.shape[1] == 3, "Cardiac vectors should have 3 components"
        assert len(time_axis) == cardiac_vectors.shape[0], "Time axis length mismatch"
        
        # Should have non-zero vectors during QRS
        qrs_start_idx = int(qrs_offset * 250)
        qrs_duration_samples = int(sinus_params["qrs_duration"] * 250)
        qrs_end_idx = qrs_start_idx + qrs_duration_samples
        
        if qrs_end_idx <= len(cardiac_vectors):
            qrs_vectors = cardiac_vectors[qrs_start_idx:qrs_end_idx]
            qrs_magnitudes = np.linalg.norm(qrs_vectors, axis=1)
            
            # Should have significant QRS activity
            max_qrs_magnitude = np.max(qrs_magnitudes)
            assert max_qrs_magnitude > 0.1, f"QRS vector magnitude {max_qrs_magnitude:.3f} too small"
    
    @pytest.mark.unit
    def test_3d_vector_generation_pvc(self, tolerance_config):
        """Test 3D vector generation for PVC beats."""
        pvc_params = BEAT_MORPHOLOGIES["pvc"].copy()
        
        time_axis, cardiac_vectors, qrs_offset = generate_single_beat_3d_vectors(
            pvc_params, "pvc", 250
        )
        
        # PVC should have no P wave activity (early portion should be minimal)
        if len(cardiac_vectors) > 10:
            early_vectors = cardiac_vectors[:10]  # First 40ms
            early_magnitudes = np.linalg.norm(early_vectors, axis=1)
            max_early_magnitude = np.max(early_magnitudes)
            
            # Should be minimal early activity (no P wave), but allow some QRS activity
            assert max_early_magnitude < 1.0, f"PVC has excessive early activity: {max_early_magnitude:.3f}"
        
        # Should use PVC-specific direction vectors
        assert cardiac_vectors.shape[1] == 3
    
    @pytest.mark.unit
    def test_biphasic_p_wave_vectors(self, tolerance_config):
        """Test biphasic P wave vector generation for realistic atrial depolarization."""
        duration = 0.4  # 400ms
        num_samples = int(duration * 250)
        time_axis = np.linspace(0, duration, num_samples)
        
        p_center = 0.05  # 50ms
        p_amplitude = 0.3
        p_duration = 0.08  # 80ms
        
        # Define different directions for right and left atrial depolarization
        phase1_direction = np.array([0.2, 0.8, 0.6])  # Right atrial - rightward, inferior, anterior
        phase2_direction = np.array([0.8, 0.6, 0.2])  # Left atrial - leftward, inferior, less anterior
        
        # Normalize directions
        phase1_direction = phase1_direction / np.linalg.norm(phase1_direction)
        phase2_direction = phase2_direction / np.linalg.norm(phase2_direction)
        
        p_wave_vectors = generate_biphasic_p_wave_vectors(
            time_axis, p_center, p_amplitude, p_duration,
            phase1_direction, phase2_direction
        )
        
        # Should return proper shape
        assert p_wave_vectors.shape == (num_samples, 3)
        
        # Should have peak activity around P wave center
        p_center_idx = int(p_center * 250)
        p_start_idx = max(0, p_center_idx - 20)
        p_end_idx = min(len(p_wave_vectors), p_center_idx + 20)
        
        p_wave_magnitudes = np.linalg.norm(p_wave_vectors[p_start_idx:p_end_idx], axis=1)
        max_p_magnitude = np.max(p_wave_magnitudes)
        
        assert max_p_magnitude > 0.05, f"Biphasic P wave magnitude {max_p_magnitude:.3f} too small"
    
    @pytest.mark.unit
    @pytest.mark.xfail(reason="Torsades QRS axis rotation not yet implemented - returns same direction")
    def test_torsades_beat_axis_rotation(self, tolerance_config):
        """Test Torsades beat vector rotation with beat number."""
        torsades_params = BEAT_MORPHOLOGIES["torsades_beat"].copy()
        
        # Generate Torsades beats with different beat numbers
        beat_numbers = [0, 1, 2, 3, 4, 5]
        qrs_directions = []
        
        for beat_num in beat_numbers:
            time_axis, cardiac_vectors, _ = generate_single_beat_3d_vectors(
                torsades_params, "torsades_beat", 250, 
                torsades_beat_number=beat_num
            )
            
            # Extract QRS direction (should be during middle of beat)
            if len(cardiac_vectors) > 50:
                mid_idx = len(cardiac_vectors) // 2
                qrs_vector = cardiac_vectors[mid_idx]
                if np.linalg.norm(qrs_vector) > 0.01:  # Only if significant vector
                    qrs_direction = qrs_vector / np.linalg.norm(qrs_vector)
                    qrs_directions.append(qrs_direction)
        
        # Should have generated multiple directions
        assert len(qrs_directions) >= 3, "Should generate multiple QRS directions for Torsades"
        
        # Directions should be different (indicating rotation)
        if len(qrs_directions) >= 2:
            direction_similarity = np.dot(qrs_directions[0], qrs_directions[-1])
            # Should not be identical (allowing for some similarity due to limited rotation)
            assert abs(direction_similarity) < 0.95, "Torsades QRS directions too similar - rotation not working"
    
    @pytest.mark.unit
    def test_axis_override_functionality(self, tolerance_config):
        """Test electrical axis override produces expected QRS direction."""
        sinus_params = BEAT_MORPHOLOGIES["sinus"].copy()
        target_axis = 90.0  # Pure inferior axis
        
        time_axis, cardiac_vectors, _ = generate_single_beat_3d_vectors(
            sinus_params, "sinus", 250,
            enable_axis_override=True,
            target_axis_degrees=target_axis
        )
        
        # Extract QRS vector (peak should be around PR interval + QRS/2)
        qrs_start_time = sinus_params["pr_interval"]
        qrs_mid_time = qrs_start_time + sinus_params["qrs_duration"] / 2
        qrs_mid_idx = int(qrs_mid_time * 250)
        
        if qrs_mid_idx < len(cardiac_vectors):
            qrs_vector = cardiac_vectors[qrs_mid_idx]
            
            if np.linalg.norm(qrs_vector) > 0.01:
                # Project to Lead I and aVF to check axis
                from ecg_simulator.full_ecg.vector_projection import LEAD_VECTOR_DIRECTIONS
                
                lead_i_projection = np.dot(qrs_vector, LEAD_VECTOR_DIRECTIONS["I"])
                lead_avf_projection = np.dot(qrs_vector, LEAD_VECTOR_DIRECTIONS["aVF"])
                
                from ecg_simulator.full_ecg.vector_projection import calculate_electrical_axis
                calculated_axis, _ = calculate_electrical_axis(lead_i_projection, lead_avf_projection)
                
                # Should be close to target axis
                axis_error = abs(calculated_axis - target_axis)
                if axis_error > 180:  # Handle wraparound
                    axis_error = 360 - axis_error
                
                assert axis_error < 15.0, \
                    f"Axis override failed: target {target_axis}°, calculated {calculated_axis}°"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("beat_type,expected_has_p_wave", [
        ("sinus", True),
        ("pvc", False),
        ("pac", True),
        ("junctional_escape", False),
        ("ventricular_escape", False),
        ("svt_beat", False),
        ("vt_beat", False),
    ])
    def test_beat_type_p_wave_presence(self, beat_type, expected_has_p_wave, tolerance_config):
        """Test that different beat types have appropriate P wave presence."""
        if beat_type not in BEAT_MORPHOLOGIES:
            pytest.skip(f"Beat type {beat_type} not in BEAT_MORPHOLOGIES")
        
        beat_params = BEAT_MORPHOLOGIES[beat_type].copy()
        
        time_axis, waveform, qrs_offset = generate_single_beat_morphology(beat_params, 250)
        
        # Check P wave presence based on PR interval and P amplitude
        has_p_wave = (beat_params.get("p_amplitude", 0) != 0 and 
                     beat_params.get("pr_interval", 0) > 0)
        
        assert has_p_wave == expected_has_p_wave, \
            f"Beat type {beat_type} P wave presence mismatch: expected {expected_has_p_wave}, got {has_p_wave}"
    
    @pytest.mark.unit
    def test_beat_morphology_parameter_bounds(self, tolerance_config):
        """Test that beat morphology parameters stay within physiological bounds."""
        for beat_type, params in BEAT_MORPHOLOGIES.items():
            # QRS duration should be reasonable
            qrs_duration = params.get("qrs_duration", 0)
            assert 0 <= qrs_duration <= 0.20, \
                f"{beat_type} QRS duration {qrs_duration:.3f}s outside bounds [0, 0.20]"
            
            # PR interval should be reasonable
            pr_interval = params.get("pr_interval", 0)
            assert 0 <= pr_interval <= 0.40, \
                f"{beat_type} PR interval {pr_interval:.3f}s outside bounds [0, 0.40]"
            
            # T wave duration should be reasonable
            t_duration = params.get("t_duration", 0)
            assert 0 <= t_duration <= 0.35, \
                f"{beat_type} T duration {t_duration:.3f}s outside bounds [0, 0.35]"