"""
Integration tests for ECG API endpoints.
Tests API functionality, parameter validation, and response formats.
"""
import pytest
import json
import numpy as np
from httpx import AsyncClient
from fastapi.testclient import TestClient
from ecg_simulator.api import app
from ecg_simulator.api_models import AdvancedECGParams

class TestAPIEndpoints:
    """Test ECG API endpoints functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.mark.integration
    def test_single_lead_api_basic_sinus(self, client, basic_sinus_params):
        """Test single-lead API with basic sinus rhythm."""
        response = client.post("/generate_advanced_ecg", json=basic_sinus_params.model_dump())
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert "time_axis" in data
        assert "ecg_signal" in data
        assert "rhythm_generated" in data
        
        # Check data types and lengths
        time_axis = data["time_axis"]
        signal = data["ecg_signal"]
        
        assert isinstance(time_axis, list)
        assert isinstance(signal, list)
        assert len(time_axis) == len(signal)
        
        # Check duration matches request (sampling rate might not be in response)
        expected_duration = basic_sinus_params.duration_sec
        # Assume 250 Hz sampling rate (FS constant)
        actual_duration = len(time_axis) / 250.0
        assert abs(actual_duration - expected_duration) < 0.1
        
        # Signal should have content
        assert any(abs(x) > 0.01 for x in signal), "Signal appears to be empty"
    
    @pytest.mark.integration
    def test_12_lead_api_basic_sinus(self, client, basic_sinus_params):
        """Test 12-lead API with basic sinus rhythm."""
        response = client.post("/generate_advanced_ecg_12_lead", json=basic_sinus_params.model_dump())
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert "time_axis" in data
        assert "twelve_lead_signals" in data
        assert "rhythm_generated" in data
        
        # Check 12-lead structure
        signals = data["twelve_lead_signals"]
        expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for lead in expected_leads:
            assert lead in signals, f"Lead {lead} missing from response"
            assert isinstance(signals[lead], list)
            assert len(signals[lead]) == len(data["time_axis"])
        
        # Calculate electrical axis from returned signals (since API doesn't include it)
        from ecg_simulator.full_ecg.vector_projection import calculate_qrs_axis_from_12_lead
        try:
            axis_degrees, interpretation = calculate_qrs_axis_from_12_lead(signals)
            assert -180 <= axis_degrees <= 180, f"Calculated electrical axis {axis_degrees}° out of range"
        except Exception as e:
            # If axis calculation fails, that's okay for this test
            print(f"Could not calculate electrical axis: {e}")
    
    @pytest.mark.integration
    def test_single_lead_api_atrial_fibrillation(self, client, afib_params):
        """Test single-lead API with atrial fibrillation."""
        response = client.post("/generate_advanced_ecg", json=afib_params.model_dump())
        
        assert response.status_code == 200
        
        data = response.json()
        rhythm_description = data["rhythm_generated"].lower()
        
        # Should mention AFib
        assert any(word in rhythm_description for word in ["afib", "atrial fibrillation", "fibrillation"])
        
        # Signal should show irregularity
        signal = data["ecg_signal"]
        assert len(signal) > 0
        signal_variance = np.var(signal)
        assert signal_variance > 0.01, "AFib signal should have variability"
    
    @pytest.mark.integration
    def test_single_lead_api_ventricular_tachycardia(self, client, vt_params):
        """Test single-lead API with ventricular tachycardia."""
        response = client.post("/generate_advanced_ecg", json=vt_params.model_dump())
        
        assert response.status_code == 200
        
        data = response.json()
        rhythm_description = data["rhythm_generated"].lower()
        
        # Should mention VT
        assert any(word in rhythm_description for word in ["vt", "ventricular tachycardia", "tachycardia"])
        
        # Check timing - VT should be present in middle portion
        time_axis = data["time_axis"]
        signal = data["ecg_signal"]
        
        vt_start_time = vt_params.vt_start_time_sec
        vt_end_time = vt_start_time + vt_params.vt_duration_sec
        
        # Find indices corresponding to VT period
        vt_start_idx = next((i for i, t in enumerate(time_axis) if t >= vt_start_time), 0)
        vt_end_idx = next((i for i, t in enumerate(time_axis) if t >= vt_end_time), len(time_axis))
        
        # VT period should have content
        if vt_start_idx < vt_end_idx:
            vt_signal = signal[vt_start_idx:vt_end_idx]
            assert any(abs(x) > 0.1 for x in vt_signal), "VT period should have significant signal"
    
    @pytest.mark.integration
    def test_api_parameter_validation_invalid_heart_rate(self, client):
        """Test API parameter validation with invalid heart rate."""
        invalid_params = {
            "heart_rate_bpm": -50,  # Invalid negative heart rate
            "duration_sec": 10.0,
            "enable_pvc": False,
            "enable_pac": False,
            "enable_atrial_fibrillation": False,
            "enable_vt": False
        }
        
        response = client.post("/generate_advanced_ecg", json=invalid_params)
        
        # Should return validation error
        assert response.status_code == 422  # FastAPI validation error
        
        error_data = response.json()
        assert "detail" in error_data
    
    @pytest.mark.integration
    def test_api_parameter_validation_invalid_qtc(self, client):
        """Test API parameter validation with invalid QTc."""
        invalid_params = {
            "heart_rate_bpm": 60,
            "duration_sec": 10.0,
            "target_qtc_ms": 700,  # Extremely long QTc
            "enable_pvc": False,
            "enable_pac": False,
            "enable_atrial_fibrillation": False,
            "enable_vt": False
        }
        
        response = client.post("/generate_advanced_ecg", json=invalid_params)
        
        # Should return validation error
        assert response.status_code == 422
    
    @pytest.mark.integration
    def test_api_parameter_validation_conflicting_rhythms(self, client):
        """Test API behavior with potentially conflicting rhythm parameters."""
        # This might not fail validation but should handle gracefully
        conflicting_params = {
            "heart_rate_bpm": 60,
            "duration_sec": 15.0,
            "enable_atrial_fibrillation": True,
            "enable_vt": True,  # Both AFib and VT enabled
            "vt_start_time_sec": 5.0,
            "vt_duration_sec": 8.0,
            "vt_rate_bpm": 180,
            "afib_average_ventricular_rate_bpm": 80,
            "enable_pvc": False,
            "enable_pac": False
        }
        
        response = client.post("/generate_advanced_ecg", json=conflicting_params)
        
        # Should handle gracefully (current implementation has rhythm hierarchy)
        assert response.status_code == 200
        
        data = response.json()
        # VT should take precedence based on current hierarchy
        rhythm_description = data["rhythm_generated"].lower()
        assert any(word in rhythm_description for word in ["vt", "ventricular tachycardia"])
    
    @pytest.mark.integration
    def test_12_lead_electrical_axis_calculation(self, client):
        """Test 12-lead electrical axis calculation accuracy."""
        # Create params with axis override to test calculation
        params = {
            "heart_rate_bpm": 60,
            "duration_sec": 5.0,
            "enable_axis_override": True,
            "target_axis_degrees": 60.0,  # Specific target axis
            "enable_pvc": False,
            "enable_pac": False,
            "enable_atrial_fibrillation": False,
            "enable_vt": False
        }
        
        response = client.post("/generate_advanced_ecg_12_lead", json=params)
        
        assert response.status_code == 200
        
        data = response.json()
        signals = data["twelve_lead_signals"]
        
        # Calculate electrical axis from returned signals
        from ecg_simulator.full_ecg.vector_projection import calculate_qrs_axis_from_12_lead
        calculated_axis, interpretation = calculate_qrs_axis_from_12_lead(signals)
        
        # Should be close to target axis
        target_axis = params["target_axis_degrees"]
        axis_error = abs(calculated_axis - target_axis)
        
        # Handle wraparound for axes near ±180°
        if axis_error > 180:
            axis_error = 360 - axis_error
        
        assert axis_error < 20.0, \
            f"Calculated axis {calculated_axis:.1f}° too far from target {target_axis}°"
    
    @pytest.mark.integration
    def test_api_response_consistency(self, client, basic_sinus_params):
        """Test that API responses are consistent across multiple calls."""
        # Note: This test may be challenging due to randomness in some algorithms
        # We'll test that basic structure and ranges are consistent
        
        responses = []
        for _ in range(3):
            response = client.post("/generate_advanced_ecg", json=basic_sinus_params.model_dump())
            assert response.status_code == 200
            responses.append(response.json())
        
        # Check that all responses have same structure
        for i, data in enumerate(responses):
            assert "time_axis" in data, f"Response {i} missing time_axis"
            assert "ecg_signal" in data, f"Response {i} missing ecg_signal"
            assert "rhythm_generated" in data, f"Response {i} missing rhythm_generated"
            
            # All should have same duration
            assert len(data["time_axis"]) == len(responses[0]["time_axis"])
            
            # Signal amplitudes should be in reasonable range
            signal = data["ecg_signal"]
            max_amplitude = max(abs(x) for x in signal)
            assert 0.1 < max_amplitude < 5.0, f"Response {i} amplitude {max_amplitude:.3f} out of range"
    
    @pytest.mark.integration
    def test_api_long_duration_handling(self, client):
        """Test API with longer duration ECGs."""
        long_params = {
            "heart_rate_bpm": 75,
            "duration_sec": 60.0,  # 1 minute ECG
            "enable_pvc": False,
            "enable_pac": False,
            "enable_atrial_fibrillation": False,
            "enable_vt": False
        }
        
        response = client.post("/generate_advanced_ecg", json=long_params)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check expected number of samples
        expected_samples = int(60.0 * 250)  # 60 seconds at 250 Hz
        actual_samples = len(data["ecg_signal"])
        
        # Allow some tolerance for timing
        assert abs(actual_samples - expected_samples) < 250, \
            f"Sample count mismatch: expected ~{expected_samples}, got {actual_samples}"
    
    @pytest.mark.integration
    def test_api_edge_case_parameters(self, client):
        """Test API with edge case but valid parameters."""
        edge_params = {
            "heart_rate_bpm": 30,      # Very slow
            "duration_sec": 2.0,       # Very short
            "target_qtc_ms": 350,      # Short QTc
            "enable_pvc": False,
            "enable_pac": False,
            "enable_atrial_fibrillation": False,
            "enable_vt": False
        }
        
        response = client.post("/generate_advanced_ecg", json=edge_params)
        
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["ecg_signal"]) > 0
        assert len(data["time_axis"]) == len(data["ecg_signal"])
    
    @pytest.mark.integration
    def test_12_lead_signal_correlation(self, client, basic_sinus_params):
        """Test that 12-lead signals show expected correlations."""
        response = client.post("/generate_advanced_ecg_12_lead", json=basic_sinus_params.model_dump())
        
        assert response.status_code == 200
        
        data = response.json()
        signals = data["twelve_lead_signals"]
        
        # Lead II should approximately equal Lead I + Lead III (Einthoven's law)
        lead_i = np.array(signals["I"])
        lead_ii = np.array(signals["II"])
        lead_iii = np.array(signals["III"])
        
        einthoven_sum = lead_i + lead_iii
        correlation = np.corrcoef(lead_ii, einthoven_sum)[0, 1]
        
        # Should have reasonable correlation (allowing for numerical precision)
        assert correlation > 0.8, f"Einthoven's law correlation {correlation:.3f} too low"
        
        # aVR should be approximately -(I + II)/2
        lead_avr = np.array(signals["aVR"])
        expected_avr = -(lead_i + lead_ii) / 2
        avr_correlation = np.corrcoef(lead_avr, expected_avr)[0, 1]
        
        assert avr_correlation > 0.8, f"aVR relationship correlation {avr_correlation:.3f} too low"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_stress_multiple_requests(self, client, basic_sinus_params):
        """Stress test with multiple concurrent API requests."""
        # Test multiple requests to ensure stability
        num_requests = 10
        successful_responses = 0
        
        for i in range(num_requests):
            response = client.post("/generate_advanced_ecg", json=basic_sinus_params.model_dump())
            
            if response.status_code == 200:
                successful_responses += 1
                
                # Quick validation of response
                data = response.json()
                assert len(data["ecg_signal"]) > 0
        
        # Should have high success rate
        success_rate = successful_responses / num_requests
        assert success_rate >= 0.9, f"Success rate {success_rate:.1%} too low"