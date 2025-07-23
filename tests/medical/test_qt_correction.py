"""
Tests for QT interval correction and rate-dependent timing.
Validates Bazett's formula implementation and physiological accuracy.
"""
import pytest
import numpy as np
from ecg_simulator.constants import (
    calculate_qt_from_heart_rate, 
    get_rate_corrected_intervals,
    SINUS_PARAMS,
    NORMAL_QTC_MS
)

class TestQTCorrection:
    """Test QT interval correction using Bazett's formula."""
    
    @pytest.mark.medical
    def test_bazett_formula_accuracy(self, tolerance_config, medical_reference_values):
        """Test that QT calculation follows Bazett's formula: QTc = QT / √(RR)"""
        target_qtc = 400.0  # ms
        
        # Test at 60 bpm (RR = 1.0 sec)
        qt_60bpm = calculate_qt_from_heart_rate(60, target_qtc)
        expected_qt_60bpm = target_qtc / 1000.0  # Should equal target since √1 = 1
        
        assert abs(qt_60bpm - expected_qt_60bpm) < tolerance_config['timing_tolerance_sec']
        
        # Test at 120 bpm (RR = 0.5 sec)
        qt_120bpm = calculate_qt_from_heart_rate(120, target_qtc)
        expected_qt_120bpm = (target_qtc / 1000.0) * np.sqrt(0.5)  # Should be shorter
        
        assert abs(qt_120bpm - expected_qt_120bpm) < tolerance_config['timing_tolerance_sec']
        
        # Verify QT shortens with higher heart rate
        assert qt_120bpm < qt_60bpm
    
    @pytest.mark.medical
    def test_qt_physiological_bounds(self, tolerance_config):
        """Test that QT intervals stay within physiological bounds (20-60% of RR)."""
        test_rates = [30, 60, 100, 150, 200]
        
        for heart_rate in test_rates:
            rr_interval = 60.0 / heart_rate
            qt_interval = calculate_qt_from_heart_rate(heart_rate, 400.0)
            
            # QT should be 20-60% of RR interval
            min_qt = rr_interval * 0.20
            max_qt = rr_interval * 0.60
            
            assert min_qt <= qt_interval <= max_qt, \
                f"QT {qt_interval:.3f}s outside bounds [{min_qt:.3f}, {max_qt:.3f}] at {heart_rate} bpm"
    
    @pytest.mark.medical
    def test_rate_corrected_intervals_distribution(self):
        """Test that rate-corrected intervals properly distribute QT time."""
        heart_rate = 80
        target_qtc = 450.0
        
        corrected_params = get_rate_corrected_intervals(heart_rate, SINUS_PARAMS, target_qtc)
        
        total_qt = (corrected_params['qrs_duration'] + 
                   corrected_params['st_duration'] + 
                   corrected_params['t_duration'])
        
        expected_qt = calculate_qt_from_heart_rate(heart_rate, target_qtc)
        
        # Allow small tolerance for rounding
        assert abs(total_qt - expected_qt) < 0.01, \
            f"Total QT {total_qt:.3f}s doesn't match expected {expected_qt:.3f}s"
    
    @pytest.mark.medical
    def test_extreme_heart_rates(self):
        """Test QT calculation at extreme but physiological heart rates."""
        # Very slow heart rate
        qt_slow = calculate_qt_from_heart_rate(20, 400.0)
        rr_slow = 60.0 / 20  # 3 seconds
        assert qt_slow <= rr_slow * 0.60  # Should hit upper bound
        
        # Very fast heart rate  
        qt_fast = calculate_qt_from_heart_rate(250, 400.0)
        rr_fast = 60.0 / 250  # 0.24 seconds
        assert qt_fast >= rr_fast * 0.20  # Should hit lower bound
        
        # Verify QT decreases with rate
        assert qt_fast < qt_slow
    
    @pytest.mark.medical
    def test_qtc_target_preservation(self):
        """Test that different heart rates produce same corrected QTc."""
        target_qtc = 420.0
        test_rates = [50, 70, 90, 110, 130]
        calculated_qtcs = []
        
        for rate in test_rates:
            qt_interval = calculate_qt_from_heart_rate(rate, target_qtc)
            rr_interval = 60.0 / rate
            
            # Calculate QTc using Bazett's formula
            qtc_calculated = (qt_interval * 1000) / np.sqrt(rr_interval)
            calculated_qtcs.append(qtc_calculated)
        
        # All calculated QTcs should be close to target
        for qtc in calculated_qtcs:
            assert abs(qtc - target_qtc) < 15.0, \
                f"Calculated QTc {qtc:.1f}ms deviates too much from target {target_qtc}ms"
    
    @pytest.mark.medical
    @pytest.mark.xfail(reason="QRS duration rate dependence not yet implemented - returns constant 0.1s")
    def test_qrs_duration_rate_dependence(self):
        """Test that QRS duration appropriately changes with heart rate."""
        base_params = SINUS_PARAMS.copy()
        
        # Slow rate - QRS should narrow slightly
        slow_params = get_rate_corrected_intervals(40, base_params, 400.0)
        assert slow_params['qrs_duration'] <= base_params['qrs_duration'] * 0.95
        
        # Fast rate - QRS should widen slightly
        fast_params = get_rate_corrected_intervals(160, base_params, 400.0)
        assert fast_params['qrs_duration'] >= base_params['qrs_duration'] * 1.05
        
        # Very fast rate - QRS widening is more pronounced
        very_fast_params = get_rate_corrected_intervals(200, base_params, 400.0)
        assert very_fast_params['qrs_duration'] > fast_params['qrs_duration']
    
    @pytest.mark.medical
    def test_invalid_heart_rate_handling(self):
        """Test handling of invalid heart rates."""
        # Zero heart rate
        qt_zero = calculate_qt_from_heart_rate(0, 400.0)
        assert qt_zero == NORMAL_QTC_MS / 1000.0  # Should return fallback
        
        # Negative heart rate
        qt_negative = calculate_qt_from_heart_rate(-50, 400.0)
        assert qt_negative == NORMAL_QTC_MS / 1000.0  # Should return fallback
    
    @pytest.mark.medical
    @pytest.mark.parametrize("qtc_target,expected_range", [
        (350, (0.25, 0.45)),  # Short QTc
        (400, (0.30, 0.50)),  # Normal QTc
        (450, (0.35, 0.55)),  # Borderline long QTc
        (500, (0.40, 0.60)),  # Long QTc
    ])
    def test_qtc_ranges_clinical_validity(self, qtc_target, expected_range):
        """Test that different QTc targets produce clinically reasonable QT intervals."""
        heart_rate = 75  # Normal resting rate
        qt_interval = calculate_qt_from_heart_rate(heart_rate, qtc_target)
        
        assert expected_range[0] <= qt_interval <= expected_range[1], \
            f"QT {qt_interval:.3f}s outside expected range {expected_range} for QTc {qtc_target}ms"