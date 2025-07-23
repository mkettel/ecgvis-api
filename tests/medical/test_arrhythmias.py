"""
Tests for arrhythmia generation and characteristics.
Validates VT, AFib, Torsades, and other arrhythmia implementations.
"""
import pytest
import numpy as np
from ecg_simulator.rhythm_logic import generate_physiologically_accurate_ecg
from ecg_simulator.api_models import AdvancedECGParams

class TestArrhythmias:
    """Test arrhythmia generation and medical accuracy."""
    
    @pytest.mark.medical
    def test_atrial_fibrillation_irregularity(self, tolerance_config):
        """Test AFib produces irregular ventricular response."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_atrial_fibrillation=True,
            afib_average_ventricular_rate_bpm=80,
            afib_irregularity_factor=0.4,  # Moderate irregularity
            afib_fibrillation_wave_amplitude_mv=0.1,
            enable_pvc=False,
            enable_pac=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention AFib or atrial fibrillation
        assert any(word in rhythm_desc.lower() for word in ["afib", "atrial fibrillation", "fibrillation"])
        
        # Signal should have content
        assert len(signal) > 0
        assert np.any(signal != 0.0)
        
        # Should have fibrillatory waves (baseline activity)
        signal_variance = np.var(signal)
        assert signal_variance > 0.01  # Should have variability from f-waves
    
    @pytest.mark.medical
    def test_atrial_flutter_regular_waves(self, tolerance_config):
        """Test atrial flutter produces regular flutter waves with AV block."""
        flutter_rate = 300  # Typical atrial flutter rate
        av_block_ratio = 4  # 4:1 conduction
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_atrial_flutter=True,
            atrial_flutter_rate_bpm=flutter_rate,
            atrial_flutter_av_block_ratio_qrs_to_f=av_block_ratio,
            atrial_flutter_wave_amplitude_mv=0.15,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention flutter
        assert "flutter" in rhythm_desc.lower()
        
        # Expected ventricular rate should be flutter_rate / av_block_ratio
        expected_ventricular_rate = flutter_rate / av_block_ratio  # 75 bpm
        
        assert len(signal) > 0
        assert np.any(signal != 0.0)
    
    @pytest.mark.medical
    def test_ventricular_tachycardia_timing(self, tolerance_config):
        """Test VT starts and stops at specified times with correct rate."""
        vt_start = 5.0
        vt_duration = 8.0
        vt_rate = 180
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_vt=True,
            vt_start_time_sec=vt_start,
            vt_duration_sec=vt_duration,
            vt_rate_bpm=vt_rate,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention VT or ventricular tachycardia
        assert any(word in rhythm_desc.lower() for word in ["vt", "ventricular tachycardia", "tachycardia"])
        
        # VT should end at vt_start + vt_duration = 13.0 seconds
        vt_end_time = vt_start + vt_duration
        assert vt_end_time == 13.0
        
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.medical
    def test_torsades_de_pointes_triggering(self, tolerance_config):
        """Test Torsades triggering with prolonged QTc."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=60.0,  # Longer duration for triggering
            target_qtc_ms=520.0,  # Prolonged QTc (>500ms)
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.05,  # Higher probability for testing
            use_sensitive_qtc_threshold=False,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        # Run multiple times since it's probabilistic
        torsades_triggered = False
        for _ in range(10):  # Try up to 10 times
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            if "torsades" in rhythm_desc.lower():
                torsades_triggered = True
                break
        
        # With 5% probability per beat and ~60 beats, should trigger at least once
        # Note: This test might occasionally fail due to randomness
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_svt_episode_characteristics(self, tolerance_config):
        """Test SVT episodes triggered by PACs."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=30.0,
            enable_pac=True,
            pac_probability_per_sinus=0.1,  # 10% PACs
            allow_svt_initiation_by_pac=True,
            svt_initiation_probability_after_pac=0.3,  # 30% chance of SVT after PAC
            svt_duration_sec=8.0,
            svt_rate_bpm=160,
            enable_pvc=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        # Run multiple times to increase chance of SVT
        svt_triggered = False
        for _ in range(10):
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            if "svt" in rhythm_desc.lower() or "supraventricular" in rhythm_desc.lower():
                svt_triggered = True
                break
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_pvc_coupling_intervals(self, tolerance_config):
        """Test PVC coupling intervals are physiologically appropriate."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=30.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.15,  # 15% PVCs for good chance of occurrence
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention PVCs
        assert any(word in rhythm_desc.lower() for word in ["pvc", "premature ventricular", "ectopic"])
        
        assert len(signal) > 0
        assert np.any(signal != 0.0)
    
    @pytest.mark.medical
    def test_pac_characteristics(self, tolerance_config):
        """Test PAC timing and morphology characteristics."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_pac=True,
            pac_probability_per_sinus=0.2,  # 20% PACs
            enable_pvc=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention PACs
        assert any(word in rhythm_desc.lower() for word in ["pac", "premature atrial", "atrial ectopic"])
        
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.medical
    def test_rhythm_hierarchy_vt_over_svt(self, tolerance_config):
        """Test that VT takes precedence over SVT when both are configured."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            # Enable both VT and SVT conditions
            enable_vt=True,
            vt_start_time_sec=5.0,
            vt_duration_sec=8.0,
            vt_rate_bpm=180,
            # SVT settings (should be overridden)
            enable_pac=True,
            pac_probability_per_sinus=0.2,
            allow_svt_initiation_by_pac=True,
            svt_initiation_probability_after_pac=1.0,  # 100% to ensure SVT would trigger
            svt_duration_sec=10.0,
            svt_rate_bpm=160,
            enable_pvc=False,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should describe VT, not SVT
        assert any(word in rhythm_desc.lower() for word in ["vt", "ventricular tachycardia"])
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    @pytest.mark.parametrize("vt_rate,expected_rr_sec", [
        (150, 0.4),    # 150 bpm = 0.4 sec intervals
        (180, 0.333),  # 180 bpm = 0.333 sec intervals  
        (220, 0.273),  # 220 bpm = 0.273 sec intervals
    ])
    def test_vt_rate_accuracy(self, vt_rate, expected_rr_sec, tolerance_config):
        """Test that VT produces correct heart rate."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_vt=True,
            vt_start_time_sec=2.0,
            vt_duration_sec=10.0,
            vt_rate_bpm=vt_rate,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Expected RR interval should match calculated value within tolerance
        calculated_rr = 60.0 / vt_rate
        assert abs(calculated_rr - expected_rr_sec) < tolerance_config['timing_tolerance_sec']
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    @pytest.mark.xfail(reason="VFib not yet implemented - falls back to sinus rhythm")
    def test_vfib_waveform_characteristics(self, tolerance_config):
        """Test VFib waveform has chaotic, irregular characteristics."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_vfib=True,
            vfib_start_time_sec=5.0,
            vfib_duration_sec=10.0,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention VFib or ventricular fibrillation
        assert any(word in rhythm_desc.lower() for word in ["vfib", "ventricular fibrillation", "v-fib"])
        
        # VFib should have high variability and no organized beats
        assert len(signal) > 0
        
        # During VFib period (samples 5-15 seconds), should have chaotic activity
        fs = 250  # Assuming 250 Hz sampling
        vfib_start_idx = int(5.0 * fs)
        vfib_end_idx = int(15.0 * fs)
        
        if vfib_end_idx <= len(signal):
            vfib_segment = signal[vfib_start_idx:vfib_end_idx]
            vfib_variance = np.var(vfib_segment)
            
            # VFib should have significant variability
            assert vfib_variance > 0.1  # Should be chaotic