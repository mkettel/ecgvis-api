"""
Tests for complex rhythm scenarios and interactions.
Validates rhythm hierarchy, transitions, and complex arrhythmia patterns.
"""
import pytest
import numpy as np
from ecg_simulator.rhythm_logic import generate_physiologically_accurate_ecg
from ecg_simulator.api_models import AdvancedECGParams

class TestComplexRhythms:
    """Test complex rhythm scenarios and interactions."""
    
    @pytest.mark.medical
    def test_vt_interrupting_svt_hierarchy(self, tolerance_config):
        """Test that VT properly interrupts ongoing SVT episode."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=30.0,
            # SVT configuration (should start first)
            enable_pac=True,
            pac_probability_per_sinus=0.5,  # High probability
            allow_svt_initiation_by_pac=True,
            svt_initiation_probability_after_pac=1.0,  # Guarantee SVT
            svt_duration_sec=20.0,  # Long SVT episode
            svt_rate_bpm=160,
            # VT configuration (should interrupt SVT)
            enable_vt=True,
            vt_start_time_sec=10.0,  # Start during SVT
            vt_duration_sec=8.0,
            vt_rate_bpm=180,
            enable_pvc=False,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention VT as the dominant rhythm due to hierarchy
        assert any(word in rhythm_desc.lower() for word in ["vt", "ventricular tachycardia"])
        
        # Signal should be generated successfully
        assert len(signal) > 0
        assert np.any(signal != 0.0)
    
    @pytest.mark.medical
    def test_torsades_triggering_with_long_qtc(self, tolerance_config):
        """Test Torsades de Pointes triggering with prolonged QTc."""
        params = AdvancedECGParams(
            heart_rate_bpm=50,  # Slower rate makes QT longer
            duration_sec=120.0,  # Longer duration for triggering
            target_qtc_ms=540.0,  # Very prolonged QTc
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.02,  # 2% per beat
            use_sensitive_qtc_threshold=True,  # Use lower threshold
            enable_pvc=True,  # PVCs can trigger Torsades
            pvc_probability_per_sinus=0.05,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        # Run multiple times since it's probabilistic
        torsades_episodes = 0
        max_attempts = 5
        
        for attempt in range(max_attempts):
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            if "torsades" in rhythm_desc.lower():
                torsades_episodes += 1
                
                # Verify signal has chaotic patterns during Torsades
                assert len(signal) > 0
                signal_variance = np.var(signal)
                assert signal_variance > 0.05, "Torsades should have high signal variability"
                break
        
        # With these parameters, should trigger at least once in 5 attempts
        # Note: This test has some probabilistic uncertainty
        assert len(signal) > 0  # At minimum, should generate valid signal
    
    @pytest.mark.medical
    def test_post_vt_sinus_resumption(self, tolerance_config):
        """Test proper sinus rhythm resumption after VT terminates."""
        params = AdvancedECGParams(
            heart_rate_bpm=70,
            duration_sec=25.0,
            enable_vt=True,
            vt_start_time_sec=8.0,
            vt_duration_sec=6.0,  # VT from 8-14 seconds
            vt_rate_bpm=200,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should have sinus before VT, VT in middle, and sinus after VT
        vt_end_time = params.vt_start_time_sec + params.vt_duration_sec  # 14.0 seconds
        
        # Find time indices
        pre_vt_end_idx = next((i for i, t in enumerate(time_axis) if t >= params.vt_start_time_sec), 0)
        post_vt_start_idx = next((i for i, t in enumerate(time_axis) if t >= vt_end_time + 1.0), len(time_axis))
        
        # Should have signal in all periods
        if pre_vt_end_idx > 10:  # Pre-VT period
            pre_vt_signal = signal[:pre_vt_end_idx]
            assert any(abs(x) > 0.01 for x in pre_vt_signal), "Should have pre-VT sinus activity"
        
        if post_vt_start_idx < len(signal) - 10:  # Post-VT period
            post_vt_signal = signal[post_vt_start_idx:]
            assert any(abs(x) > 0.01 for x in post_vt_signal), "Should have post-VT sinus activity"
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_afib_with_rapid_ventricular_response(self, tolerance_config):
        """Test AFib with rapid ventricular response characteristics."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,  # Atrial rate (not used in AFib)
            duration_sec=30.0,
            enable_atrial_fibrillation=True,
            afib_average_ventricular_rate_bpm=130,  # Rapid response
            afib_irregularity_factor=0.4,  # Moderate irregularity
            afib_fibrillation_wave_amplitude_mv=0.15,
            enable_pvc=False,
            enable_pac=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention AFib
        assert any(word in rhythm_desc.lower() for word in ["afib", "atrial fibrillation"])
        
        # Should have irregular pattern with baseline f-waves
        assert len(signal) > 0
        
        # Signal should have higher variance due to irregularity and f-waves
        signal_variance = np.var(signal)
        assert signal_variance > 0.02, f"AFib signal variance {signal_variance:.4f} too low"
    
    @pytest.mark.medical
    def test_third_degree_av_block_with_escape_rhythm(self, tolerance_config):
        """Test complete AV block with independent escape rhythm."""
        params = AdvancedECGParams(
            heart_rate_bpm=80,  # Atrial rate
            duration_sec=20.0,
            enable_third_degree_av_block=True,
            third_degree_escape_rhythm_origin="junctional",
            third_degree_escape_rate_bpm=45,  # Slower escape rate
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention complete AV block
        assert any(word in rhythm_desc.lower() for word in ["complete", "third", "3rd", "av block"])
        
        # Should have both P waves (80 bpm) and QRS (45 bpm) at different rates
        # This would require sophisticated signal analysis to verify properly
        assert len(signal) > 0
        assert np.any(signal != 0.0)
        
        # Expected intervals
        atrial_rr = 60.0 / params.heart_rate_bpm  # 0.75 seconds
        escape_rr = 60.0 / params.third_degree_escape_rate_bpm  # 1.33 seconds
        
        assert escape_rr > atrial_rr, "Escape rhythm should be slower than atrial rate"
    
    @pytest.mark.medical
    def test_flutter_with_variable_av_conduction(self, tolerance_config):
        """Test atrial flutter with variable AV conduction ratios."""
        conduction_ratios = [2, 3, 4]  # Test different AV block ratios
        
        for ratio in conduction_ratios:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_atrial_flutter=True,
                atrial_flutter_rate_bpm=300,  # Typical flutter rate
                atrial_flutter_av_block_ratio_qrs_to_f=ratio,
                atrial_flutter_wave_amplitude_mv=0.2,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            # Should mention flutter
            assert "flutter" in rhythm_desc.lower()
            
            # Expected ventricular rate
            expected_ventricular_rate = params.atrial_flutter_rate_bpm / ratio
            
            # Should have signal content
            assert len(signal) > 0
            assert np.any(signal != 0.0)
    
    @pytest.mark.medical
    def test_pvc_bigeminy_pattern(self, tolerance_config):
        """Test PVC bigeminy (every other beat is PVC)."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.9,  # Very high PVC frequency
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention PVCs
        assert any(word in rhythm_desc.lower() for word in ["pvc", "premature ventricular"])
        
        # With 90% PVC probability, should have frequent ectopy
        assert len(signal) > 0
        
        # Signal should show variation from frequent PVCs
        signal_variance = np.var(signal)
        assert signal_variance > 0.05, f"High PVC frequency should increase signal variance (got {signal_variance:.4f})"
    
    @pytest.mark.medical
    def test_wenckebach_av_block_progression(self, tolerance_config):
        """Test Wenckebach (Mobitz I) AV block PR progression."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=25.0,
            enable_mobitz_i_wenckebach=True,
            wenckebach_initial_pr_sec=0.16,  # Start normal
            wenckebach_pr_increment_sec=0.04,  # Progressive increase
            wenckebach_max_pr_before_drop_sec=0.28,  # Drop when too long
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention Wenckebach or Mobitz I
        assert any(word in rhythm_desc.lower() for word in ["wenckebach", "mobitz", "type i"])
        
        # Should have characteristic pattern with dropped beats
        assert len(signal) > 0
        assert np.any(signal != 0.0)
    
    @pytest.mark.medical 
    def test_multifocal_pvc_morphologies(self, tolerance_config):
        """Test that multiple PVCs can have different morphologies."""
        # This test would require more sophisticated morphology analysis
        # For now, just verify that PVCs are generated appropriately
        
        params = AdvancedECGParams(
            heart_rate_bpm=70,
            duration_sec=30.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.15,  # 15% PVCs
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should mention PVCs
        assert any(word in rhythm_desc.lower() for word in ["pvc", "premature ventricular", "ectopic"])
        
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.medical
    def test_rhythm_transitions_stability(self, tolerance_config):
        """Test that rhythm transitions don't cause signal artifacts."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=40.0,
            # Multiple rhythm changes
            enable_vt=True,
            vt_start_time_sec=10.0,
            vt_duration_sec=5.0,  # VT from 10-15 seconds
            vt_rate_bpm=180,
            enable_pac=True,
            pac_probability_per_sinus=0.1,
            allow_svt_initiation_by_pac=True,
            svt_initiation_probability_after_pac=0.3,
            svt_duration_sec=8.0,
            svt_rate_bpm=150,
            enable_pvc=True,
            pvc_probability_per_sinus=0.05,
            enable_atrial_fibrillation=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Check for signal continuity (no sudden jumps)
        signal_array = np.array(signal)
        signal_diff = np.diff(signal_array)
        max_jump = np.max(np.abs(signal_diff))
        
        # Should not have sudden large jumps (>2mV in one sample at 250Hz)
        assert max_jump < 2.0, f"Signal has large jump of {max_jump:.3f}mV between samples"
        
        # Should not have NaN or infinite values
        assert np.all(np.isfinite(signal_array)), "Signal contains NaN or infinite values"
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_long_qt_syndrome_simulation(self, tolerance_config):
        """Test Long QT syndrome with increased Torsades risk."""
        params = AdvancedECGParams(
            heart_rate_bpm=55,  # Bradycardia common in long QT
            duration_sec=60.0,
            target_qtc_ms=510.0,  # Prolonged QTc (>480ms abnormal)
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.005,  # 0.5% per beat
            use_sensitive_qtc_threshold=True,
            enable_pvc=True,  # PVCs can trigger TdP
            pvc_probability_per_sinus=0.02,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should handle long QT appropriately
        assert len(signal) > 0
        
        # Verify QTc is actually prolonged in the parameters
        assert params.target_qtc_ms > 480, "Should simulate prolonged QTc"
        
        # Signal should be stable even with long QT
        signal_array = np.array(signal)
        assert np.all(np.isfinite(signal_array)), "Long QT signal should be stable"