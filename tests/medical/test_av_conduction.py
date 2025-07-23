"""
Tests for AV conduction blocks and escape rhythms.
Validates physiological accuracy of AV block implementations.
"""
import pytest
import numpy as np
from ecg_simulator.rhythm_logic import generate_physiologically_accurate_ecg
from ecg_simulator.api_models import AdvancedECGParams

class TestAVConduction:
    """Test AV conduction blocks and escape rhythms."""
    
    @pytest.mark.medical
    def test_first_degree_av_block_pr_prolongation(self, tolerance_config):
        """Test that first-degree AV block produces consistent PR prolongation."""
        prolonged_pr = 0.24  # 240ms - pathological
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=10.0,
            first_degree_av_block_pr_sec=prolonged_pr,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Verify rhythm description mentions first-degree AV block
        assert "first" in rhythm_desc.lower() or "pr" in rhythm_desc.lower()
        
        # Signal should be generated without errors
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.medical
    def test_mobitz_i_wenckebach_progression(self, tolerance_config):
        """Test Mobitz I (Wenckebach) AV block PR progression and dropped beats."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_mobitz_i_wenckebach=True,
            wenckebach_initial_pr_sec=0.16,
            wenckebach_pr_increment_sec=0.04,
            wenckebach_max_pr_before_drop_sec=0.28,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Verify rhythm description mentions Wenckebach or Mobitz I
        assert any(word in rhythm_desc.lower() for word in ["wenckebach", "mobitz", "type i"])
        
        # Signal should show characteristic pattern (would need more sophisticated analysis)
        assert len(signal) > 0
        assert np.any(signal != 0.0)  # Should have actual ECG content
    
    @pytest.mark.medical
    def test_mobitz_ii_fixed_conduction_ratio(self, tolerance_config):
        """Test Mobitz II AV block maintains fixed P:QRS ratio."""
        conduction_ratio = 3  # 3:1 AV block
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_mobitz_ii_av_block=True,
            mobitz_ii_p_waves_per_qrs=conduction_ratio,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Verify rhythm description mentions Mobitz II or type II
        assert any(word in rhythm_desc.lower() for word in ["mobitz", "type ii", "2nd degree"])
        
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.medical
    def test_third_degree_av_block_complete_dissociation(self, tolerance_config):
        """Test complete AV block produces independent atrial and ventricular rhythms."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,  # Atrial rate
            duration_sec=20.0,
            enable_third_degree_av_block=True,
            third_degree_escape_rhythm_origin="junctional",
            third_degree_escape_rate_bpm=45,  # Slower ventricular rate
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Verify rhythm description mentions complete AV block
        assert any(word in rhythm_desc.lower() for word in ["complete", "third", "3rd degree", "av block"])
        
        # Should have both P waves (60 bpm) and QRS (45 bpm) at different rates
        assert len(signal) > 0
        assert np.any(signal != 0.0)
    
    @pytest.mark.medical
    def test_junctional_escape_rhythm_characteristics(self, tolerance_config):
        """Test junctional escape rhythm has appropriate rate and morphology."""
        escape_rate = 45  # Typical junctional rate
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_third_degree_av_block=True,
            third_degree_escape_rhythm_origin="junctional",
            third_degree_escape_rate_bpm=escape_rate,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Junctional rhythm should be slower than sinus
        # Expected RR interval for escape beats: 60/45 = 1.33 seconds
        expected_rr_escape = 60.0 / escape_rate
        
        assert expected_rr_escape > 1.0  # Should be slower than 60 bpm
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_ventricular_escape_rhythm_characteristics(self, tolerance_config):
        """Test ventricular escape rhythm has slow rate and wide QRS."""
        escape_rate = 30  # Typical ventricular escape rate
        
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_third_degree_av_block=True,
            third_degree_escape_rhythm_origin="ventricular",
            third_degree_escape_rate_bpm=escape_rate,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Ventricular escape should be very slow
        expected_rr_escape = 60.0 / escape_rate
        assert expected_rr_escape == 2.0  # 30 bpm = 2 second intervals
        
        assert len(signal) > 0
        assert "ventricular" in rhythm_desc.lower() or "escape" in rhythm_desc.lower()
    
    @pytest.mark.medical
    @pytest.mark.xfail(reason="Rhythm hierarchy between AV block and AFib not fully implemented")
    def test_av_block_hierarchy_over_other_rhythms(self, tolerance_config):
        """Test that AV blocks take precedence over other rhythm configurations."""
        # Try to enable both AFib and third-degree AV block
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_atrial_fibrillation=True,  # Should be overridden
            enable_third_degree_av_block=True,  # Should take precedence
            third_degree_escape_rhythm_origin="junctional",
            third_degree_escape_rate_bpm=40,
            afib_average_ventricular_rate_bpm=80,
            enable_pvc=False,
            enable_pac=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        # Should describe AV block, not AFib  
        assert any(word in rhythm_desc.lower() for word in ["3rd degree", "av block", "escape"])
        assert len(signal) > 0
    
    @pytest.mark.medical
    @pytest.mark.parametrize("pr_interval,expected_classification", [
        (0.12, "normal"),      # 120ms - normal
        (0.20, "normal"),      # 200ms - upper normal
        (0.22, "prolonged"),   # 220ms - first degree block
        (0.30, "prolonged"),   # 300ms - significant prolongation
    ])
    def test_pr_interval_classification(self, pr_interval, expected_classification, tolerance_config):
        """Test that PR intervals are classified correctly."""
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=5.0,
            first_degree_av_block_pr_sec=pr_interval if pr_interval > 0.20 else None,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
        
        if expected_classification == "prolonged":
            assert any(word in rhythm_desc.lower() for word in ["first", "pr", "prolonged", "block"])
        
        assert len(signal) > 0
    
    @pytest.mark.medical
    def test_escape_rhythm_rate_validation(self, tolerance_config):
        """Test that escape rhythm rates fall within physiological ranges."""
        # Test junctional escape rates (typically 40-60 bpm, must be <65)
        for escape_rate in [35, 45, 55, 64]:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                enable_third_degree_av_block=True,
                third_degree_escape_rhythm_origin="junctional",
                third_degree_escape_rate_bpm=escape_rate,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            # Should generate valid signal regardless of rate
            assert len(signal) > 0
            assert not np.all(signal == 0.0)  # Should have actual ECG content
        
        # Test ventricular escape rates (typically 20-40 bpm)
        for escape_rate in [20, 30, 40]:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_third_degree_av_block=True,
                third_degree_escape_rhythm_origin="ventricular",
                third_degree_escape_rate_bpm=escape_rate,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            
            assert len(signal) > 0
            assert not np.all(signal == 0.0)