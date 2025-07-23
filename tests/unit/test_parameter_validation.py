"""
Tests for parameter validation and edge cases.
Validates API parameter constraints and error handling.
"""
import pytest
from pydantic import ValidationError
from ecg_simulator.api_models import AdvancedECGParams

class TestParameterValidation:
    """Test parameter validation and constraints."""
    
    @pytest.mark.unit
    def test_valid_basic_parameters(self):
        """Test that valid basic parameters are accepted."""
        valid_params = AdvancedECGParams(
            heart_rate_bpm=75,
            duration_sec=10.0,
            target_qtc_ms=400.0,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        assert valid_params.heart_rate_bpm == 75
        assert valid_params.duration_sec == 10.0
        assert valid_params.target_qtc_ms == 400.0
    
    @pytest.mark.unit
    def test_heart_rate_validation(self):
        """Test heart rate parameter validation."""
        
        # Valid heart rates
        valid_rates = [30, 60, 100, 150, 220]
        for rate in valid_rates:
            params = AdvancedECGParams(
                heart_rate_bpm=rate,
                duration_sec=5.0,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.heart_rate_bpm == rate
        
        # Invalid heart rates (should raise ValidationError)
        invalid_rates = [-10, 0, -1.5]  # Only negative and zero are invalid (gt=0)
        for rate in invalid_rates:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=rate,
                    duration_sec=5.0,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_duration_validation(self):
        """Test duration parameter validation."""
        
        # Valid durations
        valid_durations = [1.0, 5.0, 30.0, 120.0, 300.0]
        for duration in valid_durations:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=duration,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.duration_sec == duration
        
        # Invalid durations (only negative and zero are invalid, gt=0)
        invalid_durations = [-1.0, 0, -0.5]
        for duration in invalid_durations:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=duration,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_qtc_validation(self):
        """Test QTc parameter validation."""
        
        # Valid QTc values
        valid_qtcs = [300, 350, 400, 450, 500, 550, 600]
        for qtc in valid_qtcs:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                target_qtc_ms=qtc,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.target_qtc_ms == qtc
        
        # Invalid QTc values (outside 300-600 range)
        invalid_qtcs = [200, 250, 650, 700, 1000]
        for qtc in invalid_qtcs:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=10.0,
                    target_qtc_ms=qtc,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_probability_validation(self):
        """Test probability parameter validation."""
        
        # Valid probabilities
        valid_probabilities = [0.0, 0.1, 0.5, 0.9, 1.0]
        for prob in valid_probabilities:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                enable_pvc=True,
                pvc_probability_per_sinus=prob,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.pvc_probability_per_sinus == prob
        
        # Invalid probabilities
        invalid_probabilities = [-0.1, -1.0, 1.1, 2.0, 10.0]
        for prob in invalid_probabilities:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=10.0,
                    enable_pvc=True,
                    pvc_probability_per_sinus=prob,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_av_block_parameter_validation(self):
        """Test AV block parameter validation."""
        
        # Valid first-degree AV block PR intervals
        valid_pr_intervals = [0.21, 0.25, 0.30, 0.35, 0.40]
        for pr in valid_pr_intervals:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                first_degree_av_block_pr_sec=pr,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.first_degree_av_block_pr_sec == pr
        
        # Invalid PR intervals (outside 0.201-0.60 range)
        invalid_pr_intervals = [0.05, 0.10, 0.20, 0.70, 1.0]
        for pr in invalid_pr_intervals:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=10.0,
                    first_degree_av_block_pr_sec=pr,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
        
        # Valid Mobitz II ratios
        valid_ratios = [2, 3, 4, 5, 6]
        for ratio in valid_ratios:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                enable_mobitz_ii_av_block=True,
                mobitz_ii_p_waves_per_qrs=ratio,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.mobitz_ii_p_waves_per_qrs == ratio
        
        # Invalid ratios (must be ge=2)
        invalid_ratios = [0, 1, -1]
        for ratio in invalid_ratios:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=10.0,
                    enable_mobitz_ii_av_block=True,
                    mobitz_ii_p_waves_per_qrs=ratio,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_afib_parameter_validation(self):
        """Test atrial fibrillation parameter validation."""
        
        # Valid AFib parameters
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_atrial_fibrillation=True,
            afib_average_ventricular_rate_bpm=80,
            afib_irregularity_factor=0.3,
            afib_fibrillation_wave_amplitude_mv=0.1,
            enable_pvc=False,
            enable_pac=False,
            enable_vt=False
        )
        
        assert params.afib_average_ventricular_rate_bpm == 80
        assert params.afib_irregularity_factor == 0.3
        
        # Invalid AFib rate (outside 30-220 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_atrial_fibrillation=True,
                afib_average_ventricular_rate_bpm=250,  # Too fast (>220)
                enable_pvc=False,
                enable_pac=False,
                enable_vt=False
            )
        
        # Invalid irregularity factor (outside 0.05-0.50 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_atrial_fibrillation=True,
                afib_irregularity_factor=0.6,  # Too high (>0.50)
                enable_pvc=False,
                enable_pac=False,
                enable_vt=False
            )
    
    @pytest.mark.unit
    def test_vt_parameter_validation(self):
        """Test ventricular tachycardia parameter validation."""
        
        # Valid VT parameters
        params = AdvancedECGParams(
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
        
        assert params.vt_start_time_sec == 5.0
        assert params.vt_duration_sec == 8.0
        assert params.vt_rate_bpm == 180
        
        # Invalid VT start time (negative)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=20.0,
                enable_vt=True,
                vt_start_time_sec=-1.0,
                vt_duration_sec=5.0,
                vt_rate_bpm=180,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False
            )
        
        # Invalid VT rate (outside 100-250 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=20.0,
                enable_vt=True,
                vt_start_time_sec=5.0,
                vt_duration_sec=5.0,
                vt_rate_bpm=80,  # Too slow (<100)
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False
            )
    
    @pytest.mark.unit
    def test_flutter_parameter_validation(self):
        """Test atrial flutter parameter validation."""
        
        # Valid flutter parameters
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_atrial_flutter=True,
            atrial_flutter_rate_bpm=300,
            atrial_flutter_av_block_ratio_qrs_to_f=4,
            atrial_flutter_wave_amplitude_mv=0.15,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        assert params.atrial_flutter_rate_bpm == 300
        assert params.atrial_flutter_av_block_ratio_qrs_to_f == 4
        
        # Invalid flutter rate (outside 200-400 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_atrial_flutter=True,
                atrial_flutter_rate_bpm=150,  # Too slow (<200)
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
    
    @pytest.mark.unit
    def test_torsades_parameter_validation(self):
        """Test Torsades de Pointes parameter validation."""
        
        # Valid Torsades parameters
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=30.0,
            target_qtc_ms=520.0,
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.001,
            use_sensitive_qtc_threshold=False,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        assert params.torsades_probability_per_beat == 0.001
        assert params.use_sensitive_qtc_threshold == False
        
        # Invalid Torsades probability (outside 0.0-0.1 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=30.0,
                enable_torsades_risk=True,
                torsades_probability_per_beat=0.2,  # Too high (>0.1)
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
    
    @pytest.mark.unit
    def test_electrical_axis_parameter_validation(self):
        """Test electrical axis override parameter validation."""
        
        # Valid axis values
        valid_axes = [-180, -90, -30, 0, 30, 60, 90, 120, 180]
        for axis in valid_axes:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=10.0,
                enable_axis_override=True,
                target_axis_degrees=axis,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            assert params.target_axis_degrees == axis
        
        # Invalid axis values
        invalid_axes = [-200, -181, 181, 200, 360]
        for axis in invalid_axes:
            with pytest.raises(ValidationError):
                AdvancedECGParams(
                    heart_rate_bpm=60,
                    duration_sec=10.0,
                    enable_axis_override=True,
                    target_axis_degrees=axis,
                    enable_pvc=False,
                    enable_pac=False,
                    enable_atrial_fibrillation=False,
                    enable_vt=False
                )
    
    @pytest.mark.unit
    def test_wenckebach_parameter_validation(self):
        """Test Wenckebach AV block parameter validation."""
        
        # Valid Wenckebach parameters
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=15.0,
            enable_mobitz_i_wenckebach=True,
            wenckebach_initial_pr_sec=0.16,
            wenckebach_pr_increment_sec=0.04,
            wenckebach_max_pr_before_drop_sec=0.28,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        assert params.wenckebach_initial_pr_sec == 0.16
        assert params.wenckebach_pr_increment_sec == 0.04
        
        # Invalid initial PR (outside 0.12-0.40 range)
        with pytest.raises(ValidationError):
            AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=15.0,
                enable_mobitz_i_wenckebach=True,
                wenckebach_initial_pr_sec=0.10,  # Too short (<0.12)
                wenckebach_pr_increment_sec=0.04,
                wenckebach_max_pr_before_drop_sec=0.28,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
    
    @pytest.mark.unit
    def test_default_values(self):
        """Test that default values are reasonable."""
        # Create params with minimal required fields
        params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=10.0,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        # Check default values are in reasonable ranges
        assert 300 <= params.target_qtc_ms <= 600
        assert params.torsades_probability_per_beat <= 0.1
        assert 0 <= params.pvc_probability_per_sinus <= 1.0
        assert 0 <= params.pac_probability_per_sinus <= 1.0
    
    @pytest.mark.unit
    def test_parameter_combinations(self):
        """Test parameter combinations that should be valid."""
        
        # Complex but valid parameter combination
        params = AdvancedECGParams(
            heart_rate_bpm=75,
            duration_sec=30.0,
            target_qtc_ms=450.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.1,
            enable_pac=True,
            pac_probability_per_sinus=0.05,
            enable_atrial_fibrillation=False,
            enable_vt=True,
            vt_start_time_sec=15.0,
            vt_duration_sec=8.0,
            vt_rate_bpm=180,
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.002,
            enable_axis_override=True,
            target_axis_degrees=60.0
        )
        
        # Should create successfully
        assert params.heart_rate_bpm == 75
        assert params.enable_vt == True
        assert params.vt_rate_bpm == 180