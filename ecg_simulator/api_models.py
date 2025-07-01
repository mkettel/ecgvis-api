# // backend/ecg-simulator/api_models.py
from pydantic import BaseModel, Field
from typing import Optional

class AdvancedECGParams(BaseModel):
    heart_rate_bpm: float = Field(60.0, gt=0, description="Base sinus rate if no other dominant rhythm.")
    duration_sec: float = Field(10.0, gt=0)

    # QT Interval Parameters (Bazett's Formula)
    target_qtc_ms: float = Field(400.0, ge=300, le=600, description="Target corrected QT interval in milliseconds (QTc). Normal range: 350-450ms (men), 360-460ms (women). Values >500ms may predispose to Torsades de Pointes.")
    
    # Torsades de Pointes Parameters
    enable_torsades_risk: bool = Field(True, description="Enable Torsades de Pointes risk assessment based on QTc")
    torsades_probability_per_beat: float = Field(0.001, ge=0.0, le=0.1, description="Base probability per beat of Torsades initiation when QTc > threshold (0.1% default)")
    use_sensitive_qtc_threshold: bool = Field(False, description="Use 480ms QTc threshold instead of 500ms for sensitive patients")

    enable_pvc: bool = Field(False)
    pvc_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    enable_pac: bool = Field(False, description="Enable Premature Atrial Contractions.")
    pac_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)

    first_degree_av_block_pr_sec: Optional[float] = Field(None, ge=0.201, le=0.60)
    enable_mobitz_ii_av_block: bool = Field(False)
    mobitz_ii_p_waves_per_qrs: int = Field(2, ge=2)
    enable_mobitz_i_wenckebach: bool = Field(False)
    wenckebach_initial_pr_sec: float = Field(0.16, ge=0.12, le=0.40)
    wenckebach_pr_increment_sec: float = Field(0.04, ge=0.01, le=0.15)
    wenckebach_max_pr_before_drop_sec: float = Field(0.32, ge=0.22, le=0.70)

    enable_third_degree_av_block: bool = Field(False)
    third_degree_escape_rhythm_origin: str = Field("junctional", pattern="^(junctional|ventricular)$")
    third_degree_escape_rate_bpm: Optional[float] = Field(None, gt=15, lt=65)

    enable_atrial_fibrillation: bool = Field(False)
    afib_average_ventricular_rate_bpm: int = Field(100, ge=30, le=220)
    afib_fibrillation_wave_amplitude_mv: float = Field(0.05, ge=0.0, le=0.2)
    afib_irregularity_factor: float = Field(0.20, ge=0.05, le=0.50)

    enable_atrial_flutter: bool = Field(False)
    atrial_flutter_rate_bpm: int = Field(300, ge=200, le=400)
    atrial_flutter_av_block_ratio_qrs_to_f: int = Field(2, ge=1)
    atrial_flutter_wave_amplitude_mv: float = Field(0.15, ge=0.05, le=0.5)

    allow_svt_initiation_by_pac: bool = Field(False, description="Allow PACs to trigger SVT episodes.")
    svt_initiation_probability_after_pac: float = Field(0.3, ge=0.0, le=1.0)
    svt_duration_sec: float = Field(10.0, gt=0)
    svt_rate_bpm: int = Field(180, ge=150, le=250)

    enable_vt: bool = Field(False, description="Enable an episode of Ventricular Tachycardia.")
    vt_start_time_sec: Optional[float] = Field(None, ge=0, description="Start time of VT episode (seconds). If None and enable_vt is true, starts near beginning.")
    vt_duration_sec: float = Field(5.0, gt=0, description="Duration of VT episode once initiated (seconds).")
    vt_rate_bpm: int = Field(160, ge=100, le=250, description="Rate of VT when active (bpm).")

    # Bundle Branch Block Parameters
    enable_rbbb: bool = Field(False, description="Enable Right Bundle Branch Block")
    enable_lbbb: bool = Field(False, description="Enable Left Bundle Branch Block")
    
    # VFib Parameters
    enable_vfib: bool = Field(False, description="Enable Ventricular Fibrillation")
    vfib_start_time_sec: Optional[float] = Field(None, ge=0, description="Start time of VFib episode")
    vfib_duration_sec: float = Field(5.0, gt=0, description="Duration of VFib episode")
    
    # Electrical Axis Override (Development/Admin)
    enable_axis_override: bool = Field(False, description="Enable manual electrical axis override")
    target_axis_degrees: float = Field(60.0, ge=-180, le=180, description="Target electrical axis in degrees (-180 to +180)")