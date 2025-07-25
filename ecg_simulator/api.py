# // backend/ecg-simulator/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_models import AdvancedECGParams
from .rhythm_logic import generate_physiologically_accurate_ecg, generate_physiologically_accurate_ecg_12_lead
from .constants import FS # Import FS from constants

app = FastAPI()


@app.post("/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):

    time_axis, ecg_signal, rhythm_description = generate_physiologically_accurate_ecg(
        heart_rate_bpm=params.heart_rate_bpm,
        duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc,
        pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac,
        pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block,
        mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs,
        enable_mobitz_i_wenckebach=params.enable_mobitz_i_wenckebach,
        wenckebach_initial_pr_sec=params.wenckebach_initial_pr_sec,
        wenckebach_pr_increment_sec=params.wenckebach_pr_increment_sec,
        wenckebach_max_pr_before_drop_sec=params.wenckebach_max_pr_before_drop_sec,
        enable_third_degree_av_block=params.enable_third_degree_av_block,
        third_degree_escape_rhythm_origin=params.third_degree_escape_rhythm_origin,
        third_degree_escape_rate_bpm=params.third_degree_escape_rate_bpm,
        enable_atrial_fibrillation=params.enable_atrial_fibrillation,
        afib_average_ventricular_rate_bpm=params.afib_average_ventricular_rate_bpm,
        afib_fibrillation_wave_amplitude_mv=params.afib_fibrillation_wave_amplitude_mv,
        afib_irregularity_factor=params.afib_irregularity_factor,
        enable_atrial_flutter=params.enable_atrial_flutter,
        atrial_flutter_rate_bpm=params.atrial_flutter_rate_bpm,
        atrial_flutter_av_block_ratio_qrs_to_f=params.atrial_flutter_av_block_ratio_qrs_to_f,
        atrial_flutter_wave_amplitude_mv=params.atrial_flutter_wave_amplitude_mv,
        allow_svt_initiation_by_pac=params.allow_svt_initiation_by_pac,
        svt_initiation_probability_after_pac=params.svt_initiation_probability_after_pac,
        svt_duration_sec=params.svt_duration_sec,
        svt_rate_bpm=params.svt_rate_bpm,
        enable_vt=params.enable_vt,
        vt_start_time_sec=params.vt_start_time_sec,
        vt_duration_sec=params.vt_duration_sec,
        vt_rate_bpm=params.vt_rate_bpm,
        enable_rbbb=params.enable_rbbb,
        enable_lbbb=params.enable_lbbb,
        enable_vfib=params.enable_vfib,
        vfib_start_time_sec=params.vfib_start_time_sec,
        vfib_duration_sec=params.vfib_duration_sec,
        enable_axis_override=params.enable_axis_override,
        target_axis_degrees=params.target_axis_degrees,
        fs=FS, # Pass the imported FS here
        target_qtc_ms=params.target_qtc_ms,
        enable_torsades_risk=params.enable_torsades_risk,
        torsades_probability_per_beat=params.torsades_probability_per_beat,
        use_sensitive_qtc_threshold=params.use_sensitive_qtc_threshold
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": rhythm_description}


@app.post("/generate_advanced_ecg_12_lead")
async def get_advanced_ecg_12_lead_data(params: AdvancedECGParams):
    """
    Generate 12-lead ECG using 3D cardiac vector projection.
    Returns all 12 standard ECG leads (I, II, III, aVR, aVL, aVF, V1-V6).
    """
    try:
        time_axis, twelve_lead_signals, rhythm_description = generate_physiologically_accurate_ecg_12_lead(
        heart_rate_bpm=params.heart_rate_bpm,
        duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc,
        pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac,
        pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block,
        mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs,
        enable_mobitz_i_wenckebach=params.enable_mobitz_i_wenckebach,
        wenckebach_initial_pr_sec=params.wenckebach_initial_pr_sec,
        wenckebach_pr_increment_sec=params.wenckebach_pr_increment_sec,
        wenckebach_max_pr_before_drop_sec=params.wenckebach_max_pr_before_drop_sec,
        enable_third_degree_av_block=params.enable_third_degree_av_block,
        third_degree_escape_rhythm_origin=params.third_degree_escape_rhythm_origin,
        third_degree_escape_rate_bpm=params.third_degree_escape_rate_bpm,
        enable_atrial_fibrillation=params.enable_atrial_fibrillation,
        afib_average_ventricular_rate_bpm=params.afib_average_ventricular_rate_bpm,
        afib_fibrillation_wave_amplitude_mv=params.afib_fibrillation_wave_amplitude_mv,
        afib_irregularity_factor=params.afib_irregularity_factor,
        enable_atrial_flutter=params.enable_atrial_flutter,
        atrial_flutter_rate_bpm=params.atrial_flutter_rate_bpm,
        atrial_flutter_av_block_ratio_qrs_to_f=params.atrial_flutter_av_block_ratio_qrs_to_f,
        atrial_flutter_wave_amplitude_mv=params.atrial_flutter_wave_amplitude_mv,
        allow_svt_initiation_by_pac=params.allow_svt_initiation_by_pac,
        svt_initiation_probability_after_pac=params.svt_initiation_probability_after_pac,
        svt_duration_sec=params.svt_duration_sec,
        svt_rate_bpm=params.svt_rate_bpm,
        enable_vt=params.enable_vt,
        vt_start_time_sec=params.vt_start_time_sec,
        vt_duration_sec=params.vt_duration_sec,
        vt_rate_bpm=params.vt_rate_bpm,
        enable_rbbb=params.enable_rbbb,
        enable_lbbb=params.enable_lbbb,
        enable_vfib=params.enable_vfib,
        vfib_start_time_sec=params.vfib_start_time_sec,
        vfib_duration_sec=params.vfib_duration_sec,
        enable_axis_override=params.enable_axis_override,
        target_axis_degrees=params.target_axis_degrees,
            fs=FS,
            target_qtc_ms=params.target_qtc_ms,
            enable_torsades_risk=params.enable_torsades_risk,
            torsades_probability_per_beat=params.torsades_probability_per_beat,
            use_sensitive_qtc_threshold=params.use_sensitive_qtc_threshold
        )
        
        # Convert numpy arrays to lists for JSON serialization
        twelve_lead_signals_json = {
            lead_name: signal.tolist() if hasattr(signal, 'tolist') else signal
            for lead_name, signal in twelve_lead_signals.items()
        }
        
        return {
            "time_axis": time_axis, 
            "twelve_lead_signals": twelve_lead_signals_json, 
            "rhythm_generated": rhythm_description
        }
    except Exception as e:
        import traceback
        print(f"ERROR in 12-lead endpoint: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise e