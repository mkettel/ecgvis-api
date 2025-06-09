# /backend/ecg_simulator/rhythm_logic.py
import numpy as np
import heapq
import math
from typing import List, Dict, Any, Optional

from .constants import (
    BASELINE_MV, MIN_REFRACTORY_PERIOD_SEC, BEAT_MORPHOLOGIES,
    PVC_COUPLING_FACTOR, PAC_COUPLING_FACTOR, get_rate_corrected_intervals
)
from .beat_generation import generate_single_beat_morphology, generate_single_beat_3d_vectors
from .full_ecg.vector_projection import project_cardiac_vector_to_12_leads
from .waveform_primitives import generate_fibrillatory_waves

# --- Event-Driven Rhythm Generation ---
class BeatEvent:
    def __init__(self, time: float, beat_type: str, source: str = "sa_node"):
        self.time = time
        self.beat_type = beat_type
        self.source = source
    def __lt__(self, other): return self.time < other.time
    def __repr__(self): return f"BeatEvent(t={self.time:.3f}, type='{self.beat_type}', src='{self.source}')"

def generate_physiologically_accurate_ecg(
    heart_rate_bpm: float, duration_sec: float,
    enable_pvc: bool, pvc_probability_per_sinus: float,
    enable_pac: bool, pac_probability_per_sinus: float,
    first_degree_av_block_pr_sec: Optional[float],
    enable_mobitz_ii_av_block: bool, mobitz_ii_p_waves_per_qrs: int,
    enable_mobitz_i_wenckebach: bool, wenckebach_initial_pr_sec: float,
    wenckebach_pr_increment_sec: float, wenckebach_max_pr_before_drop_sec: float,
    enable_third_degree_av_block: bool, third_degree_escape_rhythm_origin: str,
    third_degree_escape_rate_bpm: Optional[float],
    enable_atrial_fibrillation: bool, afib_average_ventricular_rate_bpm: int,
    afib_fibrillation_wave_amplitude_mv: float, afib_irregularity_factor: float,
    enable_atrial_flutter: bool, atrial_flutter_rate_bpm: int,
    atrial_flutter_av_block_ratio_qrs_to_f: int, atrial_flutter_wave_amplitude_mv: float,
    allow_svt_initiation_by_pac: bool,
    svt_initiation_probability_after_pac: float,
    svt_duration_sec: float,
    svt_rate_bpm: int,
    # VT Parameters
    enable_vt: bool,
    vt_start_time_sec: Optional[float],
    vt_duration_sec: float,
    vt_rate_bpm: int,
    fs: int,
    # QT Correction Parameter
    target_qtc_ms: float = 400.0
):
    base_rr_interval_sec = 60.0 / heart_rate_bpm if heart_rate_bpm > 0 else float('inf')
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)
    event_queue: List[BeatEvent] = []
    
    sa_node_next_fire_time = 0.0
    sa_node_last_actual_fire_time_for_p_wave = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    
    last_placed_qrs_onset_time = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    ventricle_ready_for_next_qrs_at_time = 0.0
    
    p_wave_counter_for_mobitz_ii = 0
    current_wenckebach_pr_sec = wenckebach_initial_pr_sec if enable_mobitz_i_wenckebach else None

    # SVT State
    is_svt_currently_active: bool = False
    svt_termination_time: Optional[float] = None
    svt_actual_start_time: Optional[float] = None
    svt_actual_end_time: Optional[float] = None

    # VT State
    is_vt_currently_active: bool = False
    vt_actual_start_time: Optional[float] = None
    vt_calculated_termination_time: Optional[float] = None 
    vt_actual_end_time: Optional[float] = None
    
    # Rhythm hierarchy determination
    is_dynamic_svt_episode_configured = allow_svt_initiation_by_pac
    is_vt_episode_configured = enable_vt
    
    is_aflutter_active_base = enable_atrial_flutter and not is_dynamic_svt_episode_configured and not is_vt_episode_configured
    is_afib_active_base = enable_atrial_fibrillation and not is_dynamic_svt_episode_configured and not is_aflutter_active_base and not is_vt_episode_configured
    is_third_degree_block_active_base = enable_third_degree_av_block and not is_dynamic_svt_episode_configured and not is_afib_active_base and not is_aflutter_active_base and not is_vt_episode_configured
    is_mobitz_i_active_base = enable_mobitz_i_wenckebach and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_vt_episode_configured)
    is_mobitz_ii_active_base = enable_mobitz_ii_av_block and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_vt_episode_configured)
    is_first_degree_av_block_active_base = (first_degree_av_block_pr_sec is not None) and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_mobitz_ii_active_base or is_vt_episode_configured)
    
    flutter_wave_rr_interval_sec = 0.0
    flutter_wave_counter_for_av_block = 0

    # Initial Event Scheduling
    if is_aflutter_active_base:
        flutter_wave_rr_interval_sec = 60.0 / atrial_flutter_rate_bpm if atrial_flutter_rate_bpm > 0 else float('inf')
        if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(0.0, "flutter_wave", "aflutter_focus"))
    elif is_afib_active_base:
        mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
        if mean_afib_rr_sec > 0 and mean_afib_rr_sec != float('inf'):
            first_qrs_delay = np.random.uniform(0.1, mean_afib_rr_sec * 0.6) 
            heapq.heappush(event_queue, BeatEvent(first_qrs_delay, "afib_conducted", "afib_av_node"))
    elif is_third_degree_block_active_base:
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
             heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
        escape_beat_type = "junctional_escape" if third_degree_escape_rhythm_origin == "junctional" else "ventricular_escape"
        default_escape_rate = 45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0
        actual_escape_rate_bpm = third_degree_escape_rate_bpm or default_escape_rate
        escape_rr_interval_sec = 60.0 / actual_escape_rate_bpm if actual_escape_rate_bpm > 0 else float('inf')
        sinus_pr_for_offset = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
        if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
             sinus_pr_for_offset = first_degree_av_block_pr_sec
        first_escape_fire_time = max(0.1, sinus_pr_for_offset + np.random.uniform(0.05, 0.15))
        if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(first_escape_fire_time, escape_beat_type, f"{third_degree_escape_rhythm_origin}_escape"))
    elif not is_vt_episode_configured or (vt_start_time_sec is not None and vt_start_time_sec > 0.05): 
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    if is_vt_episode_configured:
        actual_vt_start_time_for_signal = vt_start_time_sec if vt_start_time_sec is not None else 0.05 
        if actual_vt_start_time_for_signal < duration_sec:
            heapq.heappush(event_queue, BeatEvent(actual_vt_start_time_for_signal, "vt_initiation_signal", "vt_control"))


    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_event_time = current_event.time
        
        if current_event.beat_type == "vt_initiation_signal":
            if not is_vt_currently_active: 
                if is_svt_currently_active:
                    print(f"DEBUG: VT interrupting active SVT at {potential_event_time:.3f}.")
                    is_svt_currently_active = False 
                    svt_termination_time = None 
                    event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= potential_event_time)]
                    heapq.heapify(event_queue)

                is_vt_currently_active = True
                vt_actual_start_time = potential_event_time
                vt_calculated_termination_time = vt_actual_start_time + vt_duration_sec
                print(f"DEBUG: VT Initiated. Start: {vt_actual_start_time:.3f}, Scheduled Term Time: {vt_calculated_termination_time:.3f}")

                new_event_queue = []
                for e_val in event_queue:
                    is_sa_or_pac = e_val.source == "sa_node" or e_val.beat_type == "pac"
                    if e_val.time < vt_actual_start_time or \
                       (vt_calculated_termination_time is not None and e_val.time >= vt_calculated_termination_time) or \
                       not is_sa_or_pac: 
                        new_event_queue.append(e_val)
                event_queue = new_event_queue
                heapq.heapify(event_queue)
                print(f"DEBUG: Event queue after VT initiation cleanup: {event_queue}")

                vt_rr = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
                if vt_rr != float('inf'):
                    first_vt_beat_time = vt_actual_start_time 
                    if first_vt_beat_time < duration_sec and first_vt_beat_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                        heapq.heappush(event_queue, BeatEvent(first_vt_beat_time, "vt_beat", "vt_focus"))
                        physio_pause = 0.15
                        resume_time = vt_actual_start_time + vt_duration_sec + physio_pause
                        heapq.heappush(event_queue, BeatEvent(resume_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        if is_vt_currently_active and vt_calculated_termination_time is not None and potential_event_time >= vt_calculated_termination_time:
            print(f"DEBUG: VT Terminating. Potential Event Time: {potential_event_time:.3f}, VT Scheduled Term Time: {vt_calculated_termination_time:.3f}")
            vt_actual_end_time = vt_calculated_termination_time
            is_vt_currently_active = False
            vt_calculated_termination_time = None 

            event_queue = [e for e in event_queue if not (e.beat_type == "vt_beat" and e.time >= vt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            
            # This is the time to schedule the next SA node beat after VT
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                time_since_last_sa_p_before_vt_era = vt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_vt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_vt_pause = 0.15
                sa_node_next_fire_time_after_vt = max(vt_actual_end_time + physiological_post_vt_pause, resumed_sa_fire_time)
                print(f"DEBUG: Post-VT SA. Last P: {sa_node_last_actual_fire_time_for_p_wave:.3f}, VT End: {vt_actual_end_time:.3f}, "
                      f"Resumed SA (Proj): {resumed_sa_fire_time:.3f}, Final SA Sched: {sa_node_next_fire_time_after_vt:.3f}")
                
                if sa_node_next_fire_time_after_vt < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_vt) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_vt, "sinus", "sa_node"))
                        print(f"DEBUG: Pushed resumed Sinus beat post-VT at {sa_node_next_fire_time_after_vt:.3f}")
                else:
                    print(f"DEBUG: Resumed SA beat post-VT calculated for {sa_node_next_fire_time_after_vt:.3f} "
                          f"is NOT scheduled because it is >= duration_sec ({duration_sec:.3f}).")
            
            if current_event.beat_type == "vt_beat" and potential_event_time >= vt_actual_end_time - 0.001 : # This will be the last VT beat
                if event_queue and event_queue[0].time < duration_sec: continue
                else: break
        
        if is_svt_currently_active and svt_termination_time is not None and potential_event_time >= svt_termination_time:
            print(f"DEBUG: SVT Terminating. Potential Event Time: {potential_event_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
            svt_actual_end_time = svt_termination_time 
            is_svt_currently_active = False 
            svt_termination_time = None

            event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= svt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                time_since_last_sa_p_before_svt_era = svt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_svt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_svt_pause = 0.1 
                sa_node_next_fire_time_after_svt = max(svt_actual_end_time + physiological_post_svt_pause, resumed_sa_fire_time)
                print(f"DEBUG: Post-SVT SA. Last P: {sa_node_last_actual_fire_time_for_p_wave:.3f}, SVT End: {svt_actual_end_time:.3f}, "
                      f"Resumed SA (Proj): {resumed_sa_fire_time:.3f}, Final SA Sched: {sa_node_next_fire_time_after_svt:.3f}")

                if sa_node_next_fire_time_after_svt < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_svt) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_svt, "sinus", "sa_node"))
                        print(f"DEBUG: Pushed resumed Sinus beat post-SVT at {sa_node_next_fire_time_after_svt:.3f}")
                else:
                     print(f"DEBUG: Resumed SA beat post-SVT calculated for {sa_node_next_fire_time_after_svt:.3f} "
                           f"is NOT scheduled because it is >= duration_sec ({duration_sec:.3f}).")
            
            if current_event.beat_type == "svt_beat" and potential_event_time >= svt_actual_end_time - 0.001 :
                if event_queue and event_queue[0].time < duration_sec: continue 
                else: break
        
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac"
        is_escape_event = current_event.source.endswith("_escape")
        is_afib_qrs_event = current_event.source == "afib_av_node"
        is_flutter_wave_event = current_event.beat_type == "flutter_wave"
        is_flutter_conducted_qrs_event = current_event.beat_type == "flutter_conducted_qrs"
        is_svt_beat_event_type = current_event.beat_type == "svt_beat"
        is_vt_beat_event_type = current_event.beat_type == "vt_beat"

        if (is_svt_currently_active or is_vt_currently_active) and (is_atrial_origin_event or current_event.source == "sa_node"):
            if current_event.source == "sa_node": 
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        if not (is_svt_currently_active or is_vt_currently_active) and \
           (is_afib_active_base or is_aflutter_active_base) and is_atrial_origin_event:
            if current_event.source == "sa_node":
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                 if not (is_afib_active_base or is_aflutter_active_base or is_third_degree_block_active_base):
                    if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                        if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        
        # Determine effective heart rate for QT correction based on current rhythm
        effective_heart_rate = heart_rate_bpm  # Default to sinus rate
        if is_vt_currently_active:
            effective_heart_rate = vt_rate_bpm
        elif is_svt_currently_active:
            effective_heart_rate = svt_rate_bpm
        elif enable_atrial_fibrillation and (is_afib_active_base or current_event.beat_type == "afib_conducted"):
            effective_heart_rate = afib_average_ventricular_rate_bpm
        elif enable_atrial_flutter and (is_aflutter_active_base or current_event.beat_type == "flutter_conducted_qrs"):
            # Calculate effective ventricular rate for flutter
            effective_heart_rate = atrial_flutter_rate_bpm / atrial_flutter_av_block_ratio_qrs_to_f
        
        # Apply Bazett's formula for rate-corrected QT intervals
        current_beat_morph_params = get_rate_corrected_intervals(effective_heart_rate, current_beat_morph_params, target_qtc_ms)
        qrs_is_blocked_by_av_node = False
        draw_p_wave_only_for_this_atrial_event = False
        
        if is_flutter_wave_event:
            flutter_wave_params_local = BEAT_MORPHOLOGIES["flutter_wave"].copy()
            flutter_wave_params_local["p_amplitude"] = atrial_flutter_wave_amplitude_mv
            flutter_wave_params_local["p_duration"] = flutter_wave_rr_interval_sec
            _, y_flutter_wave_shape, _ = generate_single_beat_morphology(flutter_wave_params_local, fs, is_flutter_wave_itself=True)
            if len(y_flutter_wave_shape) > 0:
                fw_start_time_global = potential_event_time
                fw_start_sample_idx = int(fw_start_time_global * fs)
                fw_end_sample_idx = min(fw_start_sample_idx + len(y_flutter_wave_shape), num_total_samples)
                fw_samples_to_copy = fw_end_sample_idx - fw_start_sample_idx
                if fw_samples_to_copy > 0 and fw_start_sample_idx < num_total_samples:
                    full_ecg_signal_np[fw_start_sample_idx:fw_end_sample_idx] += y_flutter_wave_shape[:fw_samples_to_copy]
            
            flutter_wave_counter_for_av_block += 1
            conducts_this_flutter_wave = (flutter_wave_counter_for_av_block % atrial_flutter_av_block_ratio_qrs_to_f == 0) if atrial_flutter_av_block_ratio_qrs_to_f > 0 else False
            
            if conducts_this_flutter_wave and potential_event_time >= ventricle_ready_for_next_qrs_at_time:
                 if not is_vt_currently_active and not is_svt_currently_active: # AVN conduction suppressed by ventricular/supraventricular tachy
                    flutter_qrs_pr = BEAT_MORPHOLOGIES["flutter_conducted_qrs"]["pr_interval"] 
                    qrs_time_after_flutter = potential_event_time + flutter_qrs_pr 
                    heapq.heappush(event_queue, BeatEvent(qrs_time_after_flutter, "flutter_conducted_qrs", "aflutter_conducted"))
            
            if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_counter_for_av_block >= atrial_flutter_av_block_ratio_qrs_to_f:
                flutter_wave_counter_for_av_block = 0

            if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
                next_fw_time = potential_event_time + flutter_wave_rr_interval_sec
                if next_fw_time < duration_sec:
                    if not is_vt_currently_active and not is_svt_currently_active : # Flutter focus suppressed by tachy
                         heapq.heappush(event_queue, BeatEvent(next_fw_time, "flutter_wave", "aflutter_focus"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        if not is_svt_currently_active and not is_vt_currently_active and \
           not (is_afib_active_base or is_aflutter_active_base) and \
           not is_svt_beat_event_type and not is_vt_beat_event_type and \
           not is_afib_qrs_event and \
           not is_flutter_conducted_qrs_event and \
           not is_escape_event:
            
            if is_atrial_origin_event: 
                 sa_node_last_actual_fire_time_for_p_wave = potential_event_time

            if is_third_degree_block_active_base and is_atrial_origin_event:
                qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec :
                    current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_mobitz_i_active_base and is_atrial_origin_event:
                if current_wenckebach_pr_sec is None: current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                current_beat_morph_params["pr_interval"] = current_wenckebach_pr_sec
                if current_wenckebach_pr_sec >= wenckebach_max_pr_before_drop_sec:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                    current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                else: current_wenckebach_pr_sec += wenckebach_pr_increment_sec
            elif is_mobitz_ii_active_base and is_atrial_origin_event:
                p_wave_counter_for_mobitz_ii += 1
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1 and mobitz_ii_p_waves_per_qrs > 1 :
                     qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs : p_wave_counter_for_mobitz_ii = 0
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                     current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_first_degree_av_block_active_base and is_atrial_origin_event and first_degree_av_block_pr_sec:
                current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        if not draw_p_wave_only_for_this_atrial_event and \
           not qrs_is_blocked_by_av_node and \
           not is_escape_event and not is_vt_beat_event_type and \
           potential_event_time < ventricle_ready_for_next_qrs_at_time:
            qrs_is_blocked_by_av_node = True
            if is_atrial_origin_event: draw_p_wave_only_for_this_atrial_event = True 
        
        if qrs_is_blocked_by_av_node:
            if draw_p_wave_only_for_this_atrial_event:
                _, y_p_wave_shape, p_wave_offset_for_drawing = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=True)
                if len(y_p_wave_shape) > 0:
                    p_wave_start_time_global = potential_event_time - p_wave_offset_for_drawing
                    p_start_sample_idx_global = int(p_wave_start_time_global * fs)
                    p_shape_start_idx, p_place_start_idx = 0, p_start_sample_idx_global
                    if p_place_start_idx < 0: p_shape_start_idx = -p_place_start_idx; p_place_start_idx = 0
                    p_samples_in_shape = len(y_p_wave_shape) - p_shape_start_idx
                    p_samples_in_signal = num_total_samples - p_place_start_idx
                    p_samples_to_copy = min(p_samples_in_shape, p_samples_in_signal)
                    if p_samples_to_copy > 0:
                        p_shape_end_idx = p_shape_start_idx + p_samples_to_copy; p_place_end_idx = p_place_start_idx + p_samples_to_copy
                        full_ecg_signal_np[p_place_start_idx : p_place_end_idx] += y_p_wave_shape[p_shape_start_idx : p_shape_end_idx]

            if current_event.source == "sa_node" and not is_svt_currently_active and not is_vt_currently_active and not (is_afib_active_base or is_aflutter_active_base):
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        _, y_beat_shape, qrs_offset_from_shape_start = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=False)
        if len(y_beat_shape) > 0:
            waveform_start_time_global = potential_event_time - qrs_offset_from_shape_start
            start_sample_index_global = int(waveform_start_time_global * fs)
            shape_start_idx, place_start_idx = 0, start_sample_index_global
            if place_start_idx < 0: shape_start_idx = -place_start_idx; place_start_idx = 0
            samples_in_shape_remaining = len(y_beat_shape) - shape_start_idx
            samples_in_signal_remaining = num_total_samples - place_start_idx
            samples_to_copy = min(samples_in_shape_remaining, samples_in_signal_remaining)
            if samples_to_copy > 0:
                shape_end_idx = shape_start_idx + samples_to_copy; place_end_idx = place_start_idx + samples_to_copy
                full_ecg_signal_np[place_start_idx : place_end_idx] += y_beat_shape[shape_start_idx : shape_end_idx]

        actual_rr_to_this_beat = potential_event_time - last_placed_qrs_onset_time
        last_placed_qrs_onset_time = potential_event_time
        qrs_duration_this_beat = current_beat_morph_params.get('qrs_duration', 0.10)
        ventricle_ready_for_next_qrs_at_time = potential_event_time + max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat * 1.8 if qrs_duration_this_beat else MIN_REFRACTORY_PERIOD_SEC)

        if is_vt_beat_event_type and is_vt_currently_active:
            vt_rr_interval_sec = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
            next_vt_event_time = potential_event_time + vt_rr_interval_sec
            if vt_rr_interval_sec != float('inf') and next_vt_event_time < duration_sec and \
               next_vt_event_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_vt_event_time, "vt_beat", "vt_focus"))
                print(f"DEBUG: Pushed VT beat at {next_vt_event_time:.3f}")

        elif is_svt_beat_event_type and is_svt_currently_active:
            svt_rr_interval_sec = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
            next_svt_event_time = potential_event_time + svt_rr_interval_sec
            if svt_rr_interval_sec != float('inf') and next_svt_event_time < duration_sec and \
               next_svt_event_time < (svt_termination_time if svt_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_svt_event_time, "svt_beat", "svt_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else svt_rr_interval_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and \
                   pvc_time < next_svt_event_time - 0.100 and \
                   (svt_termination_time is None or pvc_time < svt_termination_time - 0.100) :
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_afib_qrs_event: 
            mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
            std_dev_rr = mean_afib_rr_sec * afib_irregularity_factor
            next_rr_variation = np.random.normal(0, std_dev_rr)
            tentative_next_rr = mean_afib_rr_sec + next_rr_variation
            min_physiological_rr = max(MIN_REFRACTORY_PERIOD_SEC, (qrs_duration_this_beat or 0)) + 0.05 
            next_rr = max(min_physiological_rr, tentative_next_rr)
            next_afib_qrs_event_time = potential_event_time + next_rr
            if mean_afib_rr_sec != float('inf') and next_afib_qrs_event_time < duration_sec:
                if not is_vt_currently_active and not is_svt_currently_active : # Afib suppressed by other tachy
                    heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_afib_qrs_event_time - 0.100 : 
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_flutter_conducted_qrs_event: 
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    ventricular_rr_in_flutter = (flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f) if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_rr_interval_sec > 0 else float('inf')
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        
        elif current_event.source == "sa_node": 
            sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active and not is_vt_currently_active: 
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active:
                    pac_time = potential_event_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                    if pac_time > potential_event_time + 0.100 and pac_time < sa_node_next_fire_time - 0.100: 
                        heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active:
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    pr_interval_for_next_sinus = current_beat_morph_params.get('pr_interval', BEAT_MORPHOLOGIES["sinus"]['pr_interval'])
                    next_potential_sa_qrs = sa_node_next_fire_time + pr_interval_for_next_sinus
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs - 0.100:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pac": 
            sa_node_next_fire_time = potential_event_time + base_rr_interval_sec 
            new_event_queue = [e for e in event_queue if not (e.source == "sa_node")] 
            heapq.heapify(new_event_queue); event_queue = new_event_queue
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active and not is_vt_currently_active: 
                    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

            if is_dynamic_svt_episode_configured and not is_svt_currently_active and not is_vt_currently_active and \
               not is_afib_active_base and not is_aflutter_active_base and not is_third_degree_block_active_base:
                if np.random.rand() < svt_initiation_probability_after_pac:
                    is_svt_currently_active = True
                    svt_actual_start_time = potential_event_time 
                    svt_termination_time = svt_actual_start_time + svt_duration_sec
                    physio_pause = 0.1
                    resume_time  = svt_actual_start_time + svt_duration_sec + physio_pause
                    if resume_time < duration_sec:
                        heapq.heappush(
                          event_queue,
                          BeatEvent(resume_time, "sinus", "sa_node")
                        )
                    print(f"DEBUG: SVT Initiated by PAC. PAC time: {potential_event_time:.3f}, SVT Start: {svt_actual_start_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
                    print(f"DEBUG: SA P Wave Time at SVT init: {sa_node_last_actual_fire_time_for_p_wave:.3f}")
                    
                    event_queue = [e for e in event_queue if not (e.source == "sa_node" and e.time >= svt_actual_start_time and e.time < svt_termination_time)]
                    heapq.heapify(event_queue)

                    svt_rr = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
                    if svt_rr != float('inf'):
                        first_svt_beat_time = svt_actual_start_time + svt_rr 
                        if first_svt_beat_time < duration_sec and first_svt_beat_time < svt_termination_time:
                            heapq.heappush(event_queue, BeatEvent(first_svt_beat_time, "svt_beat", "svt_focus"))
            
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus: 
                if not is_vt_currently_active and not is_svt_currently_active:
                    coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    pr_for_next_sinus_after_pac = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
                    if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                        pr_for_next_sinus_after_pac = first_degree_av_block_pr_sec
                    next_potential_sa_qrs_after_pac_reset = sa_node_next_fire_time + pr_for_next_sinus_after_pac
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs_after_pac_reset - 0.100:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pvc":
            if not is_vt_currently_active and not is_svt_currently_active: # PVC logic only if not overridden by other tachy
                sinus_qrs_before_pvc_cycle_approx = last_placed_qrs_onset_time - actual_rr_to_this_beat
                if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                    end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle_approx + (2 * base_rr_interval_sec)
                    ventricle_ready_for_next_qrs_at_time = max(ventricle_ready_for_next_qrs_at_time, end_of_compensatory_pause_for_qrs - 0.02)

        elif is_escape_event: 
            if not is_vt_currently_active and not is_svt_currently_active:
                escape_rate_used = third_degree_escape_rate_bpm or \
                                (45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0)
                escape_rr_interval_sec = 60.0 / escape_rate_used if escape_rate_used > 0 else float('inf')
                if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
                    next_escape_fire_time = potential_event_time + escape_rr_interval_sec
                    if next_escape_fire_time < duration_sec:
                        heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, current_event.beat_type, current_event.source))

    if is_afib_active_base and not svt_actual_start_time and not vt_actual_start_time : 
        f_waves = generate_fibrillatory_waves(duration_sec, afib_fibrillation_wave_amplitude_mv, fs)
        full_ecg_signal_np += f_waves
    
    if is_svt_currently_active and svt_actual_start_time is not None and svt_termination_time is None: 
        svt_actual_end_time = duration_sec # SVT ran till end
    if is_vt_currently_active and vt_actual_start_time is not None and vt_calculated_termination_time is None:
        vt_actual_end_time = duration_sec # VT ran till end


    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    
    description_parts = []
    base_desc_set = False

    if vt_actual_start_time is not None and vt_actual_end_time is not None:
        vt_desc = f"Ventricular Tachycardia ({vt_rate_bpm}bpm) from {vt_actual_start_time:.1f}s to {vt_actual_end_time:.1f}s"
        underlying_rhythm_desc_pre_vt = f"Sinus Rhythm at {heart_rate_bpm}bpm" 
        av_block_sub_desc = []
        if not (vt_start_time_sec is not None and vt_start_time_sec < 0.1): 
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc: underlying_rhythm_desc_pre_vt += " with " + " & ".join(av_block_sub_desc)
        description_parts.append(f"{underlying_rhythm_desc_pre_vt} interrupted by an episode of {vt_desc}")
        base_desc_set = True
    
    elif svt_actual_start_time is not None and svt_actual_end_time is not None:
        svt_desc = f"SVT ({svt_rate_bpm}bpm) from {svt_actual_start_time:.1f}s to {svt_actual_end_time:.1f}s"
        underlying_rhythm_desc = f"Sinus Rhythm at {heart_rate_bpm}bpm"
        av_block_sub_desc = []
        if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
        elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
        elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
            av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc: underlying_rhythm_desc += " with " + " & ".join(av_block_sub_desc)
        description_parts.append(f"{underlying_rhythm_desc} with an episode of {svt_desc}")
        base_desc_set = True
    
    if not base_desc_set:
        if is_aflutter_active_base:
            description_parts.append(f"Atrial Flutter ({atrial_flutter_rate_bpm}bpm atrial) with {atrial_flutter_av_block_ratio_qrs_to_f}:1 AV Conduction")
        elif is_afib_active_base:
            description_parts.append(f"Atrial Fibrillation (Avg Ventricular Rate: {afib_average_ventricular_rate_bpm}bpm)")
        elif is_third_degree_block_active_base:
            escape_desc = f"{third_degree_escape_rhythm_origin.capitalize()} Escape ({(third_degree_escape_rate_bpm or (45 if third_degree_escape_rhythm_origin == 'junctional' else 30)):.0f}bpm)"
            description_parts.append(f"3rd Degree AV Block (Atrial Rate {heart_rate_bpm}bpm, Ventricular: {escape_desc})")
        else: 
            description_parts.append(f"Sinus Rhythm at {heart_rate_bpm}bpm")
            av_block_sub_desc = []
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
            if av_block_sub_desc: description_parts[-1] += " with " + " & ".join(av_block_sub_desc)

    ectopic_desc = []
    if enable_pac and pac_probability_per_sinus > 0 and \
       not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base) and \
       svt_actual_start_time is None and vt_actual_start_time is None : 
        ectopic_desc.append(f"PACs ({pac_probability_per_sinus*100:.0f}%)")
    
    if enable_pvc and pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"PVCs ({pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc:
        conjunction = " and "
        if description_parts:
            last_part = description_parts[-1]
            if "with" in last_part or "interrupted by" in last_part or "episode of" in last_part :
                conjunction = " and "
            elif not last_part.endswith(")") and not "with" in last_part:
                 conjunction = " with "
        elif not description_parts : 
             conjunction = "" 
             if len(ectopic_desc) > 1:
                 first_ectopic = ectopic_desc.pop(0)
                 description_parts.append(first_ectopic)
                 conjunction = " & " 
             else: 
                 description_parts.append(ectopic_desc[0])
                 ectopic_desc = [] 

        if description_parts and ectopic_desc: 
            description_parts[-1] += conjunction + " & ".join(ectopic_desc)
        elif not description_parts and ectopic_desc: 
             description_parts.append(" & ".join(ectopic_desc))

    final_description = " ".join(description_parts).replace("  ", " ").strip()
    final_description = final_description.replace(" with and ", " with ").replace(" and and ", " and ")
    if not final_description: final_description = "Simulated ECG Data"

    return full_time_axis_np.tolist(), full_ecg_signal_np.tolist(), final_description


def generate_physiologically_accurate_ecg_12_lead(
    heart_rate_bpm: float, duration_sec: float,
    enable_pvc: bool, pvc_probability_per_sinus: float,
    enable_pac: bool, pac_probability_per_sinus: float,
    first_degree_av_block_pr_sec: Optional[float],
    enable_mobitz_ii_av_block: bool, mobitz_ii_p_waves_per_qrs: int,
    enable_mobitz_i_wenckebach: bool, wenckebach_initial_pr_sec: float,
    wenckebach_pr_increment_sec: float, wenckebach_max_pr_before_drop_sec: float,
    enable_third_degree_av_block: bool, third_degree_escape_rhythm_origin: str,
    third_degree_escape_rate_bpm: Optional[float],
    enable_atrial_fibrillation: bool, afib_average_ventricular_rate_bpm: int,
    afib_fibrillation_wave_amplitude_mv: float, afib_irregularity_factor: float,
    enable_atrial_flutter: bool, atrial_flutter_rate_bpm: int,
    atrial_flutter_av_block_ratio_qrs_to_f: int, atrial_flutter_wave_amplitude_mv: float,
    allow_svt_initiation_by_pac: bool,
    svt_initiation_probability_after_pac: float,
    svt_duration_sec: float,
    svt_rate_bpm: int,
    enable_vt: bool,
    vt_start_time_sec: Optional[float],
    vt_duration_sec: float,
    vt_rate_bpm: int,
    fs: int, # fs is now explicitly passed
    # QT Correction Parameter
    target_qtc_ms: float = 400.0
):
    """
    Generates a 12-lead ECG using 3D cardiac vectors with full rhythm logic.
    """
    base_rr_interval_sec = 60.0 / heart_rate_bpm if heart_rate_bpm > 0 else float('inf')
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    
    full_cardiac_vectors = np.zeros((num_total_samples, 3)) # For 12-lead
    
    event_queue: List[BeatEvent] = []
    sa_node_next_fire_time = 0.0
    sa_node_last_actual_fire_time_for_p_wave = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    last_placed_qrs_onset_time = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    ventricle_ready_for_next_qrs_at_time = 0.0
    p_wave_counter_for_mobitz_ii = 0
    current_wenckebach_pr_sec = wenckebach_initial_pr_sec if enable_mobitz_i_wenckebach else None

    is_svt_currently_active: bool = False
    svt_termination_time: Optional[float] = None
    svt_actual_start_time: Optional[float] = None
    svt_actual_end_time: Optional[float] = None

    is_vt_currently_active: bool = False
    vt_actual_start_time: Optional[float] = None
    vt_calculated_termination_time: Optional[float] = None 
    vt_actual_end_time: Optional[float] = None
    
    # Rhythm hierarchy determination (copied from single lead)
    is_dynamic_svt_episode_configured = allow_svt_initiation_by_pac
    is_vt_episode_configured = enable_vt
    is_aflutter_active_base = enable_atrial_flutter and not is_dynamic_svt_episode_configured and not is_vt_episode_configured
    is_afib_active_base = enable_atrial_fibrillation and not is_dynamic_svt_episode_configured and not is_aflutter_active_base and not is_vt_episode_configured
    is_third_degree_block_active_base = enable_third_degree_av_block and not is_dynamic_svt_episode_configured and not is_afib_active_base and not is_aflutter_active_base and not is_vt_episode_configured
    is_mobitz_i_active_base = enable_mobitz_i_wenckebach and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_vt_episode_configured)
    is_mobitz_ii_active_base = enable_mobitz_ii_av_block and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_vt_episode_configured)
    is_first_degree_av_block_active_base = (first_degree_av_block_pr_sec is not None) and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_mobitz_ii_active_base or is_vt_episode_configured)
    
    flutter_wave_rr_interval_sec = 0.0
    flutter_wave_counter_for_av_block = 0

    # Initial Event Scheduling (copied from single lead)
    if is_aflutter_active_base:
        flutter_wave_rr_interval_sec = 60.0 / atrial_flutter_rate_bpm if atrial_flutter_rate_bpm > 0 else float('inf')
        if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(0.0, "flutter_wave", "aflutter_focus"))
    elif is_afib_active_base:
        mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
        if mean_afib_rr_sec > 0 and mean_afib_rr_sec != float('inf'):
            first_qrs_delay = np.random.uniform(0.1, mean_afib_rr_sec * 0.6) 
            heapq.heappush(event_queue, BeatEvent(first_qrs_delay, "afib_conducted", "afib_av_node"))
    elif is_third_degree_block_active_base:
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
             heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
        escape_beat_type = "junctional_escape" if third_degree_escape_rhythm_origin == "junctional" else "ventricular_escape"
        default_escape_rate = 45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0
        actual_escape_rate_bpm = third_degree_escape_rate_bpm or default_escape_rate
        escape_rr_interval_sec = 60.0 / actual_escape_rate_bpm if actual_escape_rate_bpm > 0 else float('inf')
        sinus_pr_for_offset = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
        if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
             sinus_pr_for_offset = first_degree_av_block_pr_sec
        first_escape_fire_time = max(0.1, sinus_pr_for_offset + np.random.uniform(0.05, 0.15))
        if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(first_escape_fire_time, escape_beat_type, f"{third_degree_escape_rhythm_origin}_escape"))
    elif not is_vt_episode_configured or (vt_start_time_sec is not None and vt_start_time_sec > 0.05): 
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    if is_vt_episode_configured:
        actual_vt_start_time_for_signal = vt_start_time_sec if vt_start_time_sec is not None else 0.05 
        if actual_vt_start_time_for_signal < duration_sec:
            heapq.heappush(event_queue, BeatEvent(actual_vt_start_time_for_signal, "vt_initiation_signal", "vt_control"))

    # --- Main Event Loop (Merged Logic) ---
    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_event_time = current_event.time
        
        # VT Initiation
        if current_event.beat_type == "vt_initiation_signal":
            if not is_vt_currently_active: 
                if is_svt_currently_active:
                    is_svt_currently_active = False 
                    svt_termination_time = None 
                    event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= potential_event_time)]
                    heapq.heapify(event_queue)
                is_vt_currently_active = True
                vt_actual_start_time = potential_event_time
                vt_calculated_termination_time = vt_actual_start_time + vt_duration_sec
                new_event_queue = []
                for e_val in event_queue:
                    is_sa_or_pac = e_val.source == "sa_node" or e_val.beat_type == "pac"
                    if e_val.time < vt_actual_start_time or \
                       (vt_calculated_termination_time is not None and e_val.time >= vt_calculated_termination_time) or \
                       not is_sa_or_pac: 
                        new_event_queue.append(e_val)
                event_queue = new_event_queue
                heapq.heapify(event_queue)
                vt_rr = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
                if vt_rr != float('inf'):
                    first_vt_beat_time = vt_actual_start_time 
                    if first_vt_beat_time < duration_sec and first_vt_beat_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                        heapq.heappush(event_queue, BeatEvent(first_vt_beat_time, "vt_beat", "vt_focus"))
                        physio_pause_after_vt = 0.15
                        resume_sinus_time = vt_actual_start_time + vt_duration_sec + physio_pause_after_vt
                        if resume_sinus_time < duration_sec:
                            heapq.heappush(event_queue, BeatEvent(resume_sinus_time, "sinus", "sa_node_resume_post_vt"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        # VT Termination
        if is_vt_currently_active and vt_calculated_termination_time is not None and potential_event_time >= vt_calculated_termination_time:
            vt_actual_end_time = vt_calculated_termination_time
            is_vt_currently_active = False
            vt_calculated_termination_time = None 
            event_queue = [e for e in event_queue if not (e.beat_type == "vt_beat" and e.time >= vt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            if current_event.source != "sa_node_resume_post_vt":
                if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                    time_since_last_sa_p_before_vt_era = vt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                    num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_vt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                    resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                    physiological_post_vt_pause = 0.15
                    sa_node_next_fire_time_after_vt = max(vt_actual_end_time + physiological_post_vt_pause, resumed_sa_fire_time)
                    if sa_node_next_fire_time_after_vt < duration_sec:
                        is_resume_already_scheduled = any(e.source == "sa_node_resume_post_vt" and abs(e.time - sa_node_next_fire_time_after_vt) < base_rr_interval_sec * 0.5 for e in event_queue)
                        if not is_resume_already_scheduled and not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_vt) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_vt, "sinus", "sa_node"))
            if current_event.beat_type == "vt_beat" and potential_event_time >= vt_actual_end_time - 0.001 : 
                if event_queue and event_queue[0].time < duration_sec: continue
                else: break
        
        # SVT Termination
        if is_svt_currently_active and svt_termination_time is not None and potential_event_time >= svt_termination_time:
            svt_actual_end_time = svt_termination_time 
            is_svt_currently_active = False 
            svt_termination_time = None
            event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= svt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            if current_event.source != "sa_node_resume_post_svt":
                if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                    time_since_last_sa_p_before_svt_era = svt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                    num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_svt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                    resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                    physiological_post_svt_pause = 0.1 
                    sa_node_next_fire_time_after_svt = max(svt_actual_end_time + physiological_post_svt_pause, resumed_sa_fire_time)
                    if sa_node_next_fire_time_after_svt < duration_sec:
                        is_resume_already_scheduled = any(e.source == "sa_node_resume_post_svt" and abs(e.time - sa_node_next_fire_time_after_svt) < base_rr_interval_sec * 0.5 for e in event_queue)
                        if not is_resume_already_scheduled and not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_svt) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_svt, "sinus", "sa_node"))
            if current_event.beat_type == "svt_beat" and potential_event_time >= svt_actual_end_time - 0.001 :
                if event_queue and event_queue[0].time < duration_sec: continue 
                else: break
        
        # Define event types
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac" or current_event.source.startswith("sa_node_resume")
        is_escape_event = current_event.source.endswith("_escape")
        is_afib_qrs_event = current_event.source == "afib_av_node"
        is_flutter_wave_event = current_event.beat_type == "flutter_wave"
        is_flutter_conducted_qrs_event = current_event.beat_type == "flutter_conducted_qrs"
        is_svt_beat_event_type = current_event.beat_type == "svt_beat"
        is_vt_beat_event_type = current_event.beat_type == "vt_beat"

        # Skip suppressed atrial events
        if (is_svt_currently_active or is_vt_currently_active) and is_atrial_origin_event and not current_event.source.startswith("sa_node_resume"):
            if current_event.source == "sa_node": 
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break
        if not (is_svt_currently_active or is_vt_currently_active) and \
           (is_afib_active_base or is_aflutter_active_base) and is_atrial_origin_event and not current_event.source.startswith("sa_node_resume"):
            if current_event.source == "sa_node":
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                 if not (is_afib_active_base or is_aflutter_active_base or is_third_degree_block_active_base): # Only schedule next SA if not overridden
                    if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                        if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break
        
        if current_event.source.startswith("sa_node_resume"): # Standardize source for morphology
            current_event.source = "sa_node"
            is_atrial_origin_event = True # Re-evaluate

        # Get beat parameters and determine AV block
        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        
        # Determine effective heart rate for QT correction based on current rhythm
        effective_heart_rate = heart_rate_bpm  # Default to sinus rate
        if is_vt_currently_active:
            effective_heart_rate = vt_rate_bpm
        elif is_svt_currently_active:
            effective_heart_rate = svt_rate_bpm
        elif enable_atrial_fibrillation and (is_afib_active_base or current_event.beat_type == "afib_conducted"):
            effective_heart_rate = afib_average_ventricular_rate_bpm
        elif enable_atrial_flutter and (is_aflutter_active_base or current_event.beat_type == "flutter_conducted_qrs"):
            # Calculate effective ventricular rate for flutter
            effective_heart_rate = atrial_flutter_rate_bpm / atrial_flutter_av_block_ratio_qrs_to_f
        
        # Apply Bazett's formula for rate-corrected QT intervals
        current_beat_morph_params = get_rate_corrected_intervals(effective_heart_rate, current_beat_morph_params)
        qrs_is_blocked_by_av_node = False
        draw_p_wave_only_for_this_atrial_event = False
        
        # Flutter wave specific processing for 3D vectors
        if is_flutter_wave_event:
            flutter_wave_params_local = BEAT_MORPHOLOGIES["flutter_wave"].copy()
            flutter_wave_params_local["p_amplitude"] = atrial_flutter_wave_amplitude_mv
            flutter_wave_params_local["p_duration"] = flutter_wave_rr_interval_sec
            _, flutter_vectors, _ = generate_single_beat_3d_vectors(flutter_wave_params_local, "flutter_wave", fs, is_flutter_wave_itself=True)
            if len(flutter_vectors) > 0:
                fw_start_time_global = potential_event_time
                fw_start_sample_idx = int(fw_start_time_global * fs)
                fw_end_sample_idx = min(fw_start_sample_idx + len(flutter_vectors), num_total_samples)
                fw_samples_to_copy = fw_end_sample_idx - fw_start_sample_idx
                if fw_samples_to_copy > 0 and fw_start_sample_idx < num_total_samples:
                    full_cardiac_vectors[fw_start_sample_idx:fw_end_sample_idx] += flutter_vectors[:fw_samples_to_copy]
            flutter_wave_counter_for_av_block += 1
            conducts_this_flutter_wave = (flutter_wave_counter_for_av_block % atrial_flutter_av_block_ratio_qrs_to_f == 0) if atrial_flutter_av_block_ratio_qrs_to_f > 0 else False
            if conducts_this_flutter_wave and potential_event_time >= ventricle_ready_for_next_qrs_at_time:
                 if not is_vt_currently_active and not is_svt_currently_active:
                    flutter_qrs_pr = BEAT_MORPHOLOGIES["flutter_conducted_qrs"]["pr_interval"] 
                    qrs_time_after_flutter = potential_event_time + flutter_qrs_pr 
                    heapq.heappush(event_queue, BeatEvent(qrs_time_after_flutter, "flutter_conducted_qrs", "aflutter_conducted"))
            if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_counter_for_av_block >= atrial_flutter_av_block_ratio_qrs_to_f:
                flutter_wave_counter_for_av_block = 0
            if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
                next_fw_time = potential_event_time + flutter_wave_rr_interval_sec
                if next_fw_time < duration_sec:
                    if not is_vt_currently_active and not is_svt_currently_active :
                         heapq.heappush(event_queue, BeatEvent(next_fw_time, "flutter_wave", "aflutter_focus"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        # AV Block Logic
        if not is_svt_currently_active and not is_vt_currently_active and \
           not (is_afib_active_base or is_aflutter_active_base) and \
           not is_svt_beat_event_type and not is_vt_beat_event_type and \
           not is_afib_qrs_event and not is_flutter_conducted_qrs_event and not is_escape_event:
            if is_atrial_origin_event: 
                 sa_node_last_actual_fire_time_for_p_wave = potential_event_time
            if is_third_degree_block_active_base and is_atrial_origin_event:
                qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec :
                    current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_mobitz_i_active_base and is_atrial_origin_event:
                if current_wenckebach_pr_sec is None: current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                current_beat_morph_params["pr_interval"] = current_wenckebach_pr_sec
                if current_wenckebach_pr_sec >= wenckebach_max_pr_before_drop_sec:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                    current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                else: current_wenckebach_pr_sec += wenckebach_pr_increment_sec
            elif is_mobitz_ii_active_base and is_atrial_origin_event:
                p_wave_counter_for_mobitz_ii += 1
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1 and mobitz_ii_p_waves_per_qrs > 1 :
                     qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs : p_wave_counter_for_mobitz_ii = 0
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                     current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_first_degree_av_block_active_base and is_atrial_origin_event and first_degree_av_block_pr_sec:
                current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        # Ventricular Refractory Period Block
        if not draw_p_wave_only_for_this_atrial_event and \
           not qrs_is_blocked_by_av_node and \
           not is_escape_event and not is_vt_beat_event_type and \
           potential_event_time < ventricle_ready_for_next_qrs_at_time:
            qrs_is_blocked_by_av_node = True
            if is_atrial_origin_event: draw_p_wave_only_for_this_atrial_event = True 
        
        # Process Blocked Atrial Event (Draw P only)
        if qrs_is_blocked_by_av_node:
            if draw_p_wave_only_for_this_atrial_event:
                _, p_vectors, p_wave_offset_for_drawing = generate_single_beat_3d_vectors(current_beat_morph_params, current_event.beat_type, fs, draw_only_p=True)
                if len(p_vectors) > 0:
                    p_wave_start_time_global = potential_event_time - p_wave_offset_for_drawing
                    p_start_sample_idx_global = int(p_wave_start_time_global * fs)
                    p_shape_start_idx, p_place_start_idx = 0, p_start_sample_idx_global
                    if p_place_start_idx < 0: p_shape_start_idx = -p_place_start_idx; p_place_start_idx = 0
                    p_samples_in_shape = len(p_vectors) - p_shape_start_idx
                    p_samples_in_signal = num_total_samples - p_place_start_idx
                    p_samples_to_copy = min(p_samples_in_shape, p_samples_in_signal)
                    if p_samples_to_copy > 0:
                        p_shape_end_idx = p_shape_start_idx + p_samples_to_copy; p_place_end_idx = p_place_start_idx + p_samples_to_copy
                        full_cardiac_vectors[p_place_start_idx : p_place_end_idx] += p_vectors[p_shape_start_idx : p_shape_end_idx]
            # Schedule next SA node if this was a blocked SA beat
            if current_event.source == "sa_node" and not is_svt_currently_active and not is_vt_currently_active and not (is_afib_active_base or is_aflutter_active_base):
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        # Process Conducted Beat (Full P-QRS-T)
        _, beat_vectors, qrs_offset_from_shape_start = generate_single_beat_3d_vectors(current_beat_morph_params, current_event.beat_type, fs, draw_only_p=False)
        if len(beat_vectors) > 0:
            waveform_start_time_global = potential_event_time - qrs_offset_from_shape_start
            start_sample_index_global = int(waveform_start_time_global * fs)
            shape_start_idx, place_start_idx = 0, start_sample_index_global
            if place_start_idx < 0: shape_start_idx = -place_start_idx; place_start_idx = 0
            samples_in_shape_remaining = len(beat_vectors) - shape_start_idx
            samples_in_signal_remaining = num_total_samples - place_start_idx
            samples_to_copy = min(samples_in_shape_remaining, samples_in_signal_remaining)
            if samples_to_copy > 0:
                shape_end_idx = shape_start_idx + samples_to_copy; place_end_idx = place_start_idx + samples_to_copy
                full_cardiac_vectors[place_start_idx : place_end_idx] += beat_vectors[shape_start_idx : shape_end_idx]

        actual_rr_to_this_beat = potential_event_time - last_placed_qrs_onset_time
        last_placed_qrs_onset_time = potential_event_time
        qrs_duration_this_beat = current_beat_morph_params.get('qrs_duration', 0.10)
        ventricle_ready_for_next_qrs_at_time = potential_event_time + max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat * 1.8 if qrs_duration_this_beat else MIN_REFRACTORY_PERIOD_SEC)

        # Schedule Next Events based on current conducted beat type
        if is_vt_beat_event_type and is_vt_currently_active:
            vt_rr_interval_sec = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
            next_vt_event_time = potential_event_time + vt_rr_interval_sec
            if vt_rr_interval_sec != float('inf') and next_vt_event_time < duration_sec and \
               next_vt_event_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_vt_event_time, "vt_beat", "vt_focus"))
        elif is_svt_beat_event_type and is_svt_currently_active:
            svt_rr_interval_sec = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
            next_svt_event_time = potential_event_time + svt_rr_interval_sec
            if svt_rr_interval_sec != float('inf') and next_svt_event_time < duration_sec and \
               next_svt_event_time < (svt_termination_time if svt_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_svt_event_time, "svt_beat", "svt_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else svt_rr_interval_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > ventricle_ready_for_next_qrs_at_time and \
                   pvc_time < next_svt_event_time - 0.100 and \
                   (svt_termination_time is None or pvc_time < svt_termination_time - 0.100) :
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif is_afib_qrs_event: 
            mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
            std_dev_rr = mean_afib_rr_sec * afib_irregularity_factor
            next_rr_variation = np.random.normal(0, std_dev_rr)
            tentative_next_rr = mean_afib_rr_sec + next_rr_variation
            min_physiological_rr = max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat ) + 0.05 
            next_rr = max(min_physiological_rr, tentative_next_rr)
            next_afib_qrs_event_time = potential_event_time + next_rr
            if mean_afib_rr_sec != float('inf') and next_afib_qrs_event_time < duration_sec:
                if not is_vt_currently_active and not is_svt_currently_active :
                    heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > ventricle_ready_for_next_qrs_at_time and pvc_time < next_afib_qrs_event_time - 0.100 : 
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif is_flutter_conducted_qrs_event: 
             if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    ventricular_rr_in_flutter = (flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f) if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_rr_interval_sec > 0 else float('inf')
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > ventricle_ready_for_next_qrs_at_time: 
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.source == "sa_node":
            sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active and not is_vt_currently_active: 
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active:
                    pac_time = potential_event_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                    if pac_time > potential_event_time + 0.100 and \
                       (sa_node_next_fire_time == float('inf') or pac_time < sa_node_next_fire_time - 0.100): 
                        heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active:
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    pr_interval_for_next_sinus = current_beat_morph_params.get('pr_interval', BEAT_MORPHOLOGIES["sinus"]['pr_interval'])
                    next_potential_sa_qrs = sa_node_next_fire_time + pr_interval_for_next_sinus if sa_node_next_fire_time != float('inf') else float('inf')
                    if pvc_time > ventricle_ready_for_next_qrs_at_time and \
                       (next_potential_sa_qrs == float('inf') or pvc_time < next_potential_sa_qrs - 0.100):
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.beat_type == "pac":
            sa_node_next_fire_time = potential_event_time + base_rr_interval_sec
            new_event_queue_pac_reset = [e for e in event_queue if not (e.source == "sa_node" and e.time >= potential_event_time)]
            heapq.heapify(new_event_queue_pac_reset); event_queue = new_event_queue_pac_reset
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active and not is_vt_currently_active: 
                    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if is_dynamic_svt_episode_configured and not is_svt_currently_active and not is_vt_currently_active and \
               not is_afib_active_base and not is_aflutter_active_base and not is_third_degree_block_active_base:
                if np.random.rand() < svt_initiation_probability_after_pac:
                    is_svt_currently_active = True
                    svt_actual_start_time = potential_event_time
                    svt_termination_time = svt_actual_start_time + svt_duration_sec
                    event_queue = [e for e in event_queue if not (e.source == "sa_node" and e.time >= svt_actual_start_time and (svt_termination_time is None or e.time < svt_termination_time))]
                    heapq.heapify(event_queue)
                    svt_rr = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
                    if svt_rr != float('inf'):
                        first_svt_beat_time = svt_actual_start_time + svt_rr 
                        if first_svt_beat_time < duration_sec and (svt_termination_time is None or first_svt_beat_time < svt_termination_time):
                            heapq.heappush(event_queue, BeatEvent(first_svt_beat_time, "svt_beat", "svt_focus"))
                    physio_pause_after_svt = 0.1
                    resume_sinus_time_svt = svt_actual_start_time + svt_duration_sec + physio_pause_after_svt
                    if resume_sinus_time_svt < duration_sec:
                        heapq.heappush(event_queue, BeatEvent(resume_sinus_time_svt, "sinus", "sa_node_resume_post_svt"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus: 
                if not is_vt_currently_active and not is_svt_currently_active:
                    coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    pr_for_next_sinus_after_pac = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
                    if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                        pr_for_next_sinus_after_pac = first_degree_av_block_pr_sec
                    next_potential_sa_qrs_after_pac_reset = sa_node_next_fire_time + pr_for_next_sinus_after_pac if sa_node_next_fire_time != float('inf') else float('inf')
                    if pvc_time > ventricle_ready_for_next_qrs_at_time and \
                       (next_potential_sa_qrs_after_pac_reset == float('inf') or pvc_time < next_potential_sa_qrs_after_pac_reset - 0.100):
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.beat_type == "pvc":
            pass # Compensatory pause handled by refractory period and SA node timing
        elif is_escape_event: 
            if not is_vt_currently_active and not is_svt_currently_active:
                escape_rate_used = third_degree_escape_rate_bpm or \
                                (45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0)
                escape_rr_interval_sec = 60.0 / escape_rate_used if escape_rate_used > 0 else float('inf')
                if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
                    next_escape_fire_time = potential_event_time + escape_rr_interval_sec
                    if next_escape_fire_time < duration_sec:
                        heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, current_event.beat_type, current_event.source))

    # Add AFib fibrillatory waves (3D) if active
    if is_afib_active_base and not svt_actual_start_time and not vt_actual_start_time: 
        f_waves_1d = generate_fibrillatory_waves(duration_sec, afib_fibrillation_wave_amplitude_mv, fs)
        # Define a representative 3D direction for f-waves (can be refined)
        afib_direction = np.array([0.1, 0.7, 0.2]) # Example: somewhat inferior and slightly anterior/left
        norm = np.linalg.norm(afib_direction)
        if norm > 0: afib_direction /= norm
        
        f_waves_3d = np.outer(f_waves_1d, afib_direction)
        # Ensure lengths match before adding
        len_diff = len(full_cardiac_vectors) - len(f_waves_3d)
        if len_diff == 0:
            full_cardiac_vectors += f_waves_3d
        elif len_diff > 0: # full_cardiac_vectors is longer
            full_cardiac_vectors[:len(f_waves_3d)] += f_waves_3d
        else: # f_waves_3d is longer
            full_cardiac_vectors += f_waves_3d[:len(full_cardiac_vectors)]


    # Handle tachycardias running to the end
    if is_svt_currently_active and svt_actual_start_time is not None and svt_termination_time is None: 
        svt_actual_end_time = duration_sec 
    if is_vt_currently_active and vt_actual_start_time is not None and vt_calculated_termination_time is None:
        vt_actual_end_time = duration_sec

    # Add baseline noise to 3D vectors
    noise_amplitude = 0.02 # mV
    noise_3d = noise_amplitude * np.random.normal(0, 1, full_cardiac_vectors.shape)
    full_cardiac_vectors += noise_3d
    
    # Project 3D vectors to 12 leads
    twelve_lead_signals = project_cardiac_vector_to_12_leads(full_cardiac_vectors)
    
    # --- Generate Rhythm Description (Copied and adapted from single lead) ---
    description_parts = []
    base_desc_set = False # Flag to ensure base rhythm description is set once

    # Handle VT description first as it's a dominant event
    if vt_actual_start_time is not None and vt_actual_end_time is not None:
        vt_desc = f"Ventricular Tachycardia ({vt_rate_bpm}bpm) from {vt_actual_start_time:.1f}s to {vt_actual_end_time:.1f}s"
        # Describe underlying rhythm if VT doesn't start at t=0
        underlying_rhythm_desc_pre_vt = f"Sinus Rhythm at {heart_rate_bpm}bpm" 
        av_block_sub_desc = []
        # Only add AV block details if VT doesn't start immediately (allowing underlying rhythm to be seen)
        if not (vt_start_time_sec is not None and vt_start_time_sec < 0.1): # Arbitrary threshold for "immediate"
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        
        if av_block_sub_desc: underlying_rhythm_desc_pre_vt += " with " + " & ".join(av_block_sub_desc)
        
        if vt_actual_start_time > 0.1: # If VT is not from the very beginning
             description_parts.append(f"{underlying_rhythm_desc_pre_vt} interrupted by an episode of {vt_desc}")
        else: # VT from the start
            description_parts.append(vt_desc)
        base_desc_set = True
    
    # Handle SVT if no VT or if SVT is the primary event described
    elif svt_actual_start_time is not None and svt_actual_end_time is not None:
        svt_desc = f"SVT ({svt_rate_bpm}bpm) from {svt_actual_start_time:.1f}s to {svt_actual_end_time:.1f}s"
        underlying_rhythm_desc = f"Sinus Rhythm at {heart_rate_bpm}bpm"
        av_block_sub_desc = []
        if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
        elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
        elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
            av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc: underlying_rhythm_desc += " with " + " & ".join(av_block_sub_desc)
        
        if svt_actual_start_time > 0.1 : # If SVT is not from the very beginning
            description_parts.append(f"{underlying_rhythm_desc} with an episode of {svt_desc}")
        else: # SVT from the start
             description_parts.append(svt_desc)
        base_desc_set = True
    
    # If no VT or SVT episodes dominated, describe the base rhythm
    if not base_desc_set:
        if is_aflutter_active_base:
            description_parts.append(f"Atrial Flutter ({atrial_flutter_rate_bpm}bpm atrial) with {atrial_flutter_av_block_ratio_qrs_to_f}:1 AV Conduction")
        elif is_afib_active_base:
            description_parts.append(f"Atrial Fibrillation (Avg Ventricular Rate: {afib_average_ventricular_rate_bpm}bpm)")
        elif is_third_degree_block_active_base:
            escape_desc = f"{third_degree_escape_rhythm_origin.capitalize()} Escape ({(third_degree_escape_rate_bpm or (45 if third_degree_escape_rhythm_origin == 'junctional' else 30)):.0f}bpm)"
            description_parts.append(f"3rd Degree AV Block (Atrial Rate {heart_rate_bpm}bpm, Ventricular: {escape_desc})")
        else: # Default to Sinus Rhythm with potential AV blocks
            description_parts.append(f"Sinus Rhythm at {heart_rate_bpm}bpm")
            av_block_sub_desc = []
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
            if av_block_sub_desc: description_parts[-1] += " with " + " & ".join(av_block_sub_desc)

    # Add ectopic descriptions
    ectopic_desc_parts = []
    if enable_pac and pac_probability_per_sinus > 0 and \
       not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base) and \
       svt_actual_start_time is None and vt_actual_start_time is None : # Only add PACs if not in these overriding rhythms or tachy
        ectopic_desc_parts.append(f"PACs ({pac_probability_per_sinus*100:.0f}%)")
    
    if enable_pvc and pvc_probability_per_sinus > 0:
        # Could add logic here to check if any PVCs were actually generated if desired for more accuracy
        ectopic_desc_parts.append(f"PVCs ({pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc_parts:
        conjunction = " and " # Default conjunction if description already exists
        if description_parts:
            last_part = description_parts[-1]
            # If the last part already has a 'with' or is a complex tachycardia description
            if "with" in last_part or "interrupted by" in last_part or "episode of" in last_part or "Ventricular Tachycardia" in last_part or "SVT" in last_part:
                conjunction = " and " # Keep using 'and'
            elif not last_part.endswith(")") and not "with" in last_part : # For simple rhythm descriptions
                 conjunction = " with "
        elif not description_parts : # No base rhythm description yet (shouldn't happen with current logic)
             conjunction = "" # Start fresh
             if len(ectopic_desc_parts) > 1:
                 first_ectopic = ectopic_desc_parts.pop(0)
                 description_parts.append(first_ectopic)
                 conjunction = " & " 
             else: 
                 description_parts.append(ectopic_desc_parts[0])
                 ectopic_desc_parts = [] # Clear it as it's now in description_parts

        # Append the ectopic descriptions
        if description_parts and ectopic_desc_parts: 
            description_parts[-1] += conjunction + " & ".join(ectopic_desc_parts)
        elif not description_parts and ectopic_desc_parts: # If somehow no base description, just list ectopics
             description_parts.append(" & ".join(ectopic_desc_parts))


    final_description = " ".join(description_parts).replace("  ", " ").strip()
    # Clean up potential double conjunctions like "with and" or "and and"
    final_description = final_description.replace(" with and ", " with ").replace(" and and ", " and ")
    if not final_description: 
        final_description = "12-Lead ECG Simulation"

    return full_time_axis_np.tolist(), twelve_lead_signals, final_description