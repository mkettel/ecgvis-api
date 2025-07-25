#!/usr/bin/env python3
"""
Debug the cardiac vector projection to understand amplitude issues.
"""
import sys
sys.path.append('.')
import numpy as np

from ecg_simulator.beat_generation import generate_single_beat_3d_vectors
from ecg_simulator.full_ecg.vector_projection import project_cardiac_vector_to_12_leads
from ecg_simulator.constants import SINUS_PARAMS, FS

def debug_cardiac_projection():
    """Debug the cardiac vector projection in detail."""
    
    # Generate a sinus beat  
    t_relative, cardiac_vectors, qrs_offset = generate_single_beat_3d_vectors(
        params=SINUS_PARAMS,
        beat_type="sinus", 
        fs=FS
    )
    
    print(f"=== Cardiac Vector Analysis ===")
    print(f"Total samples: {len(cardiac_vectors)}")
    print(f"QRS offset: {qrs_offset:.3f}s (sample {int(qrs_offset * FS)})")
    
    # Find QRS region
    qrs_start_idx = int(qrs_offset * FS)
    qrs_duration_samples = int(SINUS_PARAMS['qrs_duration'] * FS)  # 0.1s * 250 = 25 samples
    qrs_end_idx = qrs_start_idx + qrs_duration_samples
    
    print(f"QRS region: samples {qrs_start_idx} to {qrs_end_idx}")
    
    # Analyze cardiac vectors in QRS region
    qrs_vectors = cardiac_vectors[qrs_start_idx:qrs_end_idx] if qrs_end_idx <= len(cardiac_vectors) else cardiac_vectors[qrs_start_idx:]
    
    if len(qrs_vectors) == 0:
        print("ERROR: No QRS vectors found!")
        return
        
    print(f"QRS vectors shape: {qrs_vectors.shape}")
    
    # Debug: Print some QRS vectors to see what's happening
    print("First 5 QRS vectors:")
    for i in range(min(5, len(qrs_vectors))):
        vec = qrs_vectors[i]
        mag = np.linalg.norm(vec)
        print(f"  QRS[{i}]: [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}] |V|={mag:.3f}")
    
    # Also check the entire cardiac_vectors array for any non-zero vectors
    all_magnitudes = np.linalg.norm(cardiac_vectors, axis=1)
    non_zero_indices = np.where(all_magnitudes > 0.001)[0]
    
    print(f"\nNon-zero cardiac vectors found at indices: {non_zero_indices[:10]}...")  # Show first 10
    print(f"Total non-zero vectors: {len(non_zero_indices)}")
    
    if len(non_zero_indices) > 0:
        global_peak_idx = np.argmax(all_magnitudes)
        global_peak_mag = all_magnitudes[global_peak_idx]
        global_peak_vec = cardiac_vectors[global_peak_idx]
        print(f"Global peak at index {global_peak_idx}: [{global_peak_vec[0]:.3f}, {global_peak_vec[1]:.3f}, {global_peak_vec[2]:.3f}] |V|={global_peak_mag:.3f}")
    
    # Find peak cardiac vector magnitude in QRS
    qrs_magnitudes = np.linalg.norm(qrs_vectors, axis=1)
    peak_idx = np.argmax(qrs_magnitudes)
    peak_magnitude = qrs_magnitudes[peak_idx]
    peak_vector = qrs_vectors[peak_idx]
    
    print(f"Peak cardiac vector: [{peak_vector[0]:.3f}, {peak_vector[1]:.3f}, {peak_vector[2]:.3f}]")
    print(f"Peak magnitude: {peak_magnitude:.3f} mV")
    
    # Project to 12-lead
    ecg_signals = project_cardiac_vector_to_12_leads(cardiac_vectors)
    
    print(f"\n=== ECG Signal Analysis ===")
    
    # Check QRS region in projected signals
    key_leads = ['V1', 'I', 'V5']
    
    for lead in key_leads:
        signal = ecg_signals[lead]
        qrs_signal = signal[qrs_start_idx:qrs_end_idx] if qrs_end_idx <= len(signal) else signal[qrs_start_idx:]
        
        full_min = np.min(signal)
        full_max = np.max(signal) 
        qrs_min = np.min(qrs_signal) if len(qrs_signal) > 0 else 0
        qrs_max = np.max(qrs_signal) if len(qrs_signal) > 0 else 0
        
        print(f"{lead:3}: Full signal [{full_min:.3f}, {full_max:.3f}], QRS region [{qrs_min:.3f}, {qrs_max:.3f}]")
        
        # Manual projection check for peak vector
        from ecg_simulator.full_ecg.vector_projection import LEAD_VECTOR_DIRECTIONS
        lead_vector = LEAD_VECTOR_DIRECTIONS[lead]
        manual_projection = np.dot(peak_vector, lead_vector)
        print(f"     Manual peak projection: {manual_projection:.3f} mV")
    
    print(f"\n=== Timing Analysis ===")
    print(f"Beat duration: {len(t_relative)/FS:.3f}s")
    print(f"P wave duration: {SINUS_PARAMS['p_duration']:.3f}s") 
    print(f"PR interval: {SINUS_PARAMS['pr_interval']:.3f}s")
    print(f"QRS duration: {SINUS_PARAMS['qrs_duration']:.3f}s")
    
    # Check if the beat is truncated
    expected_duration = (SINUS_PARAMS['pr_interval'] + SINUS_PARAMS['qrs_duration'] + 
                        SINUS_PARAMS['st_duration'] + SINUS_PARAMS['t_duration'] + 0.05)
    actual_duration = len(t_relative) / FS
    
    print(f"Expected duration: {expected_duration:.3f}s")
    print(f"Actual duration: {actual_duration:.3f}s")
    
    if actual_duration < expected_duration * 0.9:
        print("⚠️  Beat appears to be truncated!")

if __name__ == "__main__":
    debug_cardiac_projection()