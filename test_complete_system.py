#!/usr/bin/env python3
"""
Test the complete new physiological QRS system end-to-end.
"""
import sys
sys.path.append('.')

from ecg_simulator.beat_generation import generate_single_beat_3d_vectors
from ecg_simulator.full_ecg.vector_projection import project_cardiac_vector_to_12_leads
from ecg_simulator.constants import SINUS_PARAMS, FS

def test_complete_physiological_system():
    """Test the complete new physiological QRS system."""
    
    print("=== Complete Physiological QRS System Test ===\n")
    
    # Generate a sinus beat using the new system
    t_relative, cardiac_vectors, qrs_offset = generate_single_beat_3d_vectors(
        params=SINUS_PARAMS,
        beat_type="sinus",
        fs=FS,
        draw_only_p=False
    )
    
    print(f"Generated beat: {len(t_relative)} samples, {len(cardiac_vectors)} cardiac vectors")
    print(f"Duration: {len(t_relative)/FS:.3f} seconds")
    print(f"QRS offset: {qrs_offset:.3f} seconds")
    
    # Project to 12-lead ECG
    ecg_signals = project_cardiac_vector_to_12_leads(cardiac_vectors)
    
    print("\n=== 12-Lead ECG Analysis ===")
    
    # Analyze key leads for expected patterns
    key_leads = ['V1', 'I', 'II', 'aVL', 'aVF', 'V5', 'V6']
    
    for lead_name in key_leads:
        signal = ecg_signals[lead_name]
        min_val = min(signal)
        max_val = max(signal)
        range_val = max_val - min_val
        
        print(f"{lead_name:3}: Range {min_val:6.3f} to {max_val:6.3f} mV (span: {range_val:.3f} mV)")
        
        # Check for expected patterns
        if lead_name == 'V1':
            has_r_wave = max_val > 0.1
            has_s_wave = min_val < -0.3
            pattern = "rS" if has_r_wave and has_s_wave else "Other"
            print(f"     V1 Pattern: {pattern} ({'✓' if has_r_wave and has_s_wave else '✗'})")
            
        elif lead_name in ['I', 'aVL', 'V5', 'V6']:  # Lateral leads
            has_q_wave = min_val < -0.02
            has_r_wave = max_val > 0.8
            pattern = "qR" if has_q_wave and has_r_wave else "R" if has_r_wave else "Other"
            print(f"     {lead_name} Pattern: {pattern} (Q:{'✓' if has_q_wave else '✗'}, R:{'✓' if has_r_wave else '✗'})")
            
        elif lead_name in ['II', 'aVF']:  # Inferior leads
            has_q_wave = min_val < -0.02
            has_r_wave = max_val > 0.8
            pattern = "qR" if has_q_wave and has_r_wave else "R" if has_r_wave else "Other"
            print(f"     {lead_name} Pattern: {pattern} (Q:{'✓' if has_q_wave else '✗'}, R:{'✓' if has_r_wave else '✗'})")
    
    print("\n=== Expected vs Actual Patterns ===")
    
    # Summary of expected patterns for normal sinus rhythm
    expected_patterns = {
        'V1': 'rS pattern (small r, deep S)',
        'I': 'qR pattern (small Q, tall R)', 
        'II': 'qR pattern (small Q, tall R)',
        'aVL': 'qR or R pattern',
        'aVF': 'qR pattern (small Q, tall R)', 
        'V5': 'qR pattern (small Q, tall R)',
        'V6': 'qR pattern (small Q, tall R)'
    }
    
    print("Expected patterns for normal sinus rhythm:")
    for lead, pattern in expected_patterns.items():
        print(f"  {lead:3}: {pattern}")
    
    print("\n=== System Performance Summary ===")
    
    # Check overall success
    v1_signal = ecg_signals['V1']
    v1_rs_success = max(v1_signal) > 0.1 and min(v1_signal) < -0.3
    
    lateral_leads = ['I', 'V5', 'V6']
    lateral_success = all(max(ecg_signals[lead]) > 0.8 for lead in lateral_leads)
    
    q_wave_count = sum(1 for lead in ['I', 'II', 'aVL', 'aVF', 'V5', 'V6'] if min(ecg_signals[lead]) < -0.02)
    
    print(f"✓ V1 rS pattern: {'PASS' if v1_rs_success else 'FAIL'}")
    print(f"✓ Lateral R waves: {'PASS' if lateral_success else 'FAIL'}")
    print(f"✓ Q waves present: {q_wave_count}/6 leads")
    
    overall_success = v1_rs_success and lateral_success and q_wave_count >= 3
    print(f"\n🎉 Overall System: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print("The new physiological QRS system is generating realistic ECG patterns!")
    else:
        print("The system has improved but still needs fine-tuning for optimal patterns.")

if __name__ == "__main__":
    test_complete_physiological_system()