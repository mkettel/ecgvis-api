#!/usr/bin/env python3
"""
Test script to validate physiological QRS vector generation.
"""
import numpy as np
import sys
sys.path.append('.')

from ecg_simulator.beat_generation import calculate_instantaneous_qrs_vector
from ecg_simulator.full_ecg.vector_projection import project_cardiac_vector_to_12_leads

def test_qrs_physiology():
    """Test that our QRS vector evolution produces expected ECG patterns."""
    
    print("=== Testing Physiological QRS Vector Evolution ===\n")
    
    # Standard QRS amplitudes for testing
    qrs_amplitudes = {
        'q_amplitude': -0.08,  # Small Q waves
        'r_amplitude': 1.4,    # Dominant R waves  
        's_amplitude': -0.6    # Deeper S waves for proper V1 pattern
    }
    
    # Test key time points during QRS
    test_points = [
        (0.1, "Early septal"),
        (0.2, "Peak septal"),
        (0.4, "Early free wall"),
        (0.6, "Peak free wall"),
        (0.8, "Late free wall"), 
        (0.9, "Early basal"),
        (1.0, "Peak basal")
    ]
    
    print("QRS Evolution Analysis:")
    print("Time\tPhase\t\tVector Direction\t\t\tMagnitude")
    print("-" * 70)
    
    # Track vectors for projection analysis
    all_vectors = []
    all_progress = []
    
    for progress, phase_name in test_points:
        direction, magnitude = calculate_instantaneous_qrs_vector(progress, qrs_amplitudes)
        all_vectors.append(direction * magnitude)
        all_progress.append(progress)
        
        print(f"{progress:.1f}\t{phase_name:12}\t[{direction[0]:5.2f}, {direction[1]:5.2f}, {direction[2]:5.2f}]\t{magnitude:6.3f}")
    
    # Project to 12-lead ECG to validate patterns
    print("\n=== Projected ECG Lead Analysis ===")
    print("Expected patterns:")
    print("- V1: Small r wave (septal), then deep S wave (free wall + basal)")
    print("- Lead I: Small Q wave (septal), then dominant R wave (free wall)")
    print("- aVF: Similar to Lead I but more inferior emphasis")
    
    # Convert to array for projection
    vectors_array = np.array(all_vectors)
    projected_leads = project_cardiac_vector_to_12_leads(vectors_array)
    
    print("\nActual projected amplitudes:")
    key_leads = ['V1', 'I', 'aVF', 'V5']
    
    for lead in key_leads:
        signal = projected_leads[lead]
        min_val = np.min(signal)
        max_val = np.max(signal)
        print(f"{lead:3}: Min={min_val:6.3f}mV, Max={max_val:6.3f}mV, Range={max_val-min_val:6.3f}mV")
        
        # Check for expected patterns
        if lead == 'V1':
            has_small_r = max_val > 0.1 and max_val < 0.5
            has_deep_s = min_val < -0.3
            print(f"     V1 Pattern: Small r wave={'✓' if has_small_r else '✗'}, Deep S wave={'✓' if has_deep_s else '✗'}")
            
        elif lead in ['I', 'V5']:
            has_small_q = min_val < -0.02 and min_val > -0.15
            has_dominant_r = max_val > 0.8
            print(f"     {lead} Pattern: Small Q wave={'✓' if has_small_q else '✗'}, Dominant R wave={'✓' if has_dominant_r else '✗'}")
    
    print("\n=== Summary ===")
    print("This analysis shows how the instantaneous cardiac vector evolves during QRS")
    print("and whether it projects correctly to create realistic ECG morphology.")
    
if __name__ == "__main__":
    test_qrs_physiology()