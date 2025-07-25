#!/usr/bin/env python3
"""
Debug V1 projection to understand why we're not getting deep S waves.
"""
import numpy as np
import sys
sys.path.append('.')

from ecg_simulator.beat_generation import calculate_instantaneous_qrs_vector
from ecg_simulator.full_ecg.vector_projection import LEAD_VECTOR_DIRECTIONS

def debug_v1_projection():
    """Debug detailed V1 projection during QRS evolution."""
    
    print("=== V1 Lead Vector Analysis ===")
    v1_lead = LEAD_VECTOR_DIRECTIONS["V1"]
    print(f"V1 lead vector: [{v1_lead[0]:.3f}, {v1_lead[1]:.3f}, {v1_lead[2]:.3f}]")
    print(f"V1 emphasizes: X={v1_lead[0]:.3f} (right-left), Y={v1_lead[1]:.3f} (sup-inf), Z={v1_lead[2]:.3f} (post-ant)")
    print()
    
    qrs_amplitudes = {
        'q_amplitude': -0.08,
        'r_amplitude': 1.4,
        's_amplitude': -0.6
    }
    
    print("=== QRS Phase Analysis for V1 ===")
    print("Time\tCardiac Vector\t\t\t\tV1 Projection")
    print("-" * 65)
    
    # Test key phases
    test_times = [0.15, 0.50, 0.85]  # Septal peak, free wall peak, basal peak
    phase_names = ["Septal", "Free Wall", "Basal"]
    
    for i, (time, phase) in enumerate(zip(test_times, phase_names)):
        direction, magnitude = calculate_instantaneous_qrs_vector(time, qrs_amplitudes)
        cardiac_vector = direction * magnitude
        
        # Calculate V1 projection
        v1_projection = np.dot(cardiac_vector, v1_lead)
        
        print(f"{time:.2f}\t[{cardiac_vector[0]:6.3f}, {cardiac_vector[1]:6.3f}, {cardiac_vector[2]:6.3f}]\t{v1_projection:7.3f}mV ({phase})")
        
        # Break down the contribution
        x_contrib = cardiac_vector[0] * v1_lead[0]
        y_contrib = cardiac_vector[1] * v1_lead[1] 
        z_contrib = cardiac_vector[2] * v1_lead[2]
        
        print(f"\t  X contrib: {cardiac_vector[0]:.3f} * {v1_lead[0]:.3f} = {x_contrib:.3f}")
        print(f"\t  Y contrib: {cardiac_vector[1]:.3f} * {v1_lead[1]:.3f} = {y_contrib:.3f}")
        print(f"\t  Z contrib: {cardiac_vector[2]:.3f} * {v1_lead[2]:.3f} = {z_contrib:.3f}")
        print(f"\t  Total: {x_contrib:.3f} + {y_contrib:.3f} + {z_contrib:.3f} = {v1_projection:.3f}")
        print()
    
    print("=== Expected V1 Pattern ===")
    print("1. Septal phase: Small positive r wave (leftward septal creates small + in V1)")
    print("2. Free wall phase: Continues positive (LV free wall moving away, but anterior component dominates)")
    print("3. Basal phase: Deep negative S wave (posterior movement creates strong negative in V1)")
    print()
    print("For deep S wave: Need strong negative Z component in basal phase")
    print(f"V1 Z weight = {v1_lead[2]:.3f}, so need cardiac Z < -0.3 for deep S wave")

if __name__ == "__main__":
    debug_v1_projection()