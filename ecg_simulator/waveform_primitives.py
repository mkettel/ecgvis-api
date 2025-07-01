# // backend/ecg-simulator/waveform_primitives.py
import numpy as np
from .constants import FS, BASELINE_MV, VFIB_PARAMS

# --- Waveform Primitive & Single Beat Generation ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    if width_std_dev <= 1e-9: return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

# --- Fibrillatory Wave Generation for AFib ---
def generate_fibrillatory_waves(duration_sec: float, amplitude_mv: float, fs: int = FS):
    num_samples = int(duration_sec * fs)
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
    f_wave_signal = np.zeros(num_samples)
    if amplitude_mv <= 1e-6: return f_wave_signal
    num_f_waves_per_sec = np.random.uniform(5.8, 10)
    f_wave_component_duration = np.random.uniform(0.05, 0.08)
    current_time = 0.0
    while current_time < duration_sec:
        center = current_time + np.random.uniform(-f_wave_component_duration / 5, f_wave_component_duration / 5)
        amp = amplitude_mv * np.random.uniform(0.6, 1.4) * np.random.choice([-1, 1])
        width_std = f_wave_component_duration / np.random.uniform(3.5, 5.5)
        start_time_comp = center - 3 * width_std
        end_time_comp = center + 3 * width_std
        start_idx = max(0, int(np.floor(start_time_comp * fs)))
        end_idx = min(num_samples, int(np.ceil(end_time_comp * fs)))
        if start_idx < end_idx:
            t_points_fwave_comp = time_axis[start_idx:end_idx]
            if t_points_fwave_comp.size > 0 :
                 f_wave_signal[start_idx:end_idx] += gaussian_wave(t_points_fwave_comp, center, amp, width_std)
        current_time += (1.0 / num_f_waves_per_sec) * np.random.uniform(0.5, 1.5)
    return f_wave_signal


def generate_vfib_waveform(duration_sec: float, fs: int = FS):
    """
    Generate a realistic Ventricular Fibrillation waveform with chaotic, irregular patterns.
    
    Args:
        duration_sec: Duration of VFib episode in seconds
        fs: Sampling frequency in Hz
        
    Returns:
        np.ndarray: VFib waveform signal
    """
    num_samples = int(duration_sec * fs)
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
    vfib_signal = np.zeros(num_samples)
    
    if duration_sec <= 0:
        return vfib_signal
    
    # VFib characteristics from constants
    base_amplitude = VFIB_PARAMS["base_amplitude"]
    amplitude_variation = VFIB_PARAMS["amplitude_variation"]
    freq_min, freq_max = VFIB_PARAMS["frequency_range"]
    chaos_factor = VFIB_PARAMS["chaos_factor"]
    
    # Generate multiple overlapping chaotic waveforms
    num_components = np.random.randint(3, 7)  # 3-6 chaotic components
    
    for _ in range(num_components):
        # Random frequency within VFib range (4-10 Hz typically)
        frequency = np.random.uniform(freq_min, freq_max)
        
        # Chaotic amplitude modulation
        amplitude_envelope = base_amplitude * (1 + amplitude_variation * np.random.uniform(-1, 1, num_samples))
        
        # Generate base sinusoidal component
        phase = np.random.uniform(0, 2*np.pi)
        component = amplitude_envelope * np.sin(2 * np.pi * frequency * time_axis + phase)
        
        # Add chaos through frequency modulation
        freq_modulation = chaos_factor * np.random.normal(0, frequency * 0.1, num_samples)
        cumulative_phase = np.cumsum(2 * np.pi * freq_modulation / fs)
        component *= np.sin(cumulative_phase)
        
        # Add random spikes (characteristic of VFib)
        spike_times = np.random.uniform(0, duration_sec, int(duration_sec * np.random.uniform(10, 30)))
        for spike_time in spike_times:
            spike_idx = int(spike_time * fs)
            if 0 <= spike_idx < num_samples:
                spike_width = np.random.uniform(0.01, 0.03)  # 10-30ms spikes
                spike_amplitude = np.random.uniform(0.3, 1.2) * np.random.choice([-1, 1])
                
                # Create spike using gaussian
                spike_std = spike_width / 4
                spike_start = max(0, spike_idx - int(3 * spike_std * fs))
                spike_end = min(num_samples, spike_idx + int(3 * spike_std * fs))
                
                if spike_start < spike_end:
                    spike_time_points = time_axis[spike_start:spike_end]
                    spike = gaussian_wave(spike_time_points, spike_time, spike_amplitude, spike_std)
                    component[spike_start:spike_end] += spike
        
        vfib_signal += component
    
    # Add baseline wander typical of VFib
    baseline_freq = np.random.uniform(0.1, 0.5)  # Slow baseline drift
    baseline_amplitude = base_amplitude * 0.3
    baseline_wander = baseline_amplitude * np.sin(2 * np.pi * baseline_freq * time_axis)
    vfib_signal += baseline_wander
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(vfib_signal))
    if max_amplitude > 2.0:  # Prevent excessive amplitudes
        vfib_signal = vfib_signal * (2.0 / max_amplitude)
    
    return vfib_signal