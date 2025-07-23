"""
Performance tests and benchmarks for ECG generation.
Tests execution time, memory usage, and scalability.
"""
import pytest
import time
import psutil
import os
from ecg_simulator.rhythm_logic import generate_physiologically_accurate_ecg
from ecg_simulator.api_models import AdvancedECGParams
from ecg_simulator.full_ecg.vector_projection import project_cardiac_vector_to_12_leads
import numpy as np

class TestPerformance:
    """Test performance characteristics of ECG generation."""
    
    @pytest.mark.performance
    def test_basic_sinus_generation_time(self, basic_sinus_params):
        """Test that basic sinus rhythm generation completes in reasonable time."""
        start_time = time.time()
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**basic_sinus_params.model_dump())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 2 seconds for 10-second ECG
        assert execution_time < 2.0, f"Basic sinus generation took {execution_time:.3f}s (too slow)"
        
        # Verify output
        assert len(signal) > 0
        assert len(time_axis) == len(signal)
    
    @pytest.mark.performance
    def test_complex_rhythm_generation_time(self, tolerance_config):
        """Test performance with complex rhythm combinations."""
        complex_params = AdvancedECGParams(
            heart_rate_bpm=80,
            duration_sec=30.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.1,
            enable_pac=True,
            pac_probability_per_sinus=0.05,
            enable_vt=True,
            vt_start_time_sec=15.0,
            vt_duration_sec=8.0,
            vt_rate_bpm=180,
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.001,
            enable_atrial_fibrillation=False
        )
        
        start_time = time.time()
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**complex_params.model_dump())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Complex rhythms should still complete within 5 seconds for 30-second ECG
        assert execution_time < 5.0, f"Complex rhythm generation took {execution_time:.3f}s (too slow)"
        
        assert len(signal) > 0
    
    @pytest.mark.performance
    def test_long_duration_scaling(self, tolerance_config):
        """Test performance scaling with ECG duration."""
        durations = [10.0, 30.0, 60.0, 120.0]
        execution_times = []
        
        for duration in durations:
            params = AdvancedECGParams(
                heart_rate_bpm=60,
                duration_sec=duration,
                enable_pvc=False,
                enable_pac=False,
                enable_atrial_fibrillation=False,
                enable_vt=False
            )
            
            start_time = time.time()
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify output scales correctly
            expected_samples = int(duration * 250)
            actual_samples = len(signal)
            assert abs(actual_samples - expected_samples) < 250, \
                f"Sample count mismatch for {duration}s duration"
        
        # Execution time should scale roughly linearly with duration
        # Allow for some overhead, but shouldn't be quadratic
        time_per_second = [exec_time / duration for exec_time, duration in zip(execution_times, durations)]
        
        # Time per second should be relatively consistent (within 50% variation)
        min_time_per_sec = min(time_per_second)
        max_time_per_sec = max(time_per_second)
        
        assert max_time_per_sec / min_time_per_sec < 2.0, \
            f"Execution time scaling poor: {time_per_second}"
    
    @pytest.mark.performance
    def test_12_lead_projection_performance(self, sample_12_lead_vectors):
        """Test performance of 12-lead vector projection."""
        time_axis, cardiac_vectors = sample_12_lead_vectors
        
        # Test with larger vector array
        large_duration = 60.0  # 60 seconds
        large_samples = int(large_duration * 250)
        large_time_axis = np.linspace(0, large_duration, large_samples)
        
        # Create large cardiac vector array
        qrs_direction = np.array([0.7, 0.6, 0.4])
        qrs_direction = qrs_direction / np.linalg.norm(qrs_direction)
        
        # Simple repeated pattern for testing
        pattern_length = len(cardiac_vectors)
        num_repeats = large_samples // pattern_length + 1
        large_cardiac_vectors = np.tile(cardiac_vectors, (num_repeats, 1))[:large_samples]
        
        start_time = time.time()
        
        projections = project_cardiac_vector_to_12_leads(large_cardiac_vectors)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should project 60 seconds of data in under 1 second
        assert execution_time < 1.0, f"12-lead projection took {execution_time:.3f}s (too slow)"
        
        # Verify all leads generated
        assert len(projections) == 12
        for lead_signal in projections.values():
            assert len(lead_signal) == large_samples
    
    @pytest.mark.performance
    def test_memory_usage_basic(self, basic_sinus_params):
        """Test memory usage for basic ECG generation."""
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate ECG
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**basic_sinus_params.model_dump())
        
        # Get memory after generation
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Should not use excessive memory (< 100MB for 10-second ECG)
        assert memory_increase < 100, f"Memory usage {memory_increase:.1f}MB too high"
        
        # Verify output
        assert len(signal) > 0
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_long_duration(self, tolerance_config):
        """Test memory usage for long duration ECGs."""
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate 5-minute ECG
        long_params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=300.0,  # 5 minutes
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**long_params.model_dump())
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Should scale reasonably with duration (< 500MB for 5-minute ECG)
        assert memory_increase < 500, f"Memory usage {memory_increase:.1f}MB too high for long ECG"
        
        # Expected samples: 300 seconds * 250 Hz = 75,000 samples
        expected_samples = 300 * 250
        assert abs(len(signal) - expected_samples) < 250
    
    @pytest.mark.performance
    def test_concurrent_generation_performance(self, basic_sinus_params):
        """Test performance when generating multiple ECGs."""
        num_generations = 5
        total_start_time = time.time()
        
        for i in range(num_generations):
            time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**basic_sinus_params.model_dump())
            assert len(signal) > 0, f"Generation {i} failed"
        
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        avg_time_per_generation = total_execution_time / num_generations
        
        # Average time per generation should be reasonable
        assert avg_time_per_generation < 2.0, \
            f"Average generation time {avg_time_per_generation:.3f}s too slow"
    
    @pytest.mark.performance
    def test_afib_performance(self, afib_params):
        """Test performance of AFib generation (irregular rhythms)."""
        start_time = time.time()
        
        time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**afib_params.model_dump())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # AFib should not be significantly slower than sinus
        assert execution_time < 3.0, f"AFib generation took {execution_time:.3f}s (too slow)"
        
        assert len(signal) > 0
        assert "afib" in rhythm_desc.lower() or "fibrillation" in rhythm_desc.lower()
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="basic_generation")
    def test_benchmark_basic_sinus(self, benchmark, basic_sinus_params):
        """Benchmark basic sinus rhythm generation using pytest-benchmark."""
        def generate_sinus():
            return generate_physiologically_accurate_ecg(**basic_sinus_params.model_dump())
        
        result = benchmark(generate_sinus)
        time_axis, signal, rhythm_desc = result
        
        assert len(signal) > 0
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="vector_projection")
    def test_benchmark_12_lead_projection(self, benchmark, sample_12_lead_vectors):
        """Benchmark 12-lead vector projection using pytest-benchmark."""
        time_axis, cardiac_vectors = sample_12_lead_vectors
        
        def project_vectors():
            return project_cardiac_vector_to_12_leads(cardiac_vectors)
        
        projections = benchmark(project_vectors)
        
        assert len(projections) == 12
    
    @pytest.mark.performance
    def test_vector_projection_scaling(self, tolerance_config):
        """Test vector projection performance scaling."""
        sample_counts = [100, 500, 1000, 5000, 10000]
        execution_times = []
        
        qrs_direction = np.array([0.7, 0.6, 0.4])
        qrs_direction = qrs_direction / np.linalg.norm(qrs_direction)
        
        for sample_count in sample_counts:
            # Create cardiac vectors
            cardiac_vectors = np.random.randn(sample_count, 3) * 0.1
            cardiac_vectors[:, :] = qrs_direction  # Simplified for testing
            
            start_time = time.time()
            projections = project_cardiac_vector_to_12_leads(cardiac_vectors)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Verify output
            assert len(projections) == 12
            for lead_signal in projections.values():
                assert len(lead_signal) == sample_count
        
        # Should scale roughly linearly
        time_per_sample = [exec_time / sample_count for exec_time, sample_count in 
                          zip(execution_times, sample_counts)]
        
        # Time per sample should be consistent (allow larger factor due to timing precision limits)
        min_time_per_sample = min(time_per_sample)
        max_time_per_sample = max(time_per_sample)
        
        # If times are very small (< 1ms per sample), timing precision can cause large ratios
        if min_time_per_sample > 1e-6:  # > 1 microsecond per sample
            assert max_time_per_sample / min_time_per_sample < 10.0, \
                f"Vector projection scaling poor: {time_per_sample}"
        else:
            # For very fast operations, just ensure all times are reasonable
            assert all(t < 1e-4 for t in time_per_sample), \
                f"Vector projection too slow: {time_per_sample}"
    
    @pytest.mark.performance
    def test_rhythm_complexity_impact(self, tolerance_config):
        """Test how rhythm complexity affects performance."""
        
        # Simple sinus rhythm
        simple_params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_pvc=False,
            enable_pac=False,
            enable_atrial_fibrillation=False,
            enable_vt=False
        )
        
        # Complex rhythm
        complex_params = AdvancedECGParams(
            heart_rate_bpm=60,
            duration_sec=20.0,
            enable_pvc=True,
            pvc_probability_per_sinus=0.2,
            enable_pac=True,
            pac_probability_per_sinus=0.1,
            enable_atrial_fibrillation=True,
            afib_average_ventricular_rate_bpm=80,
            enable_torsades_risk=True,
            torsades_probability_per_beat=0.01,
            enable_vt=False
        )
        
        # Time simple rhythm
        start_time = time.time()
        time_axis_simple, signal_simple, _ = generate_physiologically_accurate_ecg(**simple_params.model_dump())
        simple_time = time.time() - start_time
        
        # Time complex rhythm
        start_time = time.time()
        time_axis_complex, signal_complex, _ = generate_physiologically_accurate_ecg(**complex_params.model_dump())
        complex_time = time.time() - start_time
        
        # Complex rhythm should not be more than 5x slower
        time_ratio = complex_time / simple_time if simple_time > 0 else float('inf')
        assert time_ratio < 5.0, f"Complex rhythm {time_ratio:.1f}x slower than simple"
        
        # Both should generate valid signals
        assert len(signal_simple) > 0
        assert len(signal_complex) > 0