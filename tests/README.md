# ECG Simulator Test Suite

This comprehensive test suite ensures the medical accuracy, reliability, and performance of the ECG simulator backend.

## Test Organization

### 📁 Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_vector_projection.py   # 3D vector projection tests
│   ├── test_beat_generation.py     # Beat morphology tests
│   ├── test_parameter_validation.py # Parameter validation tests
│   └── test_performance.py         # Performance benchmarks
├── integration/                # API integration tests
│   └── test_api_endpoints.py       # FastAPI endpoint tests
└── medical/                    # Medical accuracy tests
    ├── test_qt_correction.py       # QT interval and Bazett's formula
    ├── test_av_conduction.py       # AV blocks and escape rhythms
    ├── test_arrhythmias.py         # Arrhythmia generation
    └── test_complex_rhythms.py     # Complex rhythm scenarios
```

### 🏷️ Test Categories

Tests are organized using pytest markers:

- **`@pytest.mark.unit`** - Unit tests for individual functions
- **`@pytest.mark.integration`** - API and system integration tests  
- **`@pytest.mark.medical`** - Medical accuracy and physiological correctness
- **`@pytest.mark.performance`** - Performance and benchmarking tests
- **`@pytest.mark.slow`** - Tests that take longer to run

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt
```

### Quick Test Commands

```bash
# Run all tests
python3 run_tests.py

# Run quick tests only (excludes slow/performance tests)
python3 run_tests.py --quick

# Run specific test categories
python3 run_tests.py --medical      # Medical accuracy tests
python3 run_tests.py --unit         # Unit tests
python3 run_tests.py --integration  # API tests
python3 run_tests.py --performance  # Performance tests

# Generate coverage report
python3 run_tests.py --coverage --html
```

### Direct pytest Commands

```bash
# Run all tests with coverage
pytest --cov=ecg_simulator --cov-report=html

# Run specific test file
pytest tests/medical/test_qt_correction.py -v

# Run tests matching pattern
pytest -k "test_sinus" -v

# Run with specific markers
pytest -m "medical and not slow" -v

# Benchmark tests
pytest --benchmark-only --benchmark-sort=mean
```

## Test Coverage

The test suite covers:

### 🫀 Medical Accuracy (95%+ Coverage)
- **QT Interval Correction**: Bazett's formula implementation
- **AV Conduction**: All degrees of AV block, escape rhythms
- **Arrhythmias**: VT, AFib, Flutter, Torsades de Pointes
- **Rhythm Hierarchy**: VT interrupting SVT, rhythm precedence
- **Beat Morphology**: P, QRS, T wave timing and amplitudes

### 🔧 Technical Components (90%+ Coverage)
- **3D Vector Projection**: 12-lead ECG generation from cardiac vectors
- **Beat Generation**: Individual beat creation and morphology
- **Parameter Validation**: Input validation and error handling
- **API Endpoints**: Single-lead and 12-lead generation APIs

### ⚡ Performance (Benchmarked)
- **Execution Time**: Generation time scaling with duration
- **Memory Usage**: Memory efficiency for long ECGs
- **Scalability**: Performance with complex rhythm combinations

## Key Test Cases

### Medical Validation Examples

```python
def test_bazett_formula_accuracy():
    """Verify QT intervals follow Bazett's formula: QTc = QT / √(RR)"""
    # Tests QT correction at different heart rates
    
def test_third_degree_av_block_complete_dissociation():
    """Verify complete AV block produces independent atrial/ventricular rhythms"""
    # Tests P waves at 60 bpm, QRS at 45 bpm independently
    
def test_torsades_triggering_with_long_qtc():
    """Test Torsades de Pointes triggering with prolonged QTc"""
    # Tests probabilistic triggering at QTc >500ms
```

### Performance Benchmarks

```python
def test_basic_sinus_generation_time():
    """Basic sinus rhythm should generate in <2 seconds"""
    
def test_12_lead_projection_performance():
    """60-second 12-lead projection should complete in <1 second"""
```

## Expected Test Results

### Passing Criteria
- **Medical Tests**: 100% pass rate (these validate clinical accuracy)
- **Unit Tests**: 100% pass rate (these catch regressions) 
- **Integration Tests**: 100% pass rate (these ensure API reliability)
- **Performance Tests**: Meet timing benchmarks

### Performance Benchmarks
- Basic sinus generation: <2 seconds for 10-second ECG
- Complex rhythms: <5 seconds for 30-second ECG  
- 12-lead projection: <1 second for 60-second ECG
- Memory usage: <100MB for basic, <500MB for long ECGs

## Interpreting Test Failures

### Medical Test Failures
❌ **Critical** - Indicates incorrect medical implementation
- Review medical literature and algorithm implementation
- May require consultation with cardiology experts

### Unit Test Failures  
❌ **High Priority** - Indicates broken functionality
- Debug individual component failure
- Check for recent code changes that broke the component

### Integration Test Failures
❌ **Medium Priority** - Indicates API or system integration issues
- Check API parameter validation
- Verify response format consistency

### Performance Test Failures
⚠️ **Medium Priority** - Indicates performance regression
- Profile code to identify bottlenecks
- May indicate need for optimization

## Adding New Tests

### Medical Accuracy Tests
When adding new arrhythmias or medical features:

```python
@pytest.mark.medical
def test_new_arrhythmia_characteristics(self, tolerance_config):
    \"\"\"Test new arrhythmia follows expected medical patterns.\"\"\"
    params = AdvancedECGParams(...)
    time_axis, signal, rhythm_desc = generate_physiologically_accurate_ecg(**params.model_dump())
    
    # Verify medical accuracy
    assert expected_medical_pattern in rhythm_desc
    # Add specific medical validation
```

### Performance Tests
For performance-critical changes:

```python
@pytest.mark.performance
def test_new_feature_performance(self, benchmark):
    \"\"\"Benchmark new feature performance.\"\"\"
    def run_new_feature():
        return new_feature_function(params)
    
    result = benchmark(run_new_feature)
    # Verify performance meets requirements
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run ECG Tests
  run: |
    pip install -r requirements-test.txt
    python run_tests.py --quick --coverage
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
```

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)
- **Parameter Sets**: Pre-configured rhythm parameters
- **Tolerance Values**: Standard numerical tolerances  
- **Medical References**: Expected values for validation
- **Sample Data**: 3D cardiac vectors for testing

### Medical Reference Values
- Normal QTc: 350-450ms
- Normal axis: -30° to +90°
- Sinus rate: 60-100 bpm
- AV block thresholds and patterns

This test suite ensures that any changes to the ECG simulator maintain medical accuracy while improving functionality and performance.