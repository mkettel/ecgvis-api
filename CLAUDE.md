# ECG Visualizer Backend - Development Guidelines

## Core Philosophy: Physiology-First Development

All ECG generation must start with **real cardiac electrophysiology** and work backward to code implementation. Never implement mathematical shortcuts that don't reflect actual cardiac function.

### Fundamental Physiological Principles

#### 1. Cardiac Electrical Conduction Sequence
Always model in the correct physiological order:
1. **SA node** → Atrial depolarization (P wave)
2. **AV node delay** → PR interval
3. **Bundle of His** → Initial ventricular activation
4. **Bundle branches** → Septal depolarization (Q waves)
5. **Purkinje system** → Free wall depolarization (R waves)  
6. **Terminal conduction** → Basal/posterior forces (S waves)

#### 2. 3D Vector Approach
- **Think in 3D space**: Every electrical event has X, Y, Z components
- **Coordinate System**: X (Right→Left), Y (Superior→Inferior), Z (Posterior→Anterior)
- **Lead Projection**: ECG leads are "windows" viewing 3D vectors from specific angles
- **Never rotate without physiology**: Axis changes must reflect real pathophysiology

#### 3. Pathophysiology Drives Morphology
- **RV hypertrophy** → Enhanced rightward/anterior forces → Right axis, V1 Rs pattern
- **LV hypertrophy** → Enhanced leftward forces → Left axis, tall R waves in I/aVL
- **Bundle branch blocks** → Conduction delays → Wide QRS, altered morphology
- **Fascicular blocks** → Altered septal activation → Axis changes, Q wave loss

## Development Best Practices

### Code Architecture Principles

#### 1. Separation of Concerns
```
Constants (parameters) → Generation (physiology) → Projection (math) → Assembly (timing)
```
- **Constants**: Store all physiological parameters and 3D directions
- **Generation**: Implement cardiac electrophysiology 
- **Projection**: Pure mathematical 3D→12-lead conversion
- **Assembly**: Rhythm timing and beat sequencing

#### 2. Multi-Phase QRS Implementation
Always model QRS as **sequential phases**, never as single vector:
- **Septal phase**: Initial depolarization, creates Q waves
- **Free wall phase**: Main ventricular activation, creates R waves
- **Basal phase**: Terminal activation, creates S waves

Each phase has:
- **Timing envelope**: When it occurs during QRS
- **Vector direction**: 3D spatial direction
- **Magnitude scaling**: Amplitude based on pathophysiology

#### 3. Pathophysiological Vector Modification
When implementing axis changes or pathology:
1. **Start with base physiology**: Normal septal, free wall, basal vectors
2. **Identify the pathophysiology**: What cardiac condition causes this change?
3. **Modify vectors accordingly**: Enhance/reduce specific phases
4. **Validate clinically**: Does the result match real ECGs?

### ECG Morphology Validation

#### Required Patterns for Normal Sinus Rhythm
- **V1**: rS pattern (small r, deep S from posterior basal forces)
- **Lateral leads (I, aVL, V5, V6)**: qR pattern (Q from septal, R from free wall)
- **Inferior leads (II, III, aVF)**: qR pattern with inferior vector dominance
- **Precordial progression**: R wave grows V1→V6, S wave shrinks

#### Axis Deviation Validation
- **Left axis (-30° to -90°)**: 
  - Enhanced R waves in I, aVL
  - Deep S or QS in III
  - Preserved Q waves (unless fascicular block)
- **Right axis (+90° to +180°)**:
  - Enhanced R waves in III, aVF  
  - Deep S or QS in I, aVL
  - V1 transitions rS → Rs → R

#### Bundle Branch Block Patterns
- **RBBB**: Wide QRS, RSR' in V1, deep S in I/V6
- **LBBB**: Wide QRS, broad R in I/V6, QS in V1, no septal Q waves

### Medical Accuracy Standards

#### Physiological Constraints
- **QRS duration**: 60-100ms normal, >120ms for bundle branch blocks
- **Axis range**: -30° to +90° normal, outside = pathological
- **Q wave significance**: Must represent septal depolarization (RIGHT-ward vector)
- **T wave concordance**: Generally follows QRS direction unless pathology

#### Amplitude Relationships
- **Q waves**: <25% of R wave height, <40ms duration
- **R waves**: Vary by lead position and cardiac axis
- **S waves**: Reflect terminal/posterior forces
- **Precordial R/S ratio**: Transition zone around V3-V4

### Implementation Guidelines

#### When Adding New Beat Types
1. **Research the physiology**: What makes this beat different electrically?
2. **Define the pathophysiology**: Origin, conduction pathway, timing
3. **Model the 3D vectors**: How does the electrical sequence differ?
4. **Validate morphology**: Compare with clinical examples
5. **Test axis interactions**: How does pathology interact with axis changes?

#### When Implementing Arrhythmias
1. **Understand the mechanism**: Reentry, automaticity, triggered activity?
2. **Model the timing**: How does it affect beat-to-beat intervals?
3. **Consider AV relationships**: How do P waves and QRS relate?
4. **Implement variability**: Real arrhythmias have natural variation
5. **Validate with clinical examples**: Match documented patterns

#### Debugging Approach
1. **Check physiology first**: Is the electrical sequence correct?
2. **Verify 3D vectors**: Do directions match anatomical expectations?  
3. **Validate projections**: Are lead calculations mathematically correct?
4. **Compare with clinical data**: Does output match real ECGs?
5. **Test edge cases**: Extreme axes, complex pathologies

### Performance and Scalability

#### Computational Efficiency
- **Vectorized operations**: Use NumPy for batch calculations
- **Minimal object creation**: Reuse arrays where possible
- **Pre-computed constants**: Calculate directions once, reuse
- **Efficient timing**: Use time-based indexing, not iterative loops

#### Educational Optimization
- **Clean morphology**: Reduce noise for better Q wave visibility
- **Appropriate scaling**: Amplitudes that match clinical teaching
- **Consistent patterns**: Reliable morphology for repeated generation
- **Interactive responsiveness**: Real-time parameter changes

### Testing and Validation

#### Required Test Cases
- **Normal sinus rhythm**: All standard morphologies present
- **Axis sweep**: Test full range -180° to +180°
- **Pathology combinations**: Axis + bundle blocks, hypertrophy patterns
- **Lead consistency**: All 12 leads show appropriate patterns
- **Educational clarity**: Visible Q waves, proper R/S progression

#### Clinical Validation Sources
- **Textbook examples**: Validate against published ECG atlases
- **Pathophysiology references**: Ensure mechanisms are accurate
- **Clinical cases**: Test against real patient examples
- **Educational standards**: Match teaching hospital criteria

## Key Architectural Decisions

### Multi-Phase QRS Over Single Vector
**Rationale**: Real ventricular depolarization is sequential, not instantaneous. Single vectors cannot create proper Q waves, R waves, and S waves with correct timing relationships.

### Pathophysiological Axis Over Mathematical Rotation
**Rationale**: Axis deviations result from specific cardiac pathologies (hypertrophy, blocks). Simple rotation doesn't reflect real electrical changes or create appropriate morphology patterns.

### 3D Vector Space Over 2D Simplification  
**Rationale**: The heart is a 3D organ. Precordial lead morphology requires anterior-posterior (Z-axis) modeling for realistic V1-V6 patterns.

### Education-Focused Design
**Rationale**: This system teaches cardiac electrophysiology. All decisions prioritize physiological accuracy and educational clarity over computational shortcuts.

## Common Pitfalls to Avoid

- **Don't use mathematical shortcuts**: Always implement the physiology
- **Don't ignore timing relationships**: Phase overlap and sequence matter
- **Don't forget 3D space**: Z-axis is crucial for precordial leads
- **Don't hardcode morphologies**: Generate from physiological principles
- **Don't skip clinical validation**: Always compare with real ECGs
- **Don't sacrifice accuracy for simplicity**: Complex physiology requires complex models

## Development Workflow

1. **Research the physiology** behind any new feature
2. **Design the vector model** based on cardiac anatomy
3. **Implement with proper timing** and phase relationships
4. **Validate against clinical examples**
5. **Test axis interactions** and pathology combinations
6. **Optimize for educational clarity**
7. **Document the physiological reasoning**

This approach ensures every feature reflects real cardiac electrophysiology and maintains educational value for users learning ECG interpretation.