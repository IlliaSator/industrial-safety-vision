# Error Analysis

Expected failure cases to evaluate with real site footage:

- Helmets missed under occlusion, hard shadows, or similar-colored backgrounds
- Reflective vest false positives on signs, cones, or machinery
- Small workers in the background with low pixel area
- Low-light frames and high ISO noise
- Motion blur from fast vehicles or low shutter speed
- Unusual overhead or oblique camera angles
- Overlapping workers causing wrong PPE association
- Temporary detector flicker causing track fragmentation
- Perspective distortion making vehicle proximity thresholds unreliable

Mitigations include stronger dataset coverage, camera-specific validation sets, perspective calibration, temporal smoothing, better trackers, and human review of alert samples.
