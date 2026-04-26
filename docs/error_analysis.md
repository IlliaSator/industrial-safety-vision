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

## Failure Matrix

| Failure case | Likely cause | Mitigation |
| --- | --- | --- |
| Missed helmet | occlusion, small object size, hard shadows | add close/far examples, tune image size, review confidence threshold |
| Vest false positive | reflective machinery or signage | add hard negatives, improve class definitions |
| Wrong PPE association | overlapping workers | tracker-aware association or keypoint/person-part model |
| Danger-zone false alert | polygon not calibrated to camera perspective | site calibration and reviewed zone config |
| Vehicle proximity error | pixel distance not equal to physical distance | homography or depth-aware calibration |
| Alert spam | detector flicker | temporal smoothing and cooldown tuning |

## Suggested Review Workflow

1. Sample alerts and non-alert frames per camera.
2. Tag false positives, false negatives, and association errors.
3. Split errors by lighting, distance, occlusion, and camera angle.
4. Add representative failures to the validation set.
5. Re-run evaluation and compare per-class precision/recall before deployment.
