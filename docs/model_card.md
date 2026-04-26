# Model Card

## Intended Use

Industrial Safety Vision is intended for workplace safety analytics in industrial scenes: PPE compliance, worker tracking, danger-zone detection, and worker/vehicle proximity review.

## Out-Of-Scope Use

The project should not be used as an autonomous safety-critical control system, a replacement for certified safety equipment, or the sole source for disciplinary action.

## Model Requirements

Demo mode can run with generic YOLO checkpoints, but full safety mode expects a custom YOLO detector trained on classes such as `person`, `helmet`, `safety_vest`, `forklift`, and `vehicle`.

## Data Expectations

A production PPE model should be trained and evaluated on site-relevant camera footage or a carefully reviewed proxy dataset. The validation split should preserve camera/site separation where possible to avoid leakage and should include low-light, occlusion, dense-worker, and vehicle-interaction examples.

## Evaluation Expectations

Report detection metrics separately from downstream alert metrics:

- mAP@0.5 and mAP@0.5:0.95 for object detection quality
- per-class precision/recall for PPE and vehicle classes
- alert precision/recall after temporal smoothing
- latency/FPS on target hardware
- false positive and false negative samples per camera

## Limitations

- Accuracy depends heavily on camera placement, image quality, and dataset coverage.
- PPE association uses bounding-box overlap and can fail with overlapping workers.
- Low light, motion blur, occlusion, and small distant workers can reduce recall.
- Site-specific danger zones require calibration.

## Ethical And Safety Considerations

This system should support safety teams, not replace them. It should not be used as the sole source for disciplinary action or safety-critical automation without human validation, audit trails, and a reviewed deployment process.

## Deployment Cautions

- Validate on camera-specific holdout footage before use.
- Calibrate danger zones per camera.
- Review alert false positives and false negatives with safety staff.
- Monitor confidence drift, lighting changes, and camera movement.
- Keep human review in the loop for safety decisions.
- Version model checkpoints, configs, and safety-rule thresholds together.
- Keep a rollback plan for model or rule updates.

## Failure Modes

- Helmet missed when partially hidden by machinery
- Vest detected on reflective background objects
- Forklift proximity under-estimated from perspective distortion
- Identity switches during dense worker overlap

## Monitoring Recommendations

Track alert volume, per-camera false positive rate, detector confidence distribution, missing-label rates in reviewed samples, latency/FPS, and examples of suppressed alerts during cooldown.

## Current Repository Status

The committed assets demonstrate architecture and pipeline readiness. They do not claim trained PPE accuracy. The included benchmark example is a mock pipeline benchmark and should not be interpreted as neural-network performance.
