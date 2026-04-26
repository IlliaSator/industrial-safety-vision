# API Usage

Start locally in mock mode:

```bash
INDUSTRIAL_SAFETY_MOCK_DETECTOR=true python -m uvicorn industrial_safety_vision.api.main:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Model info:

```bash
curl http://localhost:8000/model/info
```

Image prediction:

```bash
curl -F "file=@docs/assets/demo_input.jpg" http://localhost:8000/predict/image
```

Video prediction:

```bash
curl -F "file=@data/samples/demo.mp4" http://localhost:8000/predict/video
```

Recent alerts:

```bash
curl http://localhost:8000/alerts
```

Service metrics:

```bash
curl http://localhost:8000/metrics
```

The API stores metrics and recent alerts in memory for the current process. Production deployments should replace this with durable storage and observability tooling.
