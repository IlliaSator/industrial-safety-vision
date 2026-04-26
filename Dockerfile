FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INDUSTRIAL_SAFETY_MOCK_DETECTOR=true

WORKDIR /app

ARG INSTALL_ML_DEPS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      fastapi>=0.110 \
      "uvicorn[standard]>=0.27" \
      python-multipart>=0.0.9 \
      pydantic>=2.6 \
      pydantic-settings>=2.2 \
      numpy>=1.24 \
      Pillow>=10.0 \
      PyYAML>=6.0 && \
    if [ "$INSTALL_ML_DEPS" = "true" ]; then \
      python -m pip install --no-cache-dir torch>=2.1 ultralytics>=8.1 opencv-python>=4.8; \
    fi && \
    python -m pip install --no-cache-dir -e . --no-deps

COPY configs ./configs
COPY scripts ./scripts
COPY data/README.md ./data/README.md

EXPOSE 8000

CMD ["uvicorn", "industrial_safety_vision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
