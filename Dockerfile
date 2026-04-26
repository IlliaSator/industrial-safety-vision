FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts
COPY data/README.md ./data/README.md

EXPOSE 8000

CMD ["uvicorn", "industrial_safety_vision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
