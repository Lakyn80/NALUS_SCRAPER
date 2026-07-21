FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN set -eux; \
    for attempt in 1 2 3 4; do \
        python -m pip install -r requirements.txt && break; \
        if [ "$attempt" -eq 4 ]; then exit 1; fi; \
        sleep $((attempt * 15)); \
    done
RUN set -eux; \
    for attempt in 1 2 3 4; do \
        python -m pip install --no-deps "sentence-transformers==5.3.0" && break; \
        if [ "$attempt" -eq 4 ]; then exit 1; fi; \
        sleep $((attempt * 15)); \
    done
RUN python -m pip check

COPY app ./app

RUN mkdir -p /app/batches /app/storage /app/models /app/artifacts /app/app/artifacts \
    && rm -rf /root/.cache/pip

# Default: run the FastAPI API server
# Override CMD to run the crawler: docker run ... python app/main.py
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
