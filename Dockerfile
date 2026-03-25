FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir --no-deps "sentence-transformers==5.3.0"
RUN python -m pip check

COPY . .

# Default: run the FastAPI API server
# Override CMD to run the crawler: docker run ... python app/main.py
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
