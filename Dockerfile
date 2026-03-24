FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Default: run the FastAPI API server
# Override CMD to run the crawler: docker run ... python app/main.py
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
