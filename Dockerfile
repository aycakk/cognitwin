FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create runtime directories (volumes will overlay these at startup)
RUN mkdir -p static/chromadb logs data/audit ontologies

EXPOSE 8011

CMD ["python", "-m", "uvicorn", "src.services.api.app:app", \
     "--host", "0.0.0.0", "--port", "8011"]
