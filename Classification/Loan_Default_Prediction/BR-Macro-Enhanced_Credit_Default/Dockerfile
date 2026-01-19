FROM python:3.11-slim

# -------------------------------------------------
# Environment settings
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# -------------------------------------------------
# Working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# System dependencies (required for LightGBM)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Project files
# -------------------------------------------------
COPY src/ src/
COPY models/ models/

# -------------------------------------------------
# Expose API port
# -------------------------------------------------
EXPOSE 8000

# -------------------------------------------------
# Run FastAPI
# -------------------------------------------------
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
