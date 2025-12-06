# ----------------------------------------------------
# Builder Stage
# ----------------------------------------------------
FROM python:3.12.1-bookworm as builder

WORKDIR /app
# libpq-dev: psycopg2 빌드에 필요
# zlib1g-dev, libjpeg-dev: PyMuPDF나 Pillow 등 이미지 관련 라이브러리 빌드에 필요
# libgomp1: PyMuPDF가 병렬 처리를 위해 런타임에 필요하며, 빌드 시에도 의존성이 걸릴 수 있음
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    zlib1g-dev \
    libjpeg-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ./src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# Production Stage
# ----------------------------------------------------
FROM python:3.12.1-slim-bookworm

WORKDIR /app
# * libgomp1: PyMuPDF 등 병렬 처리에 사용되는 C 라이브러리의 런타임
# * postgresql-client: psql 명령줄 도구가 필요할 경우
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Copy application code with ownership
COPY --chown=appuser:appuser ./src .
USER appuser
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
