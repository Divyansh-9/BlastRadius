FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY incident_env/ ./incident_env/
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .
COPY inference.py .
COPY app_ui.py .

# Expose port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Run the server
CMD ["python", "app_ui.py"]
