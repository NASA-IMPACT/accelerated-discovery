FROM python:3.12-slim

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install uv and deps
RUN pip install uv
COPY pyproject.toml README.md ./
COPY risk_annotation_ui ./risk_annotation_ui
RUN uv pip install --system --no-cache .

# Expose port
EXPOSE 8000

# Start FastAPI (pointing to the app inside risk_annotation_ui/)
CMD ["uvicorn", "risk_annotation_ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
