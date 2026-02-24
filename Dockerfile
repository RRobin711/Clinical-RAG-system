FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY app.py .
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Copy pre-built index (for deployment)
# NOTE: You must build the index locally first, then include it
COPY data/index/ data/index/

# Gradio default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
