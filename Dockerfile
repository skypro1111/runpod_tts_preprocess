FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install necessary system packages for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libsox-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Create a user without root privileges
RUN useradd -m -u 1000 appuser

# Create working directory and set permissions
WORKDIR /app
RUN chown appuser:appuser /app

# Copy the requirements file and install dependencies
COPY --chown=appuser:appuser requirements_preprocess.txt .
RUN pip3 install --no-cache-dir -r requirements_preprocess.txt && \
    pip3 install --no-cache-dir torchaudio==2.6.0 && \
    rm -rf ~/.cache/pip/*

# Create directory for checkpoints
RUN mkdir -p /app/ckpts && chown -R appuser:appuser /app/ckpts

# Copy necessary files
COPY --chown=appuser:appuser preprocess_handler.py .
COPY --chown=appuser:appuser .env_prod .
COPY --chown=appuser:appuser test_input.json .

# Copy checkpoints
COPY --chown=appuser:appuser ckpts/ ./ckpts/

# Switch to the non-root user
USER appuser

# Entry point
CMD ["python3", "-u", "preprocess_handler.py"]