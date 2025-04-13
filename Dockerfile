# Використовуємо slim версію PyTorch
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Встановлюємо змінні середовища
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Встановлюємо необхідні системні пакети для обробки аудіо
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libsox-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Створюємо користувача без прав root
RUN useradd -m -u 1000 appuser

# Створюємо робочу директорію та встановлюємо права
WORKDIR /app
RUN chown appuser:appuser /app

# Копіюємо файл з залежностями та встановлюємо їх
COPY --chown=appuser:appuser requirements_preprocess.txt .
RUN pip3 install --no-cache-dir -r requirements_preprocess.txt && \
    pip3 install --no-cache-dir torchaudio==2.6.0 && \
    rm -rf ~/.cache/pip/*

# Створюємо директорію для чекпоінтів
RUN mkdir -p /app/ckpts && chown -R appuser:appuser /app/ckpts

# Копіюємо необхідні файли
COPY --chown=appuser:appuser preprocess_handler.py .
COPY --chown=appuser:appuser .env_prod .
COPY --chown=appuser:appuser test_input.json .

# Копіюємо чекпоінти
COPY --chown=appuser:appuser ckpts/ ./ckpts/

# Перемикаємося на користувача без прав root
USER appuser

# Точка входу
CMD ["python3", "-u", "preprocess_handler.py"]