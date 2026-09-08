# Serving image only -- deliberately never installs PyTorch.
#
# torch on CPU is ~800MB installed and ~300MB resident, which does not fit a
# 512MB free tier. onnxruntime is ~50MB and the fp16 generator is ~60MB, so the
# whole service lands around 250-300MB resident. That is the entire reason the
# model is exported to ONNX rather than served from a checkpoint.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=config.settings.prod

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/serve.txt requirements/serve.txt
RUN pip install --no-cache-dir -r requirements/serve.txt

COPY server/ /app/

# The trained generator is fetched at build time rather than committed: it is
# derived from copyrighted artwork and has no business in git history.
# Set MODEL_URL to a Hugging Face Hub (or release) asset.
ARG MODEL_URL=""
ARG Z_MEAN_URL=""
RUN mkdir -p /app/models \
    && if [ -n "$MODEL_URL" ]; then curl -fsSL "$MODEL_URL" -o /app/models/generator.onnx; fi \
    && if [ -n "$Z_MEAN_URL" ]; then curl -fsSL "$Z_MEAN_URL" -o /app/models/z_mean.npy; fi

# SECRET_KEY is required by prod settings but collectstatic does not use it.
RUN SECRET_KEY=build-only python manage.py collectstatic --noinput

EXPOSE 8000

# One worker: a second doubles the resident model for no throughput gain on a
# single-core free tier. Threads handle concurrent requests instead.
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 --access-logfile -"]
