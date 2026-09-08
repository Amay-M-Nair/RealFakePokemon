"""Shared Django settings."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BASE_DIR.parent

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key-change-in-prod")
DEBUG = False
ALLOWED_HOSTS: list[str] = []

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "generator",
    "gallery",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Generator ---------------------------------------------------------------

# The exported ONNX generator and its latent mean. Downloaded at build time in
# the Dockerfile rather than committed, since it is derived from copyrighted art.
GENERATOR_MODEL_PATH = Path(
    os.environ.get("GENERATOR_MODEL_PATH", BASE_DIR / "models" / "generator.onnx")
)
GENERATOR_Z_MEAN_PATH = Path(
    os.environ.get("GENERATOR_Z_MEAN_PATH", BASE_DIR / "models" / "z_mean.npy")
)
# Free-tier containers get roughly one usable core; extra ORT threads there cost
# more in contention than they win in parallelism.
GENERATOR_THREADS = int(os.environ.get("GENERATOR_THREADS", "1"))
GENERATOR_MAX_BATCH = int(os.environ.get("GENERATOR_MAX_BATCH", "8"))
# Images per ONNX call. Activations scale with the batch: rendering 8 frames
# in one call peaks at ~581MB resident and OOMs a 512MB tier, while chunks of
# 2 hold the peak near the single-image cost. Raise only if RAM is plentiful.
GENERATOR_CHUNK_SIZE = int(os.environ.get("GENERATOR_CHUNK_SIZE", "2"))

# Seed fully determines the image, so rendered PNGs are immutable and can be
# cached forever. This is what keeps repeat traffic off the CPU entirely.
GENERATED_IMAGE_MAX_AGE = 60 * 60 * 24 * 365
