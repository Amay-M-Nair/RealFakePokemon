"""Local development settings."""

from .base import *  # noqa: F401,F403

DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
INTERNAL_IPS = ["127.0.0.1"]

# ManifestStaticFilesStorage requires collectstatic to have run; plain storage
# keeps `runserver` working from a clean checkout.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
