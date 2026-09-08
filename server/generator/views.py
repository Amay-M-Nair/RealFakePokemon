"""Views for generating and serving Pokemon.

The split here is deliberate. POST /api/generate does no inference at all -- it
picks seeds and returns markup pointing at GET /g/<seed>-<psi>.png. All the
actual work happens on that deterministic GET, which is immutable and cached
forever. Repeat traffic therefore costs zero CPU, which is what makes a free
512MB instance viable.
"""

from __future__ import annotations

import random

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.cache import cache_control
from django.views.decorators.http import require_GET, require_POST
from django_ratelimit.decorators import ratelimit

from gallery.models import Generation

from .inference import (
    GenerationRequest,
    ModelUnavailable,
    generate,
    interpolate,
    is_available,
    to_png_bytes,
)

MAX_SEED = 2**31 - 1
DEFAULT_TRUNCATION = 0.8
INTERPOLATION_STEPS = 8


def _clamp_psi(value, default: float = DEFAULT_TRUNCATION) -> float:
    try:
        return max(0.0, min(1.5, float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def image_url(seed: int, psi: float) -> str:
    """Percent-encoded psi keeps the URL an integer pair, so it caches cleanly."""
    return reverse("generator:image", kwargs={"seed": seed, "psi": round(psi * 100)})


def index(request):
    return render(
        request,
        "index.html",
        {
            "model_available": is_available(),
            "default_truncation": DEFAULT_TRUNCATION,
            "top": Generation.objects.top(limit=12),
        },
    )


@require_POST
@ratelimit(key="ip", rate="30/m", block=True)
def generate_view(request):
    """Pick seeds and hand back markup. Cheap by design -- no inference here."""
    count = _clamp_int(request.POST.get("count"), 1, 1, settings.GENERATOR_MAX_BATCH)
    psi = _clamp_psi(request.POST.get("truncation"))

    raw_seed = request.POST.get("seed", "").strip()
    if raw_seed:
        base = _clamp_int(raw_seed, 0, 0, MAX_SEED)
        seeds = [base] if count == 1 else [(base + i) % MAX_SEED for i in range(count)]
    else:
        seeds = [random.randint(0, MAX_SEED) for _ in range(count)]

    psi_pct = round(psi * 100)
    results = [
        {"seed": s, "psi": psi, "psi_pct": psi_pct, "url": image_url(s, psi)} for s in seeds
    ]

    if request.headers.get("Accept") == "application/json":
        return JsonResponse({"results": results})
    return render(request, "partials/results.html", {"results": results, "psi": psi})


@require_GET
@cache_control(public=True, max_age=settings.GENERATED_IMAGE_MAX_AGE, immutable=True)
def image_view(request, seed: int, psi: int):
    """Render one Pokemon. Immutable: (seed, psi) fully determines the bytes."""
    transparent = request.GET.get("bg") == "transparent"
    try:
        images = generate(
            GenerationRequest(
                seed=seed, truncation=psi / 100.0, count=1, transparent=transparent
            )
        )
    except ModelUnavailable as exc:
        return HttpResponse(str(exc), status=503, content_type="text/plain")

    response = HttpResponse(to_png_bytes(images[0]), content_type="image/png")
    response["Content-Disposition"] = f'inline; filename="pokemon-{seed}-{psi}.png"'
    return response


@require_GET
@cache_control(public=True, max_age=settings.GENERATED_IMAGE_MAX_AGE, immutable=True)
def interpolate_view(request, seed_a: int, seed_b: int):
    """A horizontal strip morphing one seed into another."""
    psi = _clamp_psi(request.GET.get("truncation"))
    steps = _clamp_int(request.GET.get("steps"), INTERPOLATION_STEPS, 2, 12)
    try:
        frames = interpolate(seed_a, seed_b, steps=steps, truncation=psi)
    except ModelUnavailable as exc:
        return HttpResponse(str(exc), status=503, content_type="text/plain")

    from PIL import Image

    w, h = frames[0].size
    strip = Image.new("RGB", (w * len(frames), h), (255, 255, 255))
    for i, frame in enumerate(frames):
        strip.paste(frame, (i * w, 0))
    return HttpResponse(to_png_bytes(strip), content_type="image/png")


@require_POST
@ratelimit(key="ip", rate="60/m", block=True)
def like_view(request, seed: int, psi: int):
    """Save a seed to the gallery. Idempotent per (seed, psi)."""
    entry = Generation.objects.like(seed=seed, truncation=psi / 100.0)
    return render(request, "partials/like.html", {"entry": entry})


@require_GET
def healthz(request):
    """Liveness probe that also reports whether the model actually loaded."""
    ok = is_available()
    return JsonResponse({"status": "ok" if ok else "degraded", "model": ok}, status=200 if ok else 503)
