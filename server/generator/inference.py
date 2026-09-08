"""ONNX Runtime inference for the Pokemon generator.

The session is a module-level singleton created lazily on first use: loading it
at import time would make `manage.py` commands (migrate, collectstatic) pay for
a ~60MB model they never touch, and would break any deploy where the model is
fetched after the image is built.

Generation is seed-deterministic. The same (seed, truncation) always yields the
same image on any machine, which is what lets the web layer treat rendered PNGs
as immutable and cache them forever.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.conf import settings
from PIL import Image

_session = None
_z_mean: np.ndarray | None = None
_lock = threading.Lock()

# Pixels within this distance of pure white become transparent. Generous enough
# to clear the background the model was trained on without eating too far into
# pale subjects.
WHITE_MATTE_TOLERANCE = 22.0


class ModelUnavailable(RuntimeError):
    """The ONNX generator is missing. Raised with instructions, not a traceback."""


@dataclass(frozen=True)
class GenerationRequest:
    seed: int
    truncation: float = 1.0
    count: int = 1
    transparent: bool = False


def _load() -> tuple["object", np.ndarray]:
    global _session, _z_mean
    if _session is not None:
        return _session, _z_mean

    with _lock:
        if _session is not None:  # another thread won the race
            return _session, _z_mean

        model_path = Path(settings.GENERATOR_MODEL_PATH)
        if not model_path.exists():
            raise ModelUnavailable(
                f"No generator at {model_path}. Train one and export it:\n"
                f"  python -m src.train --name fastgan\n"
                f"  python -m src.export_onnx checkpoints/fastgan/final.pt "
                f"--out {model_path} --fp16"
            )

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.GENERATOR_THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        mean_path = Path(settings.GENERATOR_Z_MEAN_PATH)
        z_mean = np.load(mean_path) if mean_path.exists() else None

        _session, _z_mean = session, z_mean
        return _session, _z_mean


def z_dim() -> int:
    session, _ = _load()
    return int(session.get_inputs()[0].shape[1])


def latents(seed: int, count: int, dim: int) -> np.ndarray:
    """Seed-deterministic latents.

    numpy's PCG64 stream is stable across versions and platforms, unlike CUDA's
    RNG -- the permalink promise depends on that.
    """
    return np.random.default_rng(seed).standard_normal((count, dim), dtype=np.float32)


def _run(session, z: np.ndarray, chunk: int) -> np.ndarray:
    """Run the generator in fixed-size chunks and stitch the results.

    Memory, not speed, decides the chunk size. A 29M-parameter generator at
    256x256 allocates large intermediate activations, and they scale with the
    batch: rendering an 8-frame interpolation in one call peaks at ~581MB
    resident, which OOMs a 512MB free tier. Chunking keeps the peak flat at the
    single-image cost (~270MB) and only costs wall time, which the immutable
    cache absorbs after the first request.
    """
    outputs = [
        session.run(["image"], {"z": z[i : i + chunk].astype(np.float32)})[0]
        for i in range(0, len(z), chunk)
    ]
    return np.concatenate(outputs) if len(outputs) > 1 else outputs[0]


def _to_uint8(raw: np.ndarray) -> np.ndarray:
    """NCHW in [-1,1] -> NHWC uint8."""
    out = ((np.clip(raw, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
    return out.transpose(0, 2, 3, 1)


def _apply_truncation(z: np.ndarray, z_mean: np.ndarray | None, truncation: float) -> np.ndarray:
    if truncation == 1.0:
        return z
    mean = np.zeros((1, z.shape[1]), dtype=np.float32) if z_mean is None else z_mean.reshape(1, -1)
    return mean + truncation * (z - mean)


def _matte_white(rgb: np.ndarray) -> Image.Image:
    """Turn the near-white training background into alpha.

    The model was trained on RGB composited over white rather than with a
    learned alpha channel (a near-binary alpha makes tanh generators produce
    halos). Recovering transparency here is the tradeoff: it works well for
    saturated subjects and will nibble at genuinely white ones, which is why
    the opaque version stays the default.
    """
    distance = 255.0 - rgb.min(axis=2)
    alpha = np.clip(distance / WHITE_MATTE_TOLERANCE, 0.0, 1.0)
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def generate(request: GenerationRequest) -> list[Image.Image]:
    """Render images for a request. Deterministic in (seed, truncation, count)."""
    session, z_mean = _load()
    count = max(1, min(request.count, settings.GENERATOR_MAX_BATCH))
    dim = int(session.get_inputs()[0].shape[1])

    z = _apply_truncation(latents(request.seed, count, dim), z_mean, request.truncation)
    out = _to_uint8(_run(session, z, settings.GENERATOR_CHUNK_SIZE))

    if request.transparent:
        return [_matte_white(img) for img in out]
    return [Image.fromarray(img, mode="RGB") for img in out]


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two latents.

    Linear interpolation dips through a low-norm region the generator never saw
    during training, which shows up as washed-out middle frames.
    """
    a_n = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=-1, keepdims=True)
    omega = np.arccos(np.clip((a_n * b_n).sum(-1, keepdims=True), -1.0, 1.0))
    sin_omega = np.sin(omega)
    if np.all(np.abs(sin_omega) < 1e-6):
        return (1 - t) * a + t * b
    return (np.sin((1 - t) * omega) / sin_omega) * a + (np.sin(t * omega) / sin_omega) * b


def interpolate(seed_a: int, seed_b: int, steps: int = 8, truncation: float = 1.0) -> list[Image.Image]:
    session, z_mean = _load()
    dim = int(session.get_inputs()[0].shape[1])
    za, zb = latents(seed_a, 1, dim), latents(seed_b, 1, dim)
    z = np.concatenate([slerp(za, zb, float(t)) for t in np.linspace(0, 1, steps)])

    z = _apply_truncation(z.astype(np.float32), z_mean, truncation)
    out = _to_uint8(_run(session, z, settings.GENERATOR_CHUNK_SIZE))
    return [Image.fromarray(img, mode="RGB") for img in out]


def is_available() -> bool:
    """Whether a model is loadable, for the health check and the UI banner."""
    try:
        _load()
        return True
    except ModelUnavailable:
        return False
