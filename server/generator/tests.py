"""End-to-end tests for the generator and gallery.

Anything touching a route that renders an image is skipped when no ONNX model
is present, so a fresh checkout still runs green before the first training run.
"""

from __future__ import annotations

import io
import unittest

from django.test import Client, TestCase
from django.urls import reverse
from PIL import Image

from gallery.models import Generation

from . import inference

MODEL_AVAILABLE = inference.is_available()
needs_model = unittest.skipUnless(MODEL_AVAILABLE, "no ONNX generator exported yet")


class PageTests(TestCase):
    def test_index_renders(self):
        response = self.client.get(reverse("generator:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Neural Pok")

    def test_index_carries_disclaimer(self):
        """The IP disclaimer is a requirement, not decoration -- assert it ships."""
        response = self.client.get(reverse("generator:index"))
        self.assertContains(response, "not affiliated with")

    def test_healthz(self):
        response = self.client.get(reverse("generator:healthz"))
        self.assertEqual(response.status_code, 200 if MODEL_AVAILABLE else 503)
        self.assertEqual(response.json()["model"], MODEL_AVAILABLE)


class GenerateEndpointTests(TestCase):
    def test_returns_cards(self):
        response = self.client.post(reverse("generator:generate"), {"count": "4", "truncation": "0.8"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.count(b"<figure"), 4)

    def test_json_negotiation(self):
        response = self.client.post(
            reverse("generator:generate"),
            {"count": "3", "seed": "7", "truncation": "0.5"},
            headers={"Accept": "application/json"},
        )
        results = response.json()["results"]
        self.assertEqual([r["seed"] for r in results], [7, 8, 9])
        self.assertEqual(results[0]["psi_pct"], 50)

    def test_explicit_seed_is_honoured(self):
        response = self.client.post(
            reverse("generator:generate"),
            {"count": "1", "seed": "12345"},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response.json()["results"][0]["seed"], 12345)

    def test_hostile_input_is_clamped_not_fatal(self):
        response = self.client.post(
            reverse("generator:generate"),
            {"count": "9999", "truncation": "not-a-float", "seed": "drop table"},
        )
        self.assertEqual(response.status_code, 200)
        # count clamps to GENERATOR_MAX_BATCH; the junk seed falls back to random
        self.assertLessEqual(response.content.count(b"<figure"), 8)

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get(reverse("generator:generate")).status_code, 405)


@needs_model
class ImageRenderTests(TestCase):
    def url(self, seed=42, psi=80):
        return reverse("generator:image", kwargs={"seed": seed, "psi": psi})

    def test_renders_png(self):
        response = self.client.get(self.url())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/png")
        image = Image.open(io.BytesIO(response.content))
        self.assertEqual(image.size, (256, 256))
        self.assertEqual(image.mode, "RGB")

    def test_is_deterministic(self):
        """The permalink and the immutable cache header both depend on this."""
        first = self.client.get(self.url()).content
        second = self.client.get(self.url()).content
        self.assertEqual(first, second)

    def test_different_seeds_differ(self):
        self.assertNotEqual(
            self.client.get(self.url(seed=42)).content,
            self.client.get(self.url(seed=43)).content,
        )

    def test_truncation_changes_output(self):
        self.assertNotEqual(
            self.client.get(self.url(psi=80)).content,
            self.client.get(self.url(psi=30)).content,
        )

    def test_immutable_cache_header(self):
        cache_control = self.client.get(self.url())["Cache-Control"]
        self.assertIn("immutable", cache_control)
        self.assertIn("max-age=31536000", cache_control)

    def test_transparent_background(self):
        response = self.client.get(self.url() + "?bg=transparent")
        image = Image.open(io.BytesIO(response.content))
        self.assertEqual(image.mode, "RGBA")

    def test_interpolation_strip(self):
        response = self.client.get(
            reverse("generator:interpolate", kwargs={"seed_a": 1, "seed_b": 2}) + "?steps=6"
        )
        self.assertEqual(response.status_code, 200)
        image = Image.open(io.BytesIO(response.content))
        self.assertEqual(image.size, (256 * 6, 256))


class GalleryTests(TestCase):
    def test_like_creates_then_increments(self):
        url = reverse("generator:like", kwargs={"seed": 42, "psi": 80})
        self.client.post(url)
        self.client.post(url)
        entry = Generation.objects.get(seed=42, truncation=0.8)
        self.assertEqual(entry.likes, 2)

    def test_seed_and_truncation_are_the_identity(self):
        """Same seed at a different psi is a different image, so a different row."""
        self.client.post(reverse("generator:like", kwargs={"seed": 42, "psi": 80}))
        self.client.post(reverse("generator:like", kwargs={"seed": 42, "psi": 30}))
        self.assertEqual(Generation.objects.filter(seed=42).count(), 2)

    def test_top_orders_by_likes_and_excludes_unliked(self):
        Generation.objects.create(seed=1, truncation=0.8, likes=5)
        Generation.objects.create(seed=2, truncation=0.8, likes=9)
        Generation.objects.create(seed=3, truncation=0.8, likes=0)
        self.assertEqual([g.seed for g in Generation.objects.top()], [2, 1])

    def test_hall_of_fame_appears_on_index(self):
        Generation.objects.create(seed=99, truncation=0.8, likes=3)
        self.assertContains(self.client.get(reverse("generator:index")), "Hall of fame")

    def test_psi_pct_round_trips(self):
        self.assertEqual(Generation(seed=1, truncation=0.8).psi_pct, 80)


class InferenceUnitTests(TestCase):
    def test_latents_are_seed_deterministic(self):
        import numpy as np

        a = inference.latents(seed=5, count=2, dim=16)
        b = inference.latents(seed=5, count=2, dim=16)
        c = inference.latents(seed=6, count=2, dim=16)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_slerp_endpoints(self):
        import numpy as np

        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(inference.slerp(a, b, 0.0), a, atol=1e-6)
        np.testing.assert_allclose(inference.slerp(a, b, 1.0), b, atol=1e-6)

    def test_slerp_preserves_norm(self):
        """Linear interpolation would dip in norm mid-path; slerp must not."""
        import numpy as np

        rng = np.random.default_rng(0)
        a, b = rng.standard_normal((1, 64)), rng.standard_normal((1, 64))
        norms = [np.linalg.norm(inference.slerp(a, b, t)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertLess(max(norms) - min(norms), 0.35 * np.mean(norms))

    def test_white_matte_makes_background_transparent(self):
        import numpy as np

        rgb = np.full((8, 8, 3), 255, dtype=np.uint8)
        rgb[2:6, 2:6] = [200, 30, 40]
        rgba = np.asarray(inference._matte_white(rgb))
        self.assertEqual(rgba[0, 0, 3], 0)      # corner is background
        self.assertEqual(rgba[4, 4, 3], 255)    # subject is opaque
