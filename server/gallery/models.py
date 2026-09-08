"""Persisted generations -- the "hall of fame" of seeds people liked."""

from django.db import models


class GenerationManager(models.Manager):
    def like(self, seed: int, truncation: float) -> "Generation":
        """Record a like, creating the row on first sight.

        F() rather than read-modify-write so concurrent likes on the same seed
        do not lose counts.
        """
        entry, created = self.get_or_create(
            seed=seed, truncation=round(truncation, 2), defaults={"likes": 1}
        )
        if not created:
            self.filter(pk=entry.pk).update(likes=models.F("likes") + 1)
            entry.refresh_from_db(fields=["likes"])
        return entry

    def top(self, limit: int = 12):
        return self.filter(likes__gt=0).order_by("-likes", "-created_at")[:limit]


class Generation(models.Model):
    seed = models.BigIntegerField()
    truncation = models.FloatField(default=0.8)
    likes = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = GenerationManager()

    class Meta:
        # (seed, truncation) fully determines the image, so it is the identity.
        constraints = [
            models.UniqueConstraint(fields=["seed", "truncation"], name="unique_seed_truncation")
        ]
        indexes = [models.Index(fields=["-likes", "-created_at"])]
        ordering = ["-likes", "-created_at"]

    def __str__(self) -> str:
        return f"seed {self.seed} @ psi {self.truncation} ({self.likes} likes)"

    @property
    def psi_pct(self) -> int:
        return round(self.truncation * 100)
