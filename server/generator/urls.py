from django.urls import path

from . import views

app_name = "generator"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/generate", views.generate_view, name="generate"),
    # psi is an integer percent so the URL stays a clean cache key.
    path("g/<int:seed>-<int:psi>.png", views.image_view, name="image"),
    path("i/<int:seed_a>-<int:seed_b>.png", views.interpolate_view, name="interpolate"),
    path("api/like/<int:seed>-<int:psi>", views.like_view, name="like"),
    path("healthz", views.healthz, name="healthz"),
]
