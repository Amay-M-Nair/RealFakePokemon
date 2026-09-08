from django.contrib import admin

from .models import Generation


@admin.register(Generation)
class GenerationAdmin(admin.ModelAdmin):
    list_display = ("seed", "truncation", "likes", "created_at")
    ordering = ("-likes",)
