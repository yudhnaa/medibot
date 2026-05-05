import json

from django.contrib import admin
from django.utils.html import format_html

from vision.models import XRayAnalysis, XRayEmbedding


@admin.register(XRayAnalysis)
class XRayAnalysisAdmin(admin.ModelAdmin):
    list_display = ("pred_label", "user", "created_at")
    list_filter = ("pred_label", "created_at")
    search_fields = ("user__username", "user__email", "pred_label")
    readonly_fields = (
        "created_at",
        "updated_at",
        "findings_pretty",
        "class_probs_pretty",
    )
    fieldsets = (
        (None, {"fields": ("user", "image", "pred_label")}),
        (
            "Analysis Output",
            {
                "fields": ("findings_pretty", "class_probs_pretty"),
                "classes": ("collapse",),
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def findings_pretty(self, obj: XRayAnalysis) -> str:
        """Helper to pretty-print findings JSON."""
        if not obj.findings:
            return ""
        try:
            return format_html("<pre>{}</pre>", json.dumps(obj.findings, indent=2))
        except Exception:
            return str(obj.findings)

    findings_pretty.short_description = "Findings"

    def class_probs_pretty(self, obj: XRayAnalysis) -> str:
        """Helper to pretty-print class probabilities JSON."""
        if not obj.class_probs:
            return ""
        try:
            return format_html("<pre>{}</pre>", json.dumps(obj.class_probs, indent=2))
        except Exception:
            return str(obj.class_probs)

    class_probs_pretty.short_description = "Class Probabilities"


@admin.register(XRayEmbedding)
class XRayEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("image_id", "label")
    list_filter = ("label",)
    search_fields = ("image_id", "label")
    readonly_fields = ("image_id", "label", "embedding", "metadata_pretty")

    def metadata_pretty(self, obj: XRayEmbedding) -> str:
        """Helper to pretty-print metadata JSON."""
        if not obj.metadata:
            return ""
        try:
            return format_html("<pre>{}</pre>", json.dumps(obj.metadata, indent=2))
        except Exception:
            return str(obj.metadata)

    metadata_pretty.short_description = "Metadata"
