from django.contrib import admin

from chatbot.models import (
    ChatbotConfig,
    ChatMessage,
    ChatSession,
    MedicalDocument,
    UserPreference,
)


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["id", "session_id", "customer", "title", "created_at", "is_active"]
    search_fields = ["session_id", "title", "customer__username"]
    list_filter = ["is_active", "created_at"]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["id", "session", "role", "created_at"]
    search_fields = ["content", "session__session_id"]
    list_filter = ["role", "created_at"]


@admin.register(MedicalDocument)
class MedicalDocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "section_type", "index_type", "source", "created_at"]
    search_fields = ["title", "content"]
    list_filter = ["section_type", "index_type", "source"]


@admin.register(ChatbotConfig)
class ChatbotConfigAdmin(admin.ModelAdmin):
    list_display = ["key", "category", "is_active", "updated_at"]
    search_fields = ["key", "description"]
    list_filter = ["category", "is_active"]


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ["customer", "response_style", "language", "updated_at"]
    search_fields = ["customer__username"]
    list_filter = ["response_style", "language"]
