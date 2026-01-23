from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbot"

    def ready(self) -> None:
        """Register signal handlers."""
        import chatbot.signals  # noqa: F401
