"""
Management command to seed default ChatbotConfig data.
Usage: python manage.py seed_config
"""

from typing import override
from django.core.management.base import BaseCommand

from chatbot.management.commands.constants import (
    CONFIG_DESCRIPTIONS,
    DEFAULT_FEATURE_FLAGS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PROCESSING_CONFIG,
    DEFAULT_RAG_CONFIG,
    DEFAULT_RATE_LIMITS,
)
from chatbot.models import ChatbotConfig, ConfigCategory


class Command(BaseCommand):
    help = "Seed default ChatbotConfig data for the chatbot"

    @override
    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force update existing configs",
        )

    @override
    def handle(self, *args, **options):
        force = options["force"]

        configs = [
            (
                ChatbotConfig.KEY_RAG_CONFIG,
                DEFAULT_RAG_CONFIG,
                ConfigCategory.RAG,
                CONFIG_DESCRIPTIONS["RAG_CONFIG"],
            ),
            (
                ChatbotConfig.KEY_MODEL_CONFIG,
                DEFAULT_MODEL_CONFIG,
                ConfigCategory.MODEL,
                CONFIG_DESCRIPTIONS["MODEL_CONFIG"],
            ),
            (
                ChatbotConfig.KEY_RATE_LIMITS,
                DEFAULT_RATE_LIMITS,
                ConfigCategory.RATE_LIMIT,
                CONFIG_DESCRIPTIONS["RATE_LIMITS"],
            ),
            (
                ChatbotConfig.KEY_FEATURE_FLAGS,
                DEFAULT_FEATURE_FLAGS,
                ConfigCategory.FEATURE,
                CONFIG_DESCRIPTIONS["FEATURE_FLAGS"],
            ),
            (
                ChatbotConfig.KEY_PROCESSING_CONFIG,
                DEFAULT_PROCESSING_CONFIG,
                ConfigCategory.PROCESSING,
                CONFIG_DESCRIPTIONS["PROCESSING_CONFIG"],
            ),
        ]

        for key, value, category, description in configs:
            config, created = ChatbotConfig.objects.get_or_create(
                key=key,
                defaults={
                    "value": value,
                    "category": category,
                    "description": description,
                    "is_active": True,
                },
            )

            if not created and force:
                config.value = value
                config.category = category
                config.description = description
                config.save()
                self.stdout.write(self.style.WARNING(f"Updated: {key}"))
            elif created:
                self.stdout.write(self.style.SUCCESS(f"Created: {key}"))
            else:
                self.stdout.write(self.style.NOTICE(f"Skipped (exists): {key}"))

        self.stdout.write(self.style.SUCCESS("\nSeed config completed!"))
