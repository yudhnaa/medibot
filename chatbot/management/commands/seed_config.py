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

        # 1. Flatten all configs into a single list
        all_configs = []

        # RAG Config
        for key, value in DEFAULT_RAG_CONFIG.items():
            all_configs.append(
                (
                    key,
                    value,
                    ConfigCategory.RAG,
                    CONFIG_DESCRIPTIONS["RAG_CONFIG"],
                )
            )

        # Model Config
        for key, value in DEFAULT_MODEL_CONFIG.items():
            all_configs.append(
                (
                    key,
                    value,
                    ConfigCategory.MODEL,
                    CONFIG_DESCRIPTIONS["MODEL_CONFIG"],
                )
            )

        # Rate Limits
        for key, value in DEFAULT_RATE_LIMITS.items():
            all_configs.append(
                (
                    key,
                    value,
                    ConfigCategory.RATE_LIMIT,
                    CONFIG_DESCRIPTIONS["RATE_LIMITS"],
                )
            )

        # Feature Flags
        for key, value in DEFAULT_FEATURE_FLAGS.items():
            all_configs.append(
                (
                    key,
                    value,
                    ConfigCategory.FEATURE,
                    CONFIG_DESCRIPTIONS["FEATURE_FLAGS"],
                )
            )

        # Processing Config
        for key, value in DEFAULT_PROCESSING_CONFIG.items():
            all_configs.append(
                (
                    key,
                    value,
                    ConfigCategory.PROCESSING,
                    CONFIG_DESCRIPTIONS["PROCESSING_CONFIG"],
                )
            )

        # 2. Iterate and create/update
        for key, value, category, description in all_configs:
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
