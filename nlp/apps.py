from typing import override

from django.apps import AppConfig


class NlpConfig(AppConfig):
    name = "nlp"

    @override
    def ready(self) -> None:
        """Load VnCoreNLP at server startup to avoid JVM restart on each call."""
        from nlp.services.text_normalizer import TextNormalizer

        TextNormalizer.warmup()
