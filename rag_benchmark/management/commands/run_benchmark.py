from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from rag_benchmark.models import BenchmarkDataset
from rag_benchmark.services import OfflineBenchmarkRunner


class Command(BaseCommand):
    help = "Run offline benchmark against runtime services for one dataset split"

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="Dataset name")
        parser.add_argument("--version", required=True, help="Dataset version")
        parser.add_argument(
            "--split",
            required=True,
            choices=["dev", "test"],
            help="Dataset split",
        )
        parser.add_argument(
            "--enable-ragas",
            action="store_true",
            default=False,
            help="Enable optional ragas judge layer",
        )

    def handle(self, *args, **options):
        try:
            dataset = BenchmarkDataset.objects.get(
                name=options["dataset"],
                version=options["version"],
            )
        except BenchmarkDataset.DoesNotExist as exc:
            raise CommandError(
                ("Dataset not found: " f"{options['dataset']}:{options['version']}")
            ) from exc

        runner = OfflineBenchmarkRunner()
        run = runner.run(
            dataset=dataset,
            split=options["split"],
            judge_configuration={"enable_ragas": bool(options["enable_ragas"])},
        )
        self.stdout.write(
            self.style.SUCCESS(
                (
                    f"Run completed: {run.run_id} "
                    f"(status={run.status}, primary_pass={run.primary_pass})"
                )
            )
        )
