from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from rag_benchmark.services import BenchmarkDatasetImporter, DatasetValidationError


class Command(BaseCommand):
    help = "Import benchmark dataset from JSONL into rag_benchmark tables"

    def add_arguments(self, parser):
        parser.add_argument("--name", required=True, help="Dataset name")
        parser.add_argument("--version", required=True, help="Dataset version")
        parser.add_argument("--file", required=True, help="Path to JSONL file")
        parser.add_argument("--description", default="", help="Dataset description")
        parser.add_argument("--schema-version", default="1.0", help="Schema version")
        parser.add_argument(
            "--activate",
            action="store_true",
            default=False,
            help="Activate imported dataset version",
        )

    def handle(self, *args, **options):
        file_path = options["file"]
        importer = BenchmarkDatasetImporter()
        try:
            with open(file_path, "rb") as fp:
                dataset, report = importer.import_jsonl(
                    name=options["name"],
                    version=options["version"],
                    file_obj=fp,
                    description=options["description"],
                    schema_version=options["schema_version"],
                    activate=bool(options["activate"]),
                )
        except FileNotFoundError as exc:
            raise CommandError(f"File not found: {file_path}") from exc
        except DatasetValidationError as exc:
            raise CommandError(str(exc)) from exc
        except Exception as exc:
            raise CommandError(f"Import failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                (
                    f"Imported {dataset.name}:{dataset.version} "
                    f"with {report.total_cases} cases"
                )
            )
        )
        if report.missing_scenarios:
            self.stdout.write(
                self.style.WARNING(
                    "Missing recommended scenarios: "
                    + ", ".join(report.missing_scenarios)
                )
            )
