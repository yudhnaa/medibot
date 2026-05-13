import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, override

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chatbot.models import IndexType, MedicalDocument, SectionType

FIELDS = [
    "title",
    "general",
    "symptom",
    "aetiologies",
    "risk",
    "diagnose_and_treaty",
    "living_and_preventive",
    "url",
]

SECTION_FIELD_BY_TYPE = {
    SectionType.GENERAL: "general",
    SectionType.SYMPTOM: "symptom",
    SectionType.AETIOLOGIES: "aetiologies",
    SectionType.RISK: "risk",
    SectionType.DIAGNOSE_AND_TREATY: "diagnose_and_treaty",
    SectionType.LIVING_AND_PREVENTIVE: "living_and_preventive",
}


class Command(BaseCommand):
    help = "Export text-only disease knowledge from vector database to CSV"

    @override
    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            help="Filter by MedicalDocument.source",
        )

    @override
    def handle(self, *args, **options):
        output_path = self._default_output_path()
        source = options.get("source")

        queryset = MedicalDocument.objects.filter(index_type=IndexType.B).order_by(
            "title",
            "section_type",
            "id",
        )
        if source:
            queryset = queryset.filter(source=source)

        rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for document in queryset.iterator():
            metadata = document.metadata or {}
            key = metadata.get("canonical_title") or document.title
            row = rows.setdefault(
                key,
                {
                    "title": metadata.get("title") or document.title,
                    "general": "",
                    "symptom": "",
                    "aetiologies": "",
                    "risk": "",
                    "diagnose_and_treaty": "",
                    "living_and_preventive": "",
                    "url": metadata.get("url", ""),
                },
            )

            field = SECTION_FIELD_BY_TYPE.get(document.section_type)
            if not field:
                continue
            row[field] = self._append_unique(row[field], document.content)
            if not row["url"] and metadata.get("url"):
                row["url"] = metadata["url"]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows.values())
        except OSError as exc:
            raise CommandError(f"Could not write CSV file: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {len(rows)} knowledge-domain rows to {output_path}"
            )
        )

    def _default_output_path(self) -> Path:
        filename = f"dataset_{timezone.localdate().isoformat()}.csv"
        return Path("backups") / "dataset" / filename

    def _append_unique(self, current: str, value: str) -> str:
        text = value.strip()
        if not text:
            return current
        existing = [item.strip() for item in current.split("\n") if item.strip()]
        if text in existing:
            return current
        existing.append(text)
        return "\n".join(existing)
