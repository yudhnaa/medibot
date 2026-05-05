from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from django.db import transaction

from rag_benchmark.models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkSplit,
    DatasetSourceFormat,
    ExpectedBehavior,
    ExpectedMode,
)
from rag_benchmark.services.constants import (
    LIST_FIELDS,
    REQUIRED_CASE_FIELDS,
    REQUIRED_SCENARIOS,
)


@dataclass(slots=True)
class DatasetImportReport:
    total_cases: int
    split_counts: dict[str, int]
    scenario_counts: dict[str, int]
    missing_scenarios: list[str]


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Dataset validation failed:\n- " + "\n- ".join(errors)
        super().__init__(message)


class BenchmarkDatasetImporter:
    """Import benchmark dataset cases from JSONL into benchmark tables."""

    def __init__(self, *, require_all_scenarios: bool = False) -> None:
        self.require_all_scenarios = require_all_scenarios

    def import_jsonl(
        self,
        *,
        name: str,
        version: str,
        file_obj: Any,
        description: str = "",
        source_format: str = DatasetSourceFormat.JSONL,
        schema_version: str = "1.0",
        activate: bool = False,
    ) -> tuple[BenchmarkDataset, DatasetImportReport]:
        raw_cases = self._read_jsonl(file_obj=file_obj)
        errors = self._validate_cases(raw_cases, version=version)
        if errors:
            raise DatasetValidationError(errors)

        report = self._build_report(raw_cases)
        if report.missing_scenarios and self.require_all_scenarios:
            missing = ", ".join(report.missing_scenarios)
            raise DatasetValidationError(
                [
                    (
                        "Dataset is missing required scenarios: "
                        f"{missing}. Add cases for full benchmark coverage."
                    )
                ]
            )

        with transaction.atomic():
            dataset, _ = BenchmarkDataset.objects.update_or_create(
                name=name.strip(),
                version=version.strip(),
                defaults={
                    "description": description.strip(),
                    "source_format": source_format,
                    "schema_version": schema_version.strip() or "1.0",
                    "total_cases": report.total_cases,
                    "split_counts": report.split_counts,
                    "composition_stats": {
                        "scenario_counts": report.scenario_counts,
                        "missing_scenarios": report.missing_scenarios,
                    },
                },
            )

            dataset.cases.all().delete()
            BenchmarkCase.objects.bulk_create(
                [
                    BenchmarkCase(
                        dataset=dataset,
                        case_id=str(item["case_id"]).strip(),
                        dataset_version=str(item["dataset_version"]).strip(),
                        split=str(item["split"]).strip().lower(),
                        question=str(item["question"]).strip(),
                        intake_payload=self._safe_dict(item.get("intake_payload", {})),
                        scenario=str(item["scenario"]).strip(),
                        expected_mode=str(item["expected_mode"]).strip(),
                        gold_titles=self._coerce_string_list(
                            item.get("gold_titles", [])
                        ),
                        forbidden_titles=self._coerce_string_list(
                            item.get("forbidden_titles", [])
                        ),
                        must_have_sections=self._coerce_string_list(
                            item.get("must_have_sections", [])
                        ),
                        expected_behavior=str(item["expected_behavior"]).strip(),
                        reference_answer=str(item.get("reference_answer", "")).strip(),
                        notes=str(item.get("notes", "")).strip(),
                        gold_analysis=self._safe_dict(item.get("gold_analysis", {})),
                        gold_primary_title=str(
                            item.get("gold_primary_title", "")
                        ).strip(),
                        must_not_sections=self._coerce_string_list(
                            item.get("must_not_sections", [])
                        ),
                        reference_context_ids=self._coerce_string_list(
                            item.get("reference_context_ids", [])
                        ),
                        requires_followup_topic=str(
                            item.get("requires_followup_topic", "")
                        ).strip(),
                        risk_level=str(item.get("risk_level", "")).strip().lower(),
                        annotation_metadata=self._safe_dict(
                            item.get("annotation_metadata", {})
                        ),
                        is_active=bool(item.get("is_active", True)),
                    )
                    for item in raw_cases
                ],
                batch_size=200,
            )
            if activate:
                dataset.activate()

        return dataset, report

    def _read_jsonl(self, *, file_obj: Any) -> list[dict[str, Any]]:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        payload = file_obj.read()
        if isinstance(payload, bytes):
            text = payload.decode("utf-8-sig")
        else:
            text = str(payload)

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(item, dict):
                errors.append(f"Line {line_no}: JSON object is required")
                continue
            rows.append(item)

        if errors:
            raise DatasetValidationError(errors)
        if not rows:
            raise DatasetValidationError(["JSONL does not contain any benchmark case"])
        return rows

    def _validate_cases(
        self,
        cases: list[dict[str, Any]],
        *,
        version: str,
    ) -> list[str]:
        errors: list[str] = []
        seen_case_ids: set[str] = set()
        split_counter: Counter[str] = Counter()

        valid_modes = {item.value for item in ExpectedMode}
        valid_behaviors = {item.value for item in ExpectedBehavior}
        valid_splits = {item.value for item in BenchmarkSplit}

        expected_version = version.strip()
        for index, case in enumerate(cases, start=1):
            tag = f"Case #{index}"
            if self._add_required_field_errors(case, tag, errors):
                continue

            case_id = self._validate_case_id(case, tag, seen_case_ids, errors)
            seen_case_ids.add(case_id)

            self._validate_case_version(case, tag, expected_version, errors)
            split = self._validate_case_split(case, tag, valid_splits, errors)
            if split in valid_splits:
                split_counter[split] += 1

            self._validate_case_enum(case, tag, "expected_mode", valid_modes, errors)
            self._validate_case_enum(
                case, tag, "expected_behavior", valid_behaviors, errors
            )
            self._validate_list_fields(case, tag, errors)
            self._validate_text_field(case, tag, "question", errors)
            self._validate_text_field(case, tag, "scenario", errors)
            self._validate_intake_payload(case, tag, errors)

        if split_counter:
            missing_splits = sorted(valid_splits.difference(split_counter.keys()))
            if missing_splits:
                errors.append(
                    (
                        "Dataset must include both dev/test split. Missing: "
                        f"{', '.join(missing_splits)}"
                    )
                )

        return errors

    def _add_required_field_errors(
        self,
        case: dict[str, Any],
        tag: str,
        errors: list[str],
    ) -> bool:
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
        if missing:
            errors.append(f"{tag}: missing fields {', '.join(missing)}")
            return True
        return False

    def _validate_case_id(
        self,
        case: dict[str, Any],
        tag: str,
        seen_case_ids: set[str],
        errors: list[str],
    ) -> str:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            errors.append(f"{tag}: case_id must be non-empty")
        elif case_id in seen_case_ids:
            errors.append(f"{tag}: duplicate case_id `{case_id}`")
        return case_id

    def _validate_case_version(
        self,
        case: dict[str, Any],
        tag: str,
        expected_version: str,
        errors: list[str],
    ) -> None:
        dataset_version = str(case.get("dataset_version", "")).strip()
        if not dataset_version:
            errors.append(f"{tag}: dataset_version must be non-empty")
        elif dataset_version != expected_version:
            errors.append(
                (
                    f"{tag}: dataset_version `{dataset_version}` does not match "
                    f"import version `{expected_version}`"
                )
            )

    def _validate_case_split(
        self,
        case: dict[str, Any],
        tag: str,
        valid_splits: set[str],
        errors: list[str],
    ) -> str:
        split = str(case.get("split", "")).strip().lower()
        if split not in valid_splits:
            errors.append(f"{tag}: invalid split `{split}`")
        return split

    def _validate_case_enum(
        self,
        case: dict[str, Any],
        tag: str,
        field: str,
        valid_values: set[str],
        errors: list[str],
    ) -> None:
        value = str(case.get(field, "")).strip()
        if value not in valid_values:
            errors.append(f"{tag}: invalid {field} `{value}`")

    def _validate_list_fields(
        self,
        case: dict[str, Any],
        tag: str,
        errors: list[str],
    ) -> None:
        for field in LIST_FIELDS:
            value = case.get(field, [])
            if value is not None and not isinstance(value, list):
                errors.append(f"{tag}: `{field}` must be a list")

    def _validate_text_field(
        self,
        case: dict[str, Any],
        tag: str,
        field: str,
        errors: list[str],
    ) -> None:
        if not str(case.get(field, "")).strip():
            errors.append(f"{tag}: {field} must be non-empty")

    def _validate_intake_payload(
        self,
        case: dict[str, Any],
        tag: str,
        errors: list[str],
    ) -> None:
        intake_payload = case.get("intake_payload", {})
        if intake_payload is not None and not isinstance(intake_payload, dict):
            errors.append(f"{tag}: intake_payload must be an object")

    def _build_report(self, cases: list[dict[str, Any]]) -> DatasetImportReport:
        split_counter = Counter(
            str(case.get("split", "")).strip().lower() for case in cases
        )
        scenario_counter = Counter(
            str(case.get("scenario", "")).strip() for case in cases
        )
        missing_scenarios = sorted(
            scenario
            for scenario in REQUIRED_SCENARIOS
            if scenario not in scenario_counter
        )
        return DatasetImportReport(
            total_cases=len(cases),
            split_counts={key: int(value) for key, value in split_counter.items()},
            scenario_counts={
                key: int(value) for key, value in scenario_counter.items()
            },
            missing_scenarios=missing_scenarios,
        )

    def _coerce_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            normalized = str(item).strip()
            if normalized:
                out.append(normalized)
        return out

    def _safe_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}
