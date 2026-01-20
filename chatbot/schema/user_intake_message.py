from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import Field, model_validator

from chatbot.schema.constants import (
    MAX_AGE,
    MAX_DISEASE_NAME_LENGTH,
    MAX_ONSET_DAYS,
    Pregnancy,
    Sex,
)


class UserIntakeMessage(BaseMessage):
    type: Literal["user_intake"] = (
        "user_intake"  # pyright: ignore[reportIncompatibleVariableOverride]
    )

    disease_name: str | None = Field(default=None, max_length=MAX_DISEASE_NAME_LENGTH)
    symptoms: list[str] = Field(default_factory=list)
    # Store negated symptoms separately to avoid mixing with positive symptoms
    symptoms_negated: list[str] = Field(default_factory=list)
    age: int | None = Field(default=None, ge=0, le=MAX_AGE)

    sex: Sex = "unknown"
    onset_days: int | None = Field(default=None, ge=0, le=MAX_ONSET_DAYS)
    chronic_conditions: list[str] = Field(default_factory=list)
    meds: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    pregnancy_status: Pregnancy | None = None
    location_country: str | None = None

    id: UUID = Field(
        default_factory=uuid4
    )  # pyright: ignore[reportIncompatibleVariableOverride]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    content: str = ""
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _auto_fill(self) -> "UserIntakeMessage":
        if not self.content:
            disease = self.disease_name or "Unknown disease"
            symptoms_str = ", ".join(self.symptoms) if self.symptoms else "None"
            neg_symptoms_str = (
                ", ".join(self.symptoms_negated) if self.symptoms_negated else "None"
            )
            age_str = f", Age: {self.age}" if self.age is not None else ""
            self.content = f"Intake: {disease}, Symptoms(+): {symptoms_str}, Symptoms(-): {neg_symptoms_str}{age_str}"  # pyright: ignore[reportIncompatibleVariableOverride]
        self.additional_kwargs.update(
            {
                "kind": "user_intake",
                "disease_name": self.disease_name,
                "symptoms": self.symptoms,
                "symptoms_negated": self.symptoms_negated,
                "age": self.age,
                "sex": self.sex,
                "onset_days": self.onset_days,
                "chronic_conditions": self.chronic_conditions,
                "meds": self.meds,
                "allergies": self.allergies,
                "pregnancy_status": self.pregnancy_status,
                "location_country": self.location_country,
                "created_at": self.created_at.isoformat(),
                "id": str(self.id),
            }
        )
        return self

    def to_human_message(self) -> HumanMessage:
        return HumanMessage(
            content=self.content, additional_kwargs=self.additional_kwargs
        )

    def to_rag_query(self) -> str:
        parts = []
        if self.disease_name:
            parts.append(f"disease:{self.disease_name}")
        if self.symptoms:
            parts.append("symptoms:" + ", ".join(self.symptoms))
        if self.age is not None:
            parts.append(f"age:{self.age}")
        if self.sex and self.sex != "unknown":
            parts.append(f"sex:{self.sex}")
        if self.onset_days is not None:
            parts.append(f"onset_days:{self.onset_days}")
        return " | ".join(parts) if parts else "user_intake"
