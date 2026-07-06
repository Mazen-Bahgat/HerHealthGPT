from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Evidence(BaseModel):
    source_url: str
    quote: str


class BaseEntry(BaseModel):
    base_id: str
    condition: str
    symptom: str
    canonical_query: str
    risk_level: RiskLevel
    recommended_action: str
    evidence: list[Evidence] = Field(min_length=1)
    conflict_note: str | None = None


class Provenance(BaseModel):
    generated_by: str
    generated_at: str
    prompt_id: str


class FinalRecord(BaseModel):
    id: str
    base_id: str
    parent_id: str | None = None
    lang: Literal["en", "fr", "ar"]
    variant_type: Literal["canonical", "paraphrase", "translation"]
    query: str
    condition: str
    risk_level: RiskLevel
    recommended_action: str
    evidence: list[Evidence] = Field(min_length=1)
    provenance: Provenance
