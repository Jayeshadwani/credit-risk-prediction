from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    applicant_id: int
    default_probability: float
    risk_category: str
    model_version: str