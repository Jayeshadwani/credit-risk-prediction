from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.predictor import (
    bundle,
    predict_default_probability,
)
from app.report_generator import (
    ReportGroundingError,
    generate_underwriting_report,
)
from app.report_schemas import UnderwritingReport
from app.config import settings
from app.policy_retriever import collection

app = FastAPI(
    title="Loan Underwriting Assistant API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=100,
    )


class UnderwritingReportRequest(BaseModel):
    case_id: str = Field(
        pattern=r"^DEMO-\d{3}$",
        examples=["DEMO-001"],
    )

    model: str | None = Field(
        default=None,
        description="Optional OpenAI model override",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Confirms that the FastAPI process is running.
    """
    return {
        "status": "healthy",
        "service": "loan-underwriting-assistant",
        "api_version": app.version,
    }


@app.get("/ready")
def readiness() -> dict[str, Any]:
    """
    Confirms that required application components are available.
    """

    checks: dict[str, Any] = {}

    # Model check
    model_loaded = bundle.get("model") is not None

    checks["model"] = {
        "ready": model_loaded,
        "model_name": bundle["metadata"]["model_name"],
        "model_version": bundle["model_version"],
    }

    # Feature configuration check
    feature_columns = bundle.get("feature_columns", [])

    checks["features"] = {
        "ready": len(feature_columns) > 0,
        "feature_count": len(feature_columns),
    }

    # Chroma vector-store check
    try:
        policy_chunk_count = collection.count()

        checks["vector_store"] = {
            "ready": policy_chunk_count > 0,
            "collection": settings.policy_collection_name,
            "chunk_count": policy_chunk_count,
        }

    except Exception as error:
        checks["vector_store"] = {
            "ready": False,
            "error": str(error),
        }

    # OpenAI configuration check
    checks["openai"] = {
        "ready": bool(settings.openai_api_key),
        "model": settings.openai_model,
    }

    is_ready = all(
        check["ready"]
        for check in checks.values()
    )

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks,
            },
        )

    return {
        "status": "ready",
        "checks": checks,
    }

@app.post("/predict")
def predict(
    request: PredictionRequest,
    include_explanation: bool = Query(
        default=False,
        description="Include local SHAP explanations",
    ),
) -> dict[str, Any]:
    try:
        predictions = predict_default_probability(
            records=request.records,
            include_explanation=include_explanation,
        )

        return {
            "predictions": predictions,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/underwriting/report",
    response_model=UnderwritingReport,
)
def generate_report(
    request: UnderwritingReportRequest,
) -> UnderwritingReport:
    try:
        report = generate_underwriting_report(
            case_id=request.case_id,
            model_name=request.model,
            save_output=True,
        )

        return report

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ReportGroundingError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "The generated report failed "
                "grounding validation."
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "The underwriting report service "
                "could not complete the request."
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unexpected internal server error.",
        ) from error