"""pharmaai/api/routes/classify.py"""
from __future__ import annotations
from fastapi import APIRouter
from pharmaai.core.schemas import ClassifyRequest, ClassifyResponse
from pharmaai.processing.domain_classifier import classifier

router = APIRouter()

@router.post("/classify", response_model=ClassifyResponse, summary="Classify text domain")
async def classify_text(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify text into pharmaceutical domain and content type.
    """
    result = classifier.classify(request.text)
    return ClassifyResponse(
        domain=result.domain,
        content_type=result.content_type,
        confidence=result.confidence,
    )