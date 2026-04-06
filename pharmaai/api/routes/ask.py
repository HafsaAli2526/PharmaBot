"""pharmaai/api/routes/ask.py"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from pharmaai.core.schemas import AskRequest, AskResponse
from pharmaai.inference.rag import rag_pipeline

router = APIRouter()
logger = logging.getLogger("pharmaai.api.ask")

@router.post("/ask", response_model=AskResponse, summary="Ask a pharmaceutical question")
async def ask(request: AskRequest) -> AskResponse:
    """
    Ask a question across pharmacovigilance, R&D, regulatory, or formula domains.
    Returns a generated answer with supporting source documents.
    """
    return await rag_pipeline.answer(request)