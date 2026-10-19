"""User and citizen feedback collection endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import FeedbackRequest, FeedbackResponse
from app.services import save_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    fb_id = save_feedback(request.model_dump())
    return FeedbackResponse(status="recorded", feedback_id=fb_id)
