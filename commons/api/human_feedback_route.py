from typing import Dict, List
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

import bittensor as bt

load_dotenv()

callback_route = "/api/human_feedback"

human_feedback_router = APIRouter(prefix="/api/human_feedback")


class EvalsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str = Field(..., description="Prompt that generated the completions")
    completions: List[str] = Field("", description="Chat group id")
    media_type: str = Field(
        ..., description="Media type of the request", regex="^(text|image)$"
    )


# this callback url gets called when task gets completed
@human_feedback_router.post("/callback")
async def task_completion_callback(request: Dict):
    bt.logging.info(f"Received task completion callback with body: {request}")
    pass
