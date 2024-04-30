from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, responses
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field
from commons.factory import Factory

from commons.utils import get_new_uuid
from template.protocol import Completion, FeedbackRequest, ScoringMethod
import bittensor as bt

load_dotenv()


reward_router = APIRouter(prefix="/api/reward_model")


class ExternalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str = Field(..., description="Prompt that generated the completions")
    completions: List[str] = Field(..., description="List of completions")
    media_type: str = Field(
        ..., description="Media type of the request", regex="^(text|image)$"
    )
    # scoring_methods: List[ScoringMethod] = Field(
    #     ..., description="List of scoring methods to use"
    # )
    # NOTE @dev this exists because for huamn feedback methods we need to account for turnaround time
    callback_url: str = Field(
        ...,
        description="URL to send the response to when requesting for specific types of methods",
    )


@reward_router.post("/")
async def reward_request_handler(request: ExternalRequest):
    synapse = FeedbackRequest(
        n_completions=len(request.completions),
        pid=get_new_uuid(),
        prompt=request.prompt,
        completions=[Completion(text=c) for c in request.completions],
    )

    try:
        validator = Factory.get_validator()
        response = await validator.send_request(synapse)
        response_json = jsonable_encoder(response)
        return responses.JSONResponse(content=response_json)
    except Exception as e:
        bt.logging.error(f"Encountered exception: {e}")
