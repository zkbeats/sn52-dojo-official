from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, responses
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

from commons.utils import get_new_uuid
from template.protocol import Completion, RankingRequest
from neurons.validator import Validator
import bittensor as bt

load_dotenv()


evals_router = APIRouter(prefix="/api/evals")


class EvalsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str = Field(..., description="Prompt that generated the completions")
    completions: List[str] = Field("", description="Chat group id")
    media_type: str = Field(
        ..., description="Media type of the request", regex="^(text|image)$"
    )


@evals_router.post("/text")
async def eval_text(request: EvalsRequest):
    synapse = RankingRequest(
        n_completions=len(request.completions),
        pid=get_new_uuid(),
        prompt=request.prompt,
        completions=[Completion(text=c) for c in request.completions],
    )

    try:
        validator = Validator()
        response = await validator.forward(synapse)
        response_json = jsonable_encoder(response)
        return responses.JSONResponse(content=response_json)
    except Exception as e:
        bt.logging.error("Encountered e")
