from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, responses
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

from commons.utils import get_new_uuid
from template.protocol import Completion, RankingRequest
# from sse_starlette.sse import EventSourceResponse
# from tenacity import (
#     AsyncRetrying,
#     RetryError,
#     retry_if_exception_cause_type,
#     retry_if_exception_type,
#     stop_after_attempt,
# )

# from commons.exceptions import ResponseFalselyMarkedAsCompletedException
# from commons.llm.agent import Agent
# from commons.llm.observability import Observability
# from commons.orm.chat_history_orm import ChatHistoryORM
# from commons.utils.environment import get_environment_name
# from commons.utils.logging import log_retry_info

load_dotenv()


evals_router = APIRouter(prefix="/api/evals")


class EvalsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str
    completions: List[str] = Field("", example="Chat group id")
    media_type: str = Field(
        ..., example="Media type of the request", regex="^(text|image)$"
    )


@evals_router.post("/text")
async def eval_text(request: EvalsRequest):
    from neurons.validator import validator

    # build request
    synapse = RankingRequest(
        n_completions=len(request.completions),
        pid=get_new_uuid(),
        prompt=request.prompt,
        completions=[Completion(text=c) for c in request.completions],
    )

    # TODO unsure if we can do this, since it will conflict with synapse
    response = await validator.forward(synapse)
    response_json = jsonable_encoder(response)
    return responses.JSONResponse(content=response_json)
