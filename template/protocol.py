from datetime import datetime
from typing import Dict, List, Optional

import bittensor as bt
from pydantic import BaseModel, Field
from strenum import StrEnum
from commons.dataset.synthetic import CodeAnswer

from commons.llm.openai_proxy import Provider
from commons.utils import get_epoch_time, get_new_uuid


class TaskType(StrEnum):
    DIALOGUE = "dialogue"
    TEXT_TO_IMAGE = "image"
    CODE_GENERATION = "code_generation"


class CriteriaType(StrEnum):
    PREFERENCE_RANKING = "preference_ranking"
    SCORING = "scoring"


class ScoringMethod(StrEnum):
    HF_MODEL = "hf_model"
    LLM_API = "llm_api"
    AWS_MTURK = "aws_mturk"
    DOJO = "dojo_worker"


# higher value in this map are priortised and allowed to override data on the miner side
SCORING_METHOD_PRIORITY: Dict[ScoringMethod, int] = {
    ScoringMethod.HF_MODEL: 1,
    ScoringMethod.LLM_API: 0,
    ScoringMethod.AWS_MTURK: 2,
    ScoringMethod.DOJO: 3,
}


class Completion(CodeAnswer):
    class Config:
        allow_mutation = False

    cid: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the completion",
    )
    model_id: str = Field(description="Model that generated the completion")
    text: str = Field(description="Text of the completion")
    rank_id: int = Field(description="Rank of the completion", examples=[1, 2, 3, 4])
    score: float = Field("Score of the completion")


# class Rank(BaseModel):
#     cid: str = Field(description="Unique identifier for the completion")
#     score: Optional[float] = Field(default=0.0, description="Score of the completion")
#     rank_id: int = Field(description="Rank of the completion", examples=[1, 2, 3, 4])


class ModelConfig(BaseModel):
    provider: Optional[Provider]
    model_name: str


class FeedbackRequest(bt.Synapse):
    epoch_timestamp: float = Field(
        default_factory=get_epoch_time,
        description="Epoch timestamp for the request",
        allow_mutation=False,
    )
    request_id: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the request",
        allow_mutation=False,
    )
    prompt: str = Field(
        description="Prompt or query from the user sent the LLM",
        allow_mutation=False,
    )
    completions: List[Completion] = Field(
        description="List of completions for the prompt",
        allow_mutation=False,
    )
    task_type: TaskType = Field(description="Type of task", allow_mutation=False)
    criteria_types: List[CriteriaType] = Field(
        description="Types of criteria for the task",
        allow_mutation=False,
    )
    scoring_method: Optional[str] = Field(
        decscription="Method to use for scoring completions"
    )
    mturk_hit_id: Optional[str] = Field(description="MTurk HIT ID for the request")
    dojo_task_id: Optional[str] = Field(description="Dojo task ID for the request")


class RankingResult(bt.Synapse):
    request_id: str = Field(
        description="Unique identifier for the request",
        allow_mutation=False,
    )
    cid_to_consensus: Dict[str, float] = Field(
        description="Consensus score for each completion", allow_mutation=False
    )
    hotkey_to_score: Dict[str, float] = Field(
        description="Hotkey to score mapping", allow_mutation=False
    )


class AWSCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    access_expiration: datetime
    environment: str


class MTurkResponse(bt.Synapse):
    mturk_hit_id: str
    aws_credentials: Optional[AWSCredentials]
    completion_id_to_score: Dict[str, float]


class DendriteQueryResponse(BaseModel):
    class Config:
        allow_mutation = True

    request: FeedbackRequest
    responses: List[FeedbackRequest]


class ScoreItem(BaseModel):
    completion_id: str
    score: float


# meant to be used with JSON mode
class ScoresResponse(BaseModel):
    scores: List[ScoreItem]


class PreferenceResponse(BaseModel):
    preference_text: int
