from datetime import datetime
from typing import Dict, List, Optional, Union

import bittensor as bt
from pydantic import BaseModel, Field
from strenum import StrEnum

# from commons.dataset.synthetic import CodeAnswer
from commons.llm.openai_proxy import Provider
from commons.utils import get_epoch_time, get_new_uuid


class TaskType(StrEnum):
    DIALOGUE = "dialogue"
    TEXT_TO_IMAGE = "image"
    CODE_GENERATION = "code_generation"


class CriteriaTypeEnum(StrEnum):
    RANKING_CRITERIA = "ranking"
    MULTI_SCORE = "multi-score"


class RankingCriteria(BaseModel):
    class Config:
        allow_mutation = False

    type: str = CriteriaTypeEnum.RANKING_CRITERIA.value
    options: List[str] = Field(
        description="List of options human labeller will see", default=[]
    )


class MultiScoreCriteria(BaseModel):
    class Config:
        allow_mutation = False

    type: str = CriteriaTypeEnum.MULTI_SCORE.value
    options: List[str] = Field(
        default=[], description="List of options human labeller will see"
    )
    min: float = Field(description="Minimum score for the task")
    max: float = Field(description="Maximum score for the task")


CriteriaType = Union[MultiScoreCriteria, RankingCriteria]


class ScoringMethod(StrEnum):
    HF_MODEL = "hf_model"
    LLM_API = "llm_api"
    AWS_MTURK = "aws_mturk"
    DOJO = "dojo"


# higher value in this map are priortised and allowed to override data on the miner side
SCORING_METHOD_PRIORITY: Dict[ScoringMethod, int] = {
    ScoringMethod.HF_MODEL: 1,
    ScoringMethod.LLM_API: 0,
    ScoringMethod.AWS_MTURK: 2,
    ScoringMethod.DOJO: 3,
}


# # NOTE refactored and p[laced into]
# class Completion(CodeAnswer):
#     class Config:
#         allow_mutation = False

#     cid: str = Field(
#         default_factory=get_new_uuid,
#         description="Unique identifier for the completion",
#     )
#     model_id: str = Field(description="Model that generated the completion")
#     text: str = Field(description="Text of the completion")
#     rank_id: int = Field(description="Rank of the completion", examples=[1, 2, 3, 4])
#     score: float = Field("Score of the completion")


class FileObject(BaseModel):
    filename: str = Field(description="Name of the file")
    content: str = Field(description="Content of the file which can be code or json")
    language: str = Field(description="Programming language of the file")


class CodeAnswer(BaseModel):
    files: List[FileObject] = Field(description="List of FileObjects")
    installation_commands: str = Field(
        description="Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run"
    )
    additional_notes: Optional[str] = Field(
        description="Any additional notes or comments about the code solution"
    )


class Response(BaseModel):
    model: str = Field(description="Model that generated the completion")
    completion: CodeAnswer

    cid: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the completion",
    )
    rank_id: Optional[int] = Field(
        description="Rank of the completion", examples=[1, 2, 3, 4]
    )
    score: Optional[float] = Field(description="Score of the completion")


class SyntheticQA(BaseModel):
    prompt: str
    responses: List[Response]


class ModelConfig(BaseModel):
    provider: Optional[Provider]
    model_name: str


class AWSCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    access_expiration: datetime
    environment: str


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
    responses: List[Response] = Field(
        description="List of completions for the prompt",
        allow_mutation=False,
    )
    task_type: str = Field(description="Type of task", allow_mutation=False)
    criteria_types: List[CriteriaType] = Field(
        description="Types of criteria for the task",
        allow_mutation=False,
    )
    scoring_method: Optional[str] = Field(
        decscription="Method to use for scoring completions"
    )
    mturk_hit_id: Optional[str] = Field(description="MTurk HIT ID for the request")
    dojo_task_id: Optional[str] = Field(description="Dojo task ID for the request")
    aws_credentials: Optional[AWSCredentials] = Field(
        description="Temporary AWS credentials from the miner that validator can use to verify task completions"
    )


class ScoringResult(bt.Synapse):
    request_id: str = Field(
        description="Unique identifier for the request",
        allow_mutation=False,
    )
    hotkey_to_scores: Dict[str, float] = Field(
        description="Hotkey to score mapping", allow_mutation=False
    )


class DendriteQueryResponse(BaseModel):
    class Config:
        allow_mutation = True

    request: FeedbackRequest
    miner_responses: List[FeedbackRequest]


class ScoreItem(BaseModel):
    model_id: str
    score: float


# meant to be used with JSON mode
class ScoresResponse(BaseModel):
    scores: List[ScoreItem]


class PreferenceResponse(BaseModel):
    preference_text: int
