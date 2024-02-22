from enum import StrEnum
from typing import Dict, List, Optional
import bittensor as bt
from pydantic import BaseModel, Field
from commons.llm.openai_proxy import Provider

from commons.utils import get_epoch_time, get_new_uuid


class ScoringMethod(StrEnum):
    HF_MODEL = "hf_model"
    LLM_API = "llm_api"
    AWS_MTURK = "aws_mturk"


class Completion(BaseModel):
    class Config:
        allow_mutation = False

    cid: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the completion",
    )
    text: str = Field(description="Text of the completion")


class Rank(BaseModel):
    cid: str = Field(description="Unique identifier for the completion")
    score: float = Field(default=0.0, description="Score of the completion")


class ModelConfig(BaseModel):
    provider: Optional[Provider]
    model_name: str


class RankingRequest(bt.Synapse):
    # filled in by validator
    epoch_timestamp: int = Field(
        default_factory=get_epoch_time,
        description="Epoch timestamp for the request",
        allow_mutation=False,
    )
    request_id: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the request",
        allow_mutation=False,
    )
    pid: str = Field(
        default_factory=get_new_uuid,
        description="Unique identifier for the prompt",
    )
    prompt: str = Field(
        description="Prompt or query from the user sent the LLM",
        allow_mutation=False,
    )
    n_completions: int = Field(
        description="Number of completions for miner to score/rank",
        allow_mutation=False,
    )
    completions: List[Completion] = Field(
        description="List of completions for the prompt",
        allow_mutation=False,
    )
    # filled in by miners
    ranks: List[Rank] = Field(
        default=[], description="List of ranks for each completion"
    )
    scoring_method: Optional[ScoringMethod] = Field(
        decscription="Method to use for scoring completions"
    )
    model_config: Optional[ModelConfig] = Field(
        description="Model configuration for Huggingface / LLM API scoring"
    )


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


class MTurkResponse(bt.Synapse):
    completion_id_to_score: Dict[str, float]
