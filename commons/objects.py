# represents a single request sent from the validator to miners
from typing import List
from pydantic import BaseModel

from template.protocol import RankingRequest, RankingResult


class DendriteQueryResponse(BaseModel):
    class Config:
        allow_mutation = False

    request: RankingRequest
    result: RankingResult


class ScoreItem(BaseModel):
    completion_id: str
    score: float


# meant to be used with JSON mode
class ScoresResponse(BaseModel):
    scores: List[ScoreItem]
