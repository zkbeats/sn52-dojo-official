# represents a single request sent from the validator to miners
from pydantic import BaseModel

from template.protocol import RankingRequest, RankingResult


class RankingData(BaseModel):
    class Config:
        allow_mutation = False

    request: RankingRequest
    result: RankingResult
