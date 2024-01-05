from typing import Dict, List, Optional
import uuid
from attr import define, field
import bittensor as bt
from pydantic import BaseModel, Field, validator

from commons.utils import get_epoch_time, get_new_uuid

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class Dummy(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by recieving axon.
    dummy_output: Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.dummy_output


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


class RankingRequest(bt.Synapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.
    """

    # Required request input, filled by sending dendrite caller.
    epoch_timestamp: int = Field(
        default=get_epoch_time(),
        description="Epoch timestamp for the request",
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
    # # Optional request output, filled by recieving axon.
    # output: Optional[Output] = None
    ranks: List[Rank] = Field(
        default=[], description="List of ranks for each completion"
    )
