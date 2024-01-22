import asyncio
from typing import List

import bittensor as bt
from commons.scoring import Scoring

from commons.utils import get_new_uuid
from template.base.validator import BaseValidatorNeuron
from template.protocol import Completion, RankingRequest, RankingResult
from template.utils.uids import get_random_uids


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

    async def _forward_consensus(self, synapse: RankingResult):
        bt.logging.debug("Sending back consensus to miners")
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = get_random_uids(
            metagraph=self.metagraph,
            k=self.config.neuron.sample_size,
            config=self.config,
        )
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=5,
        )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        # TODO change to real data
        synapse = RankingRequest(
            n_completions=3,
            pid=get_new_uuid(),
            prompt="What is your name?",
            completions=[
                Completion(
                    text="My name is Assistant, and I am a helpful assisstant created by OpenAI.",
                ),
                Completion(
                    text="My name is Llama, and I am an assistant created by Meta.",
                ),
                Completion(
                    text="My name is GPT-3, and I am an AI created by OpenAI.",
                ),
            ],
        )
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = get_random_uids(
            metagraph=self.metagraph,
            k=self.config.neuron.sample_size,
            config=self.config,
        )
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        # The dendrite client queries the network.
        responses: List[RankingRequest] = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            # Construct a dummy query. This simply contains a single integer.
            synapse=synapse,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=5,
        )
        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")
        for r in responses:
            print("process time", r.dendrite.process_time)

        # Adjust the scores based on responses from miners.
        hotkey_to_scores, ranking_result = Scoring.score_responses(responses=responses)
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(hotkey_to_scores, miner_uids)

        # TODO delay sending result to allow adequate time for scoring
        bt.logging.info(f"Forwarding consensus back to miners... {ranking_result}")
        await self._forward_consensus(synapse=ranking_result)


async def main():
    await Validator().run()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
