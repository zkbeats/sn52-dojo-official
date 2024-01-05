import asyncio
import time

import bittensor as bt

import template
from commons.utils import get_new_uuid
from template.base.validator import BaseValidatorNeuron
from template.protocol import Completion, Protocol
from template.utils.uids import get_random_uids
from template.validator.reward import get_rewards


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # TODO(developer): Anything specific to your use case you can do here

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        return await self._forward()

    async def _forward(self):
        """
        The forward function is called by the validator every time step.
        It is responsible for querying the network and scoring the responses.
        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        """
        # TODO(developer): Define how the validator selects a miner to query, how often, etc.
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        # construct our synapse
        # TODO change to real data
        synapse = Protocol(
            n_completions=2,
            pid=str(get_new_uuid()),
            prompt="What is your name?",
            completions=[
                Completion(
                    text="My name is Assistant, and I am a helpful assisstant created by OpenAI."
                ),
                Completion(
                    text="My name is Llama, and I am an assistant created by Meta."
                ),
            ],
        )

        # The dendrite client queries the network.
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            # Construct a dummy query. This simply contains a single integer.
            synapse=synapse,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=30,
        )
        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        # TODO(developer): Define how the validator scores responses.
        # Adjust the scores based on responses from miners.
        rewards = get_rewards(self, query=self.step, responses=responses)

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, miner_uids)


# TODO hook up to fastapi server
async def main():
    validator = Validator()
    await validator.run()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())

    # with Validator() as validator:
    #     while True:
    #         bt.logging.info("Validator running...", time.time())
    #         time.sleep(5)
