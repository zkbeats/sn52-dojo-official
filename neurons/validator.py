import asyncio
from typing import List
from datetime import datetime, timedelta

import bittensor as bt
from commons.data_manager import DataManager
from commons.dataset import SeedDataManager
from commons.objects import DendriteQueryResponse
from commons.consensus import Consensus

from commons.utils import get_epoch_time, get_new_uuid
from template.base.validator import BaseValidatorNeuron
from template.protocol import Completion, RankingRequest, RankingResult
from template.utils.uids import get_random_uids
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

    async def _forward_consensus(self, synapse: RankingResult, hotkeys: List[str]):
        bt.logging.debug("Sending back consensus to miners")
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=5,
        )

    async def _update_score_and_send_feedback(self):
        bt.logging.debug("Scheduled update score and send feedback triggered...")
        data = DataManager.load(path=DataManager.get_ranking_data_filepath())
        if not data:
            bt.logging.debug("Skipping scoring as no data found")
            return

        # get those where request epoch time >X h from current time
        current_time = datetime.fromtimestamp(get_epoch_time())
        print(f"Current time: {current_time}")
        for d in data:
            print(
                f"Request epoch time: {datetime.fromtimestamp(d.request.epoch_timestamp)}, raw: {d.request.epoch_timestamp}"
            )
        data: List[DendriteQueryResponse] = [
            d
            for d in data
            if current_time - datetime.fromtimestamp(d.request.epoch_timestamp)
            >= timedelta(seconds=5)
        ]
        if not data:
            bt.logging.warning("No data to update scores and send feedback")
            return

        for d in data:
            # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
            self.update_scores(d.hotkey_to_scores)
            await self._forward_consensus(
                synapse=d.result, hotkeys=list(d.hotkey_to_scores.keys())
            )
            await asyncio.sleep(5)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        prompt, completions = SeedDataManager.get_prompt_and_completions()
        request = RankingRequest(
            n_completions=len(completions),
            pid=get_new_uuid(),
            prompt=prompt,
            completions=[Completion(text=c) for c in completions],
        )

        miner_uids = get_random_uids(
            metagraph=self.metagraph,
            k=self.config.neuron.sample_size,
            config=self.config,
        )
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        if not len(axons):
            bt.logging.warning("No axons to query ... skipping")
            return

        # The dendrite client queries the network.
        responses: List[RankingRequest] = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            # Construct a dummy query. This simply contains a single integer.
            synapse=request,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=self.config.dendrite_timeout,
        )
        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        valid_responses = [r for r in responses if len(r.ranks) > 0]
        if not len(valid_responses):
            bt.logging.warning("No valid responses received from miners.")
            return

        ranking_result = Consensus.consensus_score(responses=responses)
        DataManager.append_responses(
            response=DendriteQueryResponse(
                request=request,
                result=ranking_result,
            )
        )


async def main():
    scheduler = AsyncIOScheduler(
        job_defaults={"max_instances": 1, "misfire_grace_time": 3}
    )
    validator = Validator()
    scheduler.add_job(validator._update_score_and_send_feedback, "interval", seconds=10)
    scheduler.start()

    await validator.run()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
