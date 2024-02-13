import asyncio
import time
from typing import List, Tuple
from datetime import datetime, timedelta
import copy

import bittensor as bt
from commons.data_manager import DataManager
from commons.dataset import SeedDataManager
from commons.objects import DendriteQueryResponse
from commons.consensus import Consensus

from commons.utils import get_epoch_time, get_new_uuid
from template.base.validator import BaseValidatorNeuron
from template.protocol import (
    Completion,
    MTurkResponse,
    Rank,
    RankingRequest,
    RankingResult,
)
from template.utils.uids import get_random_uids
from apscheduler.schedulers.asyncio import AsyncIOScheduler


def _filter_valid_responses(responses: List[RankingRequest]) -> List[RankingRequest]:
    return [response for response in responses if len(response.ranks) > 0]


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)
        self.axon.attach(
            forward_fn=self.forward_mturk_response,
            blacklist_fn=self.blacklist_mturk_response,
        )
        bt.logging.info(f"Axon created: {self.axon}")

    async def blacklist_mturk_response(
        self, synapse: MTurkResponse
    ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.debug(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        neuron: bt.NeuronInfo = self.metagraph.neurons[caller_uid]
        if neuron.validator_permit:
            bt.logging.trace(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} who is a validator"
            )
            return True, "Validators not allowed"

        return False, "Passed blacklist function"

    async def forward_mturk_response(self, synapse: MTurkResponse):
        """Allows miners to have a delayed response to allow for human feedback loop"""
        # 1. check request from RankingRequest
        # 2. append completion scores to request
        # 3. persist on disk via DataManager

        path = DataManager.get_ranking_data_filepath()
        data = DataManager.load(path=path)
        mturk_cids = set(synapse.completion_id_to_score.keys())

        miner_hotkey = synapse.dendrite.hotkey
        for d in data:
            request_cids = set([completion.cid for completion in d.request.completions])

            if (found_cids := request_cids.intersection(mturk_cids)) and not found_cids:
                bt.logging.warning(
                    "Miner is attempting to send feedback for a wrong completion id..."
                )
                continue

            if (
                miner_participants := [r.axon.hotkey for r in d.responses]
            ) and miner_hotkey in miner_participants:
                bt.logging.warning(
                    f"Miner already sent response for request id: {d.request.request_id}"
                )
                continue

            request_copy = copy.deepcopy(d.request)
            for cid in found_cids:
                # similar to scoring method in Miner.forward(...)
                request_copy.ranks.append(
                    Rank(cid=cid, score=synapse.completion_id_to_score[cid])
                )
                # ensure axon.hotkey has miner's hotkey because our Consensus._spearman_correlation
                # function expects axon.hotkey to be the miner's hotkey
                request_copy.axon.hotkey = miner_hotkey
                d.responses.append(request_copy)

            d.responses = _filter_valid_responses(d.responses)
        DataManager.save(path, data)
        return

    async def send_consensus(self, synapse: RankingResult, hotkeys: List[str]):
        """Send consensus score back to miners who participated in the request."""
        bt.logging.debug(
            f"Sending back consensus to miners for request id: {synapse.request_id}"
        )
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        await self.dendrite(axons=axons, synapse=synapse, deserialize=False, timeout=12)

    async def update_score_and_send_feedback(self):
        bt.logging.debug(
            f"Scheduled update score and send feedback triggered at time: {time.time()}"
        )
        data = DataManager.load(path=DataManager.get_ranking_data_filepath())
        if not data:
            bt.logging.debug(
                "Skipping scoring as no ranking data found, this means either all have been processed or you are running the validator for the first time."
            )
            return

        # get those where request epoch time >X h from current time
        current_time = datetime.fromtimestamp(get_epoch_time())
        print(f"Current time: {current_time}")
        for d in data:
            print(
                f"Request epoch time: {datetime.fromtimestamp(d.request.epoch_timestamp)}, raw: {d.request.epoch_timestamp}"
            )

        # TODO @dev change this to ensure enough time for human feedback!!!
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
            ranking_result = Consensus.consensus_score(responses=d.responses)
            # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
            self.update_scores(ranking_result.hotkey_to_score)
            await self.send_consensus(
                synapse=ranking_result,
                hotkeys=list(ranking_result.hotkey_to_score.keys()),
            )
            await asyncio.sleep(5)

    async def forward(self, synapse: RankingRequest = None) -> DendriteQueryResponse:
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        if synapse is None:
            prompt, completions = SeedDataManager.get_prompt_and_completions()
            synapse = RankingRequest(
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
        axons = [
            self.metagraph.axons[uid]
            for uid in miner_uids
            if self.metagraph.axons[uid].hotkey.casefold()
            != self.wallet.hotkey.ss58_address.casefold()
        ]
        if not len(axons):
            bt.logging.warning("No axons to query ... skipping")
            return

        # The dendrite client queries the network.
        responses: List[RankingRequest] = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            # Construct a dummy query. This simply contains a single integer.
            synapse=synapse,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=self.config.dendrite_timeout,
        )
        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        valid_responses = _filter_valid_responses(responses)
        if not len(valid_responses):
            bt.logging.warning("No valid responses received from miners.")
            return

        response_data = DendriteQueryResponse(
            request=synapse,
            responses=valid_responses,
        )
        DataManager.append_responses(response=response_data)
        return response_data


validator = Validator()


async def main():
    scheduler = AsyncIOScheduler(
        job_defaults={"max_instances": 1, "misfire_grace_time": 3}
    )
    scheduler.add_job(validator.update_score_and_send_feedback, "interval", seconds=10)
    scheduler.start()

    await validator.run()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
