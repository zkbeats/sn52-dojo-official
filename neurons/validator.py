import asyncio
from collections import defaultdict
import threading
import time
from traceback import print_exception
from typing import List, Tuple
from datetime import datetime, timedelta
import copy
from torch.nn import functional as F

import bittensor as bt
import numpy as np
import torch
from commons.data_manager import DataManager
from commons.dataset.dataset import SeedDataManager
from commons.dataset.hf_utils import HuggingFaceUtils
from commons.evals import EvalUtils, log_retry_info
from commons.factory import Factory
from commons.objects import DendriteQueryResponse
from commons.scoring import Scoring

from commons.utils import get_epoch_time, get_new_uuid, serve_axon
from template.base.neuron import BaseNeuron
from template.protocol import (
    Completion,
    MTurkResponse,
    Rank,
    RankingRequest,
    RankingResult,
    ScoringMethod,
    SCORING_METHOD_PRIORITY,
)
from template.utils.uids import get_random_miner_uids, is_miner
from tenacity import Retrying, RetryError, stop_after_attempt


def _filter_valid_responses(responses: List[RankingRequest]) -> List[RankingRequest]:
    return [response for response in responses if len(response.ranks) > 0]


class Validator(BaseNeuron):
    def __init__(self):
        super(Validator, self).__init__()

        self.axon.attach(
            forward_fn=self.forward_mturk_response,
            blacklist_fn=self.blacklist_mturk_response,
        )

        ##### FROM BaseValidatorNeuron ########################################
        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.moving_averaged_scores = None
        self.hotkey_to_accuracy = defaultdict(float)
        # self.lock = asyncio.Lock()

    async def blacklist_mturk_response(
        self, synapse: MTurkResponse
    ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.warning(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not is_miner(self.metagraph, caller_uid):
            bt.logging.warning(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} who is a validator"
            )
            return True, "Validators not allowed"

        return False, "Valid request received from miner"

    async def forward_mturk_response(self, synapse: MTurkResponse):
        """Receives MTurk responses from miners after delayed response to allow for human feedback loop"""
        # 1. check request from RankingRequest
        # 2. append completion scores to request
        # 3. persist on disk via DataManager

        path = DataManager.get_ranking_data_filepath()
        data = await DataManager.load(path=path)
        miner_hotkey = synapse.dendrite.hotkey
        if not data:
            bt.logging.error(
                f"Received MTurk response from miner {miner_hotkey} but no requests persisted on disk."
            )
            return

        mturk_cids = set(synapse.completion_id_to_score.keys())

        for d in data:
            request_cids = set([completion.cid for completion in d.request.completions])

            if (found_cids := request_cids.intersection(mturk_cids)) and not found_cids:
                bt.logging.warning(
                    f"Received mturk response with {mturk_cids=}, but no found requests with those matching CIDs."
                )
                continue

            bt.logging.info(
                f"Miner {miner_hotkey} sent human feedback for {d.request.request_id}"
            )

            existing_responses = {r.axon.hotkey: r for r in d.responses}
            if miner_hotkey in existing_responses:
                existing_method = ScoringMethod(
                    existing_responses[miner_hotkey].scoring_method
                )
                new_method = ScoringMethod.AWS_MTURK
                if (
                    SCORING_METHOD_PRIORITY[new_method]
                    > SCORING_METHOD_PRIORITY[existing_method]
                ):
                    bt.logging.info(
                        f"Replacing {existing_method} with higher priority {new_method} for request id: {d.request.request_id}"
                    )
                    d.responses.remove(existing_responses[miner_hotkey])
                else:
                    bt.logging.warning(
                        f"Miner {miner_hotkey} already sent response with equal or higher priority for request id: {d.request.request_id}, skipping..."
                    )
                    continue

            request_copy = copy.deepcopy(d.request)
            for cid in found_cids:
                # similar to scoring method in Miner.forward(...)
                request_copy.ranks.append(
                    Rank(
                        cid=cid,
                        score=synapse.completion_id_to_score[cid],
                        scoring_method=ScoringMethod.AWS_MTURK,
                    )
                )
                # ensure axon.hotkey has miner's hotkey because our Consensus._spearman_correlation
                # function expects axon.hotkey to be the miner's hotkey
                request_copy.axon.hotkey = miner_hotkey
                d.responses.append(request_copy)

            d.responses = _filter_valid_responses(d.responses)
        await DataManager.save(path, data)
        return

    async def send_consensus(self, synapse: RankingResult, hotkeys: List[str]):
        """Send consensus score back to miners who participated in the request."""
        bt.logging.debug(
            f"Sending back consensus to miners for request id: {synapse.request_id}"
        )
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=12
        )

    async def calculate_miner_classification_accuracy(self):
        data = await DataManager.load(path=DataManager.get_ranking_data_filepath())
        if not data:
            bt.logging.debug(
                "Skipping classification accuracy as no ranking data found."
            )
            return

        for d in data:
            for r in d.responses:
                participant = r.axon.hotkey
                if participant in self.hotkey_to_accuracy:
                    bt.logging.trace(
                        f"Participant {participant} already has an accuracy score... skipping"
                    )
                    continue

                if r.scoring_method not in [method for method in ScoringMethod]:
                    bt.logging.error(
                        f"Unrecognized scoring method: {r.scoring_method} for participant {participant}"
                    )
                    continue

                accuracy = await EvalUtils.classification_accuracy(
                    scoring_method=r.scoring_method, model_config=r.model_config
                )
                self.hotkey_to_accuracy[participant] = accuracy
        return

    async def reset_accuracy(self):
        if not self.hotkey_to_accuracy:
            bt.logging.warning(
                "Reset miner hotkey accuracy triggered, but no accuracy data found. Skipping..."
            )
            self.hotkey_to_accuracy.clear()
        return

    async def update_score_and_send_feedback(self):
        """While this function is triggered every X time period in AsyncIOScheduler,
        only relevant data that has passed the deadline of 8 hours will be scored and sent feedback.
        """
        if not self.axon.info().is_serving:
            bt.logging.warning("Axon is not serving yet...")
            return

        bt.logging.debug(
            f"Scheduled update score and send feedback triggered at time: {time.time()}"
        )
        data = await DataManager.load(path=DataManager.get_ranking_data_filepath())
        if not data:
            bt.logging.debug(
                "Skipping scoring as no ranking data found, this means either all have been processed or you are running the validator for the first time."
            )
            return

        current_time = datetime.fromtimestamp(get_epoch_time())
        filtered_data = [
            d
            for d in data
            if current_time - datetime.fromtimestamp(d.request.epoch_timestamp)
            >= timedelta(hours=8)
        ]
        if not filtered_data:
            bt.logging.debug(
                "Skipping scoring as no ranking data has been persisted for at least 8 hours."
            )
            return

        consumed_responses = []
        for d in filtered_data:
            ranking_result = Scoring.consensus_score(
                responses=d.responses, hotkey_to_multiplier=self.hotkey_to_accuracy
            )
            # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
            self.update_scores(ranking_result.hotkey_to_score)
            await self.send_consensus(
                synapse=ranking_result,
                hotkeys=list(ranking_result.hotkey_to_score.keys()),
            )
            await asyncio.sleep(1)
            consumed_responses.append(d)

            is_contrib_disabled = Factory.get_config().hf_dataset_contrib.off

            if not is_contrib_disabled:
                prompt = d.request.prompt
                completions = d.request.completions
                consensus_scores = [None] * len(completions)
                for i in range(len(completions)):
                    consensus_score = ranking_result.cid_to_consensus.get(
                        completions[i].cid, None
                    )
                    if not consensus_score:
                        continue
                    consensus_scores[i] = consensus_score

                HuggingFaceUtils.append_data(
                    prompt, completions, consensus_scores, self.wallet
                )

        # once we have scored certain responses, just remove them
        await DataManager.remove_responses(consumed_responses)

    async def send_request(
        self, synapse: RankingRequest = None
    ) -> DendriteQueryResponse:
        # typically the request may come from an external source however,
        # initially will seed it with some data for miners to get started
        if synapse is None:
            prompt, completions = SeedDataManager.get_prompt_and_completions()
            synapse = RankingRequest(
                n_completions=len(completions),
                pid=get_new_uuid(),
                prompt=prompt,
                completions=[Completion(text=c) for c in completions],
            )

        miner_uids = get_random_miner_uids(
            metagraph=self.metagraph, k=self.config.neuron.sample_size
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
        responses: List[RankingRequest] = await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=30
        )

        valid_responses = [
            response
            for response in responses
            if len(response.ranks) > 0
            and response.scoring_method in [method for method in ScoringMethod]
            and response.scoring_method != ScoringMethod.AWS_MTURK
        ]

        if not len(valid_responses):
            bt.logging.error("No valid responses received from miners.")
            return
        else:
            bt.logging.success(f"Received {len(valid_responses)} valid responses")

        response_data = DendriteQueryResponse(
            request=synapse,
            responses=valid_responses,
        )
        await DataManager.append_responses(response=response_data)
        return response_data

    # async def concurrent_forward(self):
    #     coroutines = [
    #         self.forward_request()
    #         for _ in range(self.config.neuron.num_concurrent_forwards)
    #     ]
    #     await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Synchronizes with the chain and updates the metagraph.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # manually always register and always sync metagraph when application starts
        self.resync_metagraph()
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        # TODO add retry mechanism for axon serve rate limit

        serve_success = serve_axon(self.subtensor, self.axon, self.config)
        if serve_success:
            bt.logging.success("Successfully served axon for validator!")
        else:
            bt.logging.error("Failed to serve axon for validator, exiting.")
            exit()

        self.axon.start()
        bt.logging.info(
            f"Serving validator axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                self.loop.run_until_complete(self.send_request())

                # # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1
                time.sleep(24)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    # def stop_run_thread(self):
    #     """
    #     Stops the validator's operations that are running in the background thread.
    #     """
    #     if self.is_running:
    #         bt.logging.debug("Stopping validator in background thread.")
    #         self.should_exit = True
    #         self.thread.join(5)
    #         self.is_running = False
    #         bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners.
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
        if self.moving_averaged_scores is None:
            self.moving_averaged_scores = self.scores.clone().detach()

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = F.normalize(self.moving_averaged_scores, p=1, dim=0)
        if torch.all(raw_weights == 0):
            bt.logging.warning(
                "All weights are zero, therefore no valid weights to set"
            )
            return

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", self.metagraph.uids.to("cpu"))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # # TODO remove call because this is already inside of nested call inside
        # of set_weights_extrinsic() function

        # # Convert to uint16 weights and uids.
        # (
        #     uint_uids,
        #     uint_weights,
        # ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
        #     uids=processed_weight_uids, weights=processed_weights
        # )
        # bt.logging.debug("uint_weights", uint_weights)
        # bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids.tolist(),
            weights=processed_weights.tolist(),
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.success("Validator set weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed")
        return result

    def resync_metagraph(self):
        """Sync the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("checking if we need to resync metagraph...")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
        # create a copy to freeze it in time

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            bt.logging.info("Metagraph unchanged")
            return

        bt.logging.info("Metagraph updated")
        # hotkey has been replaced, so we reset score
        preserved_uids = [
            uid
            for uid, hotkey in enumerate(previous_metagraph.hotkeys)
            if hotkey == self.metagraph.hotkeys[uid]
        ]

        # create a new score tensor since metagraph size is different
        updated_scores = torch.zeros(len(self.metagraph.axons)).to(self.device)
        for uid in preserved_uids:
            updated_scores[uid] = self.scores[uid]

        # Update our states
        self.scores = updated_scores
        self.hotkeys = self.metagraph.hotkeys

    def update_scores(self, hotkey_to_scores):
        """Performs exponential moving average on the scores based on the rewards received from the miners,
        after setting the self.scores variable here, `set_weights` will be called to set the weights on chain."""

        nan_value_indices = np.isnan(list(hotkey_to_scores.values()))
        if nan_value_indices.any():
            bt.logging.warning(f"NaN values detected in rewards: {hotkey_to_scores}")

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # scores dimensions might have been updated after resyncing... len(uids) != len(self.scores)
        rewards = torch.zeros((len(self.hotkeys),))
        for index, (key, value) in enumerate(hotkey_to_scores.items()):
            # handle nan values
            if nan_value_indices[index]:
                rewards[key] = 0.0
            # search metagraph for hotkey and grab uid
            uid = self.hotkeys.index(key)

            # multiply by the classification accuracy
            # TODO @dev handle case
            if (accuracy := self.hotkey_to_accuracy[key]) and accuracy == 0.0:
                bt.logging.warning(f"Classification accuracy for hotkey {key} is 0")

            rewards[uid] = value * accuracy
        bt.logging.debug(f"Rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * rewards + (1 - alpha) * self.scores.to(
            self.device
        )
        bt.logging.debug(f"Updated scores: {self.scores}")


async def log_validator_status():
    while True:
        bt.logging.info(f"Validator running... {time.time()}")
        await asyncio.sleep(20)
