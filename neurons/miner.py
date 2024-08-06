import asyncio
import copy
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, Tuple

import bittensor as bt
from loguru import logger

from commons.human_feedback.dojo import DojoAPI
from commons.utils import get_epoch_time
from template import VALIDATOR_MIN_STAKE
from template.base.miner import BaseMinerNeuron
from template.protocol import FeedbackRequest, Heartbeat, ScoringMethod, ScoringResult
from template.utils.uids import is_miner


class Miner(BaseMinerNeuron):
    _should_exit = False

    def __init__(self):
        super().__init__()
        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)

        # Attach determiners which functions are called when servicing a request.
        logger.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_feedback_request,
            blacklist_fn=self.blacklist_feedback_request,
            priority_fn=self.priority_ranking,
        ).attach(forward_fn=self.forward_result).attach(forward_fn=self.ack_heartbeat)

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        # log all incoming requests
        self.hotkey_to_request: Dict[str, FeedbackRequest] = {}

    async def ack_heartbeat(self, synapse: Heartbeat) -> Heartbeat:
        logger.debug("Received heartbeat synapse")
        if not synapse:
            logger.error("Invalid synapse object")
            return synapse

        logger.debug("Responding to heartbeat synapse")
        synapse.ack = True
        return synapse

    async def forward_result(self, synapse: ScoringResult) -> ScoringResult:
        logger.info("Received scoring result from validators")
        try:
            # Validate that synapse is not None and has the required fields
            if not synapse or not synapse.hotkey_to_scores:
                logger.error(
                    "Invalid synapse object or missing hotkey_to_scores attribute."
                )
                return synapse

            found_miner_score = synapse.hotkey_to_scores.get(
                self.wallet.hotkey.ss58_address, None
            )
            if found_miner_score is None:
                logger.error(
                    f"Miner hotkey {self.wallet.hotkey.ss58_address} not found in scoring result but yet was sent the result"
                )
                return synapse

            logger.info(f"Miner received score: {found_miner_score}")
        except KeyError as e:
            logger.error(f"KeyError in forward_result: {e}")
        except Exception as e:
            logger.error(f"Error in forward_result: {e}")

        return synapse

    async def forward_feedback_request(
        self, synapse: FeedbackRequest
    ) -> FeedbackRequest:
        try:
            # Validate that synapse, dendrite, dendrite.hotkey, and response are not None
            if not synapse or not synapse.dendrite or not synapse.dendrite.hotkey:
                logger.error("Invalid synapse: dendrite or dendrite.hotkey is None.")
                return synapse

            logger.info(f"Miner received request id: {synapse.request_id}")

            if not synapse.responses:
                logger.error("Invalid synapse: response field is None.")
                return synapse

            self.hotkey_to_request[synapse.dendrite.hotkey] = synapse

            scoring_method = self.config.scoring_method
            if scoring_method.casefold() == ScoringMethod.DOJO:
                synapse.scoring_method = ScoringMethod.DOJO
                task_ids = await DojoAPI.create_task(synapse)
                assert len(task_ids) == 1
                synapse.dojo_task_id = task_ids[0]

            else:
                logger.error("Unrecognized scoring method!")
        except Exception:
            logger.error(
                f"Error occurred while processing request id: {synapse.request_id}, error: {traceback.format_exc()}"
            )

        return synapse

    async def blacklist_feedback_request(
        self, synapse: FeedbackRequest
    ) -> Tuple[bool, str]:
        logger.info("checking blacklist function")
        caller_hotkey = synapse.dendrite.hotkey
        if caller_hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            logger.warning(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        logger.debug(f"Got request from {caller_hotkey}")

        # TODO @dev remember to remove these when going live
        if caller_hotkey.lower() in [
            "5CAmJ1Pt6HAG21Q3cJaYS3nS7yCRACDSNaxHcGj2fHmtqRDH".lower(),
            "5D2gFiQzwhCDN5sa8RQGuTTLay7HpRL2y4xh637rNQret8Ky".lower(),
        ]:
            return False, "Request received from validator on testnet"

        caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        validator_neuron: bt.NeuronInfo = self.metagraph.neurons[caller_uid]
        if is_miner(self.metagraph, caller_uid):
            return True, "Not a validator"

        if validator_neuron.stake.tao < float(VALIDATOR_MIN_STAKE):
            logger.warning(
                f"Blacklisting hotkey: {caller_hotkey} with insufficient stake, minimum stake required: {VALIDATOR_MIN_STAKE}, current stake: {validator_neuron.stake.tao}"
            )
            return True, "Insufficient validator stake"

        return False, "Valid request received from validator"

    async def priority_ranking(self, synapse: FeedbackRequest) -> float:
        """
        The priority function determines the order in which requests are handled. Higher-priority
        requests are processed before others. Miners may receive messages from multiple entities at
        once. This function determines which request should be processed first.
        Higher values indicate that the request should be processed first.
        Lower values indicate that the request should be processed later.
        """
        current_timestamp = datetime.fromtimestamp(get_epoch_time())
        dt = current_timestamp - datetime.fromtimestamp(synapse.epoch_timestamp)
        priority = float(dt.total_seconds())
        logger.debug(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    def resync_metagraph(self):
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        logger.info("Metagraph updated")

    @classmethod
    async def log_miner_status(cls):
        while not cls._should_exit:
            logger.info(f"Miner running... {time.time()}")
            await asyncio.sleep(20)
