import asyncio
import copy
import functools
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, Tuple

import bittensor as bt

from commons.objects import ObjectManager
from commons.human_feedback.aws_mturk import MTurkUtils, STSUtils
from commons.human_feedback.dojo import DojoAPI
from commons.reward_model.models import RewardModel
from commons.utils import get_epoch_time
from template import VALIDATOR_MIN_STAKE
from template.base.miner import BaseMinerNeuron
from template.protocol import (
    CriteriaType,
    FeedbackRequest,
    ScoringResult,
    ScoringMethod,
)
from template.utils.config import get_config
from template.utils.uids import is_miner


class Miner(BaseMinerNeuron):
    _should_exit = False

    def __init__(self):
        super(Miner, self).__init__()
        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_feedback_request,
            blacklist_fn=self.blacklist_feedback_request,
            priority_fn=self.priority_ranking,
        ).attach(forward_fn=self.forward_result)

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        # log all incoming requests
        self.hotkey_to_request: Dict[str, FeedbackRequest] = {}

    async def forward_result(self, synapse: ScoringResult) -> ScoringResult:
        bt.logging.info("Received scoring result from validators")

        found_miner_score = synapse.hotkey_to_scores.get(
            self.wallet.hotkey.ss58_address, None
        )
        if found_miner_score is None:
            bt.logging.error(
                f"Miner hotkey {self.wallet.hotkey.ss58_address} not found in scoring result but yet was send the result"
            )
            return
        bt.logging.info(f"Miner received score: {found_miner_score}")
        return

    async def forward_feedback_request(
        self, synapse: FeedbackRequest
    ) -> FeedbackRequest:
        bt.logging.info(f"Miner received request id: {synapse.request_id}")
        try:
            self.hotkey_to_request[synapse.dendrite.hotkey] = synapse

            scoring_method = self.config.scoring_method
            if scoring_method.casefold() == ScoringMethod.DOJO:
                synapse.scoring_method = ScoringMethod.DOJO
                task_ids = await DojoAPI.create_task(synapse)
                assert len(task_ids) == 1
                synapse.dojo_task_id = task_ids[0]

            elif scoring_method.casefold() == ScoringMethod.HF_MODEL:
                synapse.scoring_method = ScoringMethod.HF_MODEL
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        None,
                        RewardModel.hf_score_text,
                        get_config().model_name,
                        synapse.prompt,
                        completion.text,
                    )
                    for completion in synapse.responses
                ]
                scores = await asyncio.gather(*tasks)
                sorted_model_scores = sorted(
                    (
                        (completion.model, score)
                        for completion, score in zip(synapse.responses, scores)
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for i in range(len(synapse.responses)):
                    key = synapse.responses[i].model
                    index = next(
                        (
                            i
                            for i, pair in enumerate(sorted_model_scores)
                            if pair[0] == key
                        ),
                        None,
                    )
                    if index is None:
                        bt.logging.error(
                            "You fucked up and placed the wrong item in the list"
                        )
                        continue
                    synapse.responses[i].rank_id = index

            elif scoring_method.casefold() == ScoringMethod.LLM_API:
                synapse.scoring_method = ScoringMethod.LLM_API
                scores_response = await RewardModel.llm_api_score_text(
                    provider=get_config().llm_provider,
                    model_name=get_config().model_name,
                    prompt=synapse.prompt,
                    completions=synapse.responses,
                )

                for i in range(len(synapse.responses)):
                    matching_score_item = next(
                        (
                            item
                            for item in scores_response.scores
                            if item.model_id == synapse.responses[i].model
                        ),
                        None,
                    )
                    if not matching_score_item:
                        continue
                    synapse.responses[i].rank_id = matching_score_item.rank_id

                    for criteria in synapse.criteria_types:
                        sorted_model_score_pairs = sorted(
                            [
                                (item.model_id, item.score)
                                for item in scores_response.scores
                            ],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        if criteria == CriteriaType.PREFERENCE_RANKING:
                            sorted_model_rank_pairs = [
                                (model_with_score[0], i)
                                for i, model_with_score in enumerate(
                                    sorted_model_score_pairs, start=1
                                )
                            ]
                            for model_rank_pair in sorted_model_rank_pairs:
                                for completion in synapse.responses:
                                    if completion.model == model_rank_pair[0]:
                                        completion.model_rank_pair_id = model_rank_pair[
                                            1
                                        ]

                        elif criteria == CriteriaType.SCORE:
                            for score_item in scores_response.scores:
                                for i in range(len(synapse.responses)):
                                    if (
                                        score_item.model_id
                                        == synapse.responses[i].model
                                    ):
                                        synapse.responses[i].score = score_item.score

            elif scoring_method.casefold() == ScoringMethod.AWS_MTURK:
                # send off to MTurk workers in a non-blocking way
                loop = asyncio.get_event_loop()
                task = functools.partial(
                    MTurkUtils.create_mturk_task,
                    prompt=synapse.prompt,
                    completions=synapse.responses,
                    reward_in_dollars=0.10,
                )
                synapse.scoring_method = ScoringMethod.AWS_MTURK
                success, hit_id = await loop.run_in_executor(None, task)
                if success:
                    synapse.mturk_hit_id = hit_id

                credentials = STSUtils.assume_role()
                credentials.environment = (
                    ObjectManager.get_config().aws_mturk_environment
                )
                synapse.aws_credentials = credentials
            else:
                bt.logging.error("Unrecognized scoring method!")
        except:
            traceback.print_exc()

        return synapse

    async def blacklist_feedback_request(
        self, synapse: FeedbackRequest
    ) -> Tuple[bool, str]:
        bt.logging.info("checking blacklist function")
        caller_hotkey = synapse.dendrite.hotkey
        if caller_hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.warning(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        bt.logging.warning(f"Got request from {caller_hotkey}")

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
            bt.logging.warning(
                f"Blacklisting hotkey: {caller_hotkey} with insufficient stake, minimum stake required: {VALIDATOR_MIN_STAKE}, current stake: {validator_neuron.stake.tao}"
            )
            return True, "Insufficient validator stake"

        return False, "Valid request received from validator"

    async def priority_ranking(self, synapse: FeedbackRequest) -> float:
        """
        The priority function determines the order in which requests are handled. Higher-priority
        requests are processed before others. Miners may recieve messages from multiple entities at
        once. This function determines which request should be processed first.
        Higher values indicate that the request should be processed first.
        Lower values indicate that the request should be processed later.
        """
        current_timestamp = datetime.fromtimestamp(get_epoch_time())
        dt = current_timestamp - datetime.fromtimestamp(synapse.epoch_timestamp)
        priority = float(dt.total_seconds())
        bt.logging.debug(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority

    def resync_metagraph(self):
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            bt.logging.info("Metagraph unchanged")
            return

        bt.logging.info("Metagraph updated")

    @classmethod
    async def log_miner_status(cls):
        while not cls._should_exit:
            bt.logging.info(f"Miner running... {time.time()}")
            await asyncio.sleep(20)
