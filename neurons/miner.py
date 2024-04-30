import asyncio
import copy
import functools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Tuple

import bittensor as bt

from commons.factory import Factory
from commons.human_feedback.aws_mturk import MTurkUtils, STSUtils
from commons.human_feedback.dojo import DojoAPI
from commons.llm.openai_proxy import Provider
from commons.reward_model.models import ModelUtils
from commons.utils import get_epoch_time
from template import VALIDATOR_MIN_STAKE
from template.base.miner import BaseMinerNeuron
from template.protocol import (
    ModelConfig,
    FeedbackRequest,
    RankingResult,
    ScoringMethod,
)
from template.utils.config import get_config
from template.utils.uids import is_miner


class Miner(BaseMinerNeuron):
    def __init__(self):
        super(Miner, self).__init__()
        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_ranking_request,
            blacklist_fn=self.blacklist_ranking_request,
            priority_fn=self.priority_ranking,
        ).attach(forward_fn=self.forward_result)

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        # log all incoming requests
        self.hotkey_to_request: Dict[str, FeedbackRequest] = {}
        # self.executor = ThreadPoolExecutor(max_workers=4)

    async def forward_result(self, synapse: RankingResult) -> RankingResult:
        bt.logging.info("Received consensus from validators")
        return synapse

    async def forward_ranking_request(
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
                model_name = get_config().model_name
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        None,
                        ModelUtils.hf_score_text,
                        model_name,
                        synapse.prompt,
                        completion.text,
                    )
                    for completion in synapse.completions
                ]
                scores = await asyncio.gather(*tasks)
                sorted_model_scores = sorted(
                    (
                        (completion.model_id, score)
                        for completion, score in zip(synapse.completions, scores)
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for i in range(len(synapse.completions)):
                    key = synapse.completions[i].model_id
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
                    synapse.completions[i].rank_id = index

            elif scoring_method.casefold() == ScoringMethod.LLM_API:
                synapse.scoring_method = ScoringMethod.LLM_API
                synapse.model_config = ModelConfig(
                    provider=Provider(self.config.llm_provider),
                    model_name=self.config.model_name,
                )
                # TODO directly provide scores down here

            elif scoring_method.casefold() == ScoringMethod.AWS_MTURK:
                # send off to MTurk workers in a non-blocking way
                loop = asyncio.get_event_loop()
                task = functools.partial(
                    MTurkUtils.create_mturk_task,
                    prompt=synapse.prompt,
                    completions=synapse.completions,
                    reward_in_dollars=0.01,
                )
                synapse.scoring_method = ScoringMethod.AWS_MTURK
                success, hit_id = await loop.run_in_executor(None, task)
                if success:
                    synapse.mturk_hit_id = hit_id

                credentials = STSUtils.assume_role()
                credentials.environment = Factory.get_config().aws_mturk_environment
                synapse.aws_credentials = credentials
            else:
                bt.logging.error("Unrecognized scoring method!")
        except:
            traceback.print_exc()

        return synapse

    # async def send_mturk_response(self, synapse: MTurkResponse):
    #     """After receiving a response from MTurk, send the response back to the calling validator"""
    #     # 1. figure out which validator hotkey sent the original request
    #     hotkey = self._find_hotkey_by_completions(synapse.completion_id_to_score)
    #     if not hotkey and not self.hotkey_to_request:
    #         bt.logging.error(
    #             f"No hotkey found for completion ids: {synapse.completion_id_to_score.keys()}"
    #         )
    #         return
    #     # generate temporary credentials to send to the validator
    #     if not synapse.aws_credentials:
    #         credentials = STSUtils.assume_role()
    #         credentials.environment = Factory.get_config().aws_mturk_environment
    #         synapse.aws_credentials = credentials

    #     uid = self.metagraph.hotkeys.index(hotkey)
    #     axon = self.metagraph.axons[uid]
    #     await self.dendrite.forward(
    #         axons=[axon],
    #         synapse=synapse,
    #         deserialize=False,
    #         timeout=60,
    #     )
    #     return

    # def _find_hotkey_by_completions(self, completion_id_to_scores: Dict[str, float]):
    #     if not self.hotkey_to_request:
    #         bt.logging.warning(
    #             "No requests received yet... therefore no validator hotkeys to find"
    #         )
    #         return None
    #     mturk_completion_ids = set(completion_id_to_scores.keys())
    #     for k, v in self.hotkey_to_request.items():
    #         completion_ids = set([completion.cid for completion in v.completions])
    #         if (
    #             common_cids := completion_ids.intersection(mturk_completion_ids)
    #         ) and len(common_cids):
    #             return k
    #     return None

    async def blacklist_ranking_request(
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


async def log_miner_status():
    while True:
        bt.logging.info(f"Miner running... {time.time()}")
        await asyncio.sleep(20)
