import asyncio
import copy
from datetime import datetime
import threading
import time
import functools
import traceback
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import bittensor as bt
from commons.llm.openai_proxy import Provider
from commons.human_feedback.aws_mturk import MTurkUtils
from commons.reward_model.models import ModelUtils
from commons.scoring import Scoring
from commons.utils import get_epoch_time

from template.base.miner import BaseMinerNeuron
from template.protocol import (
    ModelConfig,
    MTurkResponse,
    Rank,
    RankingRequest,
    RankingResult,
    ScoringMethod,
)
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
        self.hotkey_to_request: Dict[str, RankingRequest] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def forward_result(self, synapse: RankingResult) -> RankingResult:
        bt.logging.info("Received consensus from validators")
        return synapse

    async def forward_ranking_request(self, synapse: RankingRequest) -> RankingRequest:
        try:
            print(f"Miner received request, type={str(synapse.__class__.__name__)}")
            self.hotkey_to_request[synapse.dendrite.hotkey] = synapse

            scoring_method = self.config.scoring_method
            if scoring_method.casefold() == ScoringMethod.HF_MODEL:
                for completion in synapse.completions:
                    score = ModelUtils._hf_score(
                        self.config.model_name, synapse.prompt, completion.text
                    )
                    synapse.ranks.append(
                        Rank(
                            cid=completion.cid,
                            score=score,
                        )
                    )
                synapse.scoring_method = ScoringMethod.HF_MODEL
                synapse.model_config = ModelConfig(model_name=self.config.model_name)

            elif scoring_method.casefold() == ScoringMethod.LLM_API:
                llm_provider = Provider(self.config.llm_provider)
                model_name = self.config.model_name
                scores_response = await ModelUtils._llm_api_score(
                    provider=llm_provider,
                    model_name=model_name,
                    prompt=synapse.prompt,
                    completions=synapse.completions,
                )
                for completion in synapse.completions:
                    matching_score_item = next(
                        (
                            item
                            for item in scores_response.scores
                            if item.completion_id == completion.cid
                        ),
                        None,
                    )

                    if matching_score_item:
                        synapse.ranks.append(
                            Rank(
                                cid=completion.cid,
                                score=matching_score_item.score,
                            )
                        )
                synapse.scoring_method = ScoringMethod.LLM_API
                synapse.model_config = ModelConfig(
                    provider=llm_provider, model_name=model_name
                )

            elif scoring_method.casefold() == ScoringMethod.AWS_MTURK:
                # TODO @dev eval task for aws mturk method as well, means we need WorkerIDs, etc.
                # send off to MTurk workers in a non-blocking way
                loop = asyncio.get_event_loop()
                task = functools.partial(
                    MTurkUtils.create_mturk_task,
                    prompt=synapse.prompt,
                    completions=synapse.completions,
                    reward_in_dollars=0.01,
                )
                synapse.scoring_method = ScoringMethod.AWS_MTURK
                await loop.run_in_executor(self.executor, task)
            else:
                bt.logging.error("Unrecognized scoring method!")
        except:
            traceback.print_exc()

        return synapse

    async def send_mturk_response(self, synapse: MTurkResponse):
        """After receiving a response from MTurk, send the response back to the calling validator"""
        # 1. figure out which validator hotkey sent the original request
        hotkey = self._find_hotkey_by_completions(synapse.completion_id_to_score)
        if not hotkey and not self.hotkey_to_request:
            bt.logging.error(
                f"No hotkey found for completion ids: {synapse.completion_id_to_score.keys()}"
            )
            return

        uid = self.metagraph.hotkeys.index(hotkey)
        axon = self.metagraph.axons[uid]
        await self.dendrite.forward(
            axons=[axon],
            synapse=synapse,
            deserialize=False,
            timeout=60,
        )
        return

    def _find_hotkey_by_completions(self, completion_id_to_scores: Dict[str, float]):
        if not self.hotkey_to_request:
            bt.logging.warning(
                "No requests received yet... therefore no validator hotkeys to find"
            )
            return None
        mturk_completion_ids = set(completion_id_to_scores.keys())
        for k, v in self.hotkey_to_request.items():
            completion_ids = set([completion.cid for completion in v.completions])
            if (
                common_cids := completion_ids.intersection(mturk_completion_ids)
            ) and len(common_cids):
                return k
        return None

    async def blacklist_ranking_request(
        self, synapse: RankingRequest
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

        # TODO @dev INCREASE BEFORE LAUNCH
        # TODO @dev INCREASE BEFORE LAUNCH
        # TODO @dev INCREASE BEFORE LAUNCH
        # TODO @dev INCREASE BEFORE LAUNCH
        # TODO @dev INCREASE BEFORE LAUNCH
        # TODO @dev INCREASE BEFORE LAUNCH
        # MIN_VALIDATOR_STAKE = 20_000
        MIN_VALIDATOR_STAKE = 500
        if validator_neuron.stake.tao < float(MIN_VALIDATOR_STAKE):
            bt.logging.warning(
                f"Blacklisting hotkey: {caller_hotkey} with insufficient stake, minimum stake required: {MIN_VALIDATOR_STAKE}, current stake: {validator_neuron.stake.tao}"
            )
            return True, "Insufficient validator stake"

        return False, "Valid request received from validator"

    async def priority_ranking(self, synapse: RankingRequest) -> float:
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
