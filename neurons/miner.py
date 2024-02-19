import asyncio
import threading
import time
from typing import Dict, Tuple

import bittensor as bt
from commons.llm.openai_proxy import Provider
from commons.human_feedback.aws_mturk import MTurkUtils
from commons.reward_model.models import ModelUtils

from template.base.miner import BaseMinerNeuron
from template.protocol import (
    MTurkResponse,
    Rank,
    RankingRequest,
    RankingResult,
    ScoringMethod,
)


class Miner(BaseMinerNeuron):
    """Singleton class for miner."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Miner, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        super(Miner, self).__init__()

        # TODO(developer): Anything specific to your use case you can do here
        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_ranking_request,
            blacklist_fn=self.blacklist_ranking_request,
            priority_fn=self.priority,
        ).attach(forward_fn=self.forward_result)
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.hotkey_to_request: Dict[str, RankingRequest] = {}

    async def forward_result(self, synapse: RankingResult) -> RankingResult:
        bt.logging.info("Received consensus from validators")
        return synapse

    async def forward_ranking_request(self, synapse: RankingRequest) -> RankingRequest:
        print(f"Miner received request, type={str(synapse.__class__.__name__)}")
        self.hotkey_to_request[synapse.dendrite.hotkey] = synapse

        scoring_method = self.config.scoring_method
        if scoring_method.casefold() == ScoringMethod.HF_MODEL:
            for completion in synapse.completions:
                score = ModelUtils._hf_score(
                    self.config.reward_model, synapse.prompt, completion.text
                )
                synapse.ranks.append(
                    Rank(
                        cid=completion.cid,
                        score=score,
                        scoring_method=ScoringMethod.HF_MODEL,
                    )
                )

        elif scoring_method.casefold() == ScoringMethod.LLM_API:
            scores_response = await ModelUtils._llm_api_score(
                provider=Provider.TOGETHER_AI,
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
                    [None],
                )

                if matching_score_item:
                    synapse.ranks.append(
                        Rank(
                            cid=completion.cid,
                            score=matching_score_item.score,
                            scoring_method=ScoringMethod.LLM_API,
                        )
                    )

        elif scoring_method.casefold() == ScoringMethod.AWS_MTURK:
            # send off to MTurk workers... will timeout on validator side
            create_mturk_status = MTurkUtils.create_mturk_task(
                prompt=synapse.prompt,
                completions=synapse.completions,
                reward_in_dollars=0.01,
            )

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
        await self.dendrite(
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
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.warning(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        validator_neuron: bt.NeuronInfo = self.metagraph.neurons[caller_uid]
        if not validator_neuron.validator_permit:
            return True, "Not a validator"

        # TODO re-enable before going live
        # MIN_VALIDATOR_STAKE = 20_000
        MIN_VALIDATOR_STAKE = 0
        if (
            validator_neuron.validator_permit
            and validator_neuron.stake.tao < MIN_VALIDATOR_STAKE
        ):
            bt.logging.warning(
                f"Blacklisting hotkey: {synapse.dendrite.hotkey} with insufficient stake, minimum stake required: {MIN_VALIDATOR_STAKE}, current stake: {validator_neuron.stake.tao}"
            )
            return True, "Insufficient validator stake"

        return False, "Passed blacklist function"

    async def priority(self, synapse: RankingRequest) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


async def log_miner_status():
    while True:
        bt.logging.info(f"Miner running... {time.time()}")
        await asyncio.sleep(5)
