import asyncio
import threading
import time
from typing import Dict, Tuple

import bittensor as bt
from commons.llm.openai_proxy import Provider
from commons.human_feedback.aws_mturk import MTurkUtils
from commons.reward_model.models import ModelUtils

from template.base.miner import BaseMinerNeuron
from template.protocol import MTurkResponse, Rank, RankingRequest, RankingResult

from template.utils.config import ScoringMethod


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Miner, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

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
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        ).attach(forward_fn=self.forward_result)
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.hotkey_to_request: Dict[str, RankingRequest] = {}

    async def forward_result(self, synapse: RankingResult) -> None:
        bt.logging.info("Received consensus from validators")
        return synapse

    async def forward(self, synapse: RankingRequest) -> RankingRequest:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        print(f"Miner received request, type={str(synapse.__class__.__name__)}")
        self.hotkey_to_request[synapse.dendrite.hotkey] = synapse

        # TODO make this a param in the miner config
        scoring_method = self.config.scoring_method
        if scoring_method.casefold() == ScoringMethod.HF_MODEL:
            for completion in synapse.completions:
                # TODO change method of scoring here, see models.ModelUtils for different scoring methods
                score = ModelUtils._hf_score(
                    self.config.reward_model, synapse.prompt, completion.text
                )
                synapse.ranks.append(Rank(cid=completion.cid, score=score))

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
                        Rank(cid=completion.cid, score=matching_score_item.score)
                    )

        elif scoring_method.casefold() == ScoringMethod.HUMAN_FEEDBACK:
            create_mturk_status = MTurkUtils.create_mturk_task(
                prompt=synapse.prompt,
                completions=synapse.completions,
                reward_in_dollars=0.01,
            )
            # TODO send off to MTurk workers... therefore will timeout... need to handle gracefully

        return synapse

    async def send_mturk_response(self, synapse: MTurkResponse):
        """After receiving a response from MTurk, send the response back to the calling validator"""
        # 1. figure out which validator hotkey sent the original request
        hotkey = Miner._find_hotkey_by_completions(synapse.completion_id_to_score)
        if not hotkey:
            bt.logging.error(
                f"No hotkey found for completion ids: {synapse.completion_id_to_score.keys()}"
            )
            return

        # 2. call method using dendrite.__acall__ to query validator axon
        bt.logging.info(f"Dendrite: {self.dendrite}")

        uid = self.metagraph.hotkeys.index(hotkey)
        axon = self.metagraph.axons[uid]
        await self.dendrite(
            axons=[axon],
            synapse=synapse,
            deserialize=False,
            timeout=60,
        )
        return

    @staticmethod
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

    async def blacklist(self, synapse: RankingRequest) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        bt.logging.info("checking blacklist function")
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # check validator stake if it meets minimum threshold
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        validator_neuron: bt.NeuronInfo = self.metagraph.neurons[caller_uid]
        # MIN_VALIDATOR_STAKE = 20_000
        MIN_VALIDATOR_STAKE = 0
        if (
            validator_neuron.validator_permit
            and validator_neuron.stake.tao < MIN_VALIDATOR_STAKE
        ):
            bt.logging.trace(
                f"Blacklisting hotkey: {synapse.dendrite.hotkey} with insufficient stake, minimum stake required: {MIN_VALIDATOR_STAKE}, current stake: {validator_neuron.stake.tao}"
            )
            return True, "Insufficient validator stake"

        if not validator_neuron.validator_permit:
            return True, "Not a validator"

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
