import asyncio
import copy
import threading
import time
from collections import defaultdict
from traceback import print_exception
from typing import Dict, List, Tuple

import bittensor as bt
import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from torch.nn import functional as F

from commons.data_manager import DataManager
from commons.dataset.dataset import SeedDataManager
from commons.evals import EvalUtils
from commons.human_feedback.aws_mturk import MTurkUtils, parse_assignment
from commons.logging.wandb_logging import wandb_log
from commons.reward_model.models import ModelUtils
from commons.scoring import Scoring
from commons.utils import get_epoch_time, get_new_uuid, init_wandb
from template.base.neuron import BaseNeuron
from template.protocol import (
    SCORING_METHOD_PRIORITY,
    AWSCredentials,
    Completion,
    DendriteQueryResponse,
    Modality,
    MTurkResponse,
    Rank,
    RankingRequest,
    RankingResult,
    ScoringMethod,
)
from template.utils.uids import get_random_miner_uids, is_miner


def _filter_valid_responses(responses: List[RankingRequest]) -> List[RankingRequest]:
    return [response for response in responses if len(response.ranks) > 0]


class Validator(BaseNeuron):
    def __init__(self):
        super(Validator, self).__init__()
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.hotkey_to_accuracy = defaultdict(float)

        self.axon.attach(
            forward_fn=self.forward_mturk_response,
            blacklist_fn=self.blacklist_mturk_response,
        )

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)
        self.load_state()
        bt.logging.debug(f"Scores state: {self.scores}")

        # Init sync with the network. Updates the metagraph.
        self.sync()
        init_wandb(config=self.config, my_uid=self.uid, wallet=self.wallet)

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

    @staticmethod
    def verify_mturk_task(
        aws_credentials: AWSCredentials,
        hit_id: str,
        completion_id_to_score: Dict[str, float],
    ):
        temp_client = MTurkUtils.get_client(
            access_key_id=aws_credentials.access_key_id,
            secret_access_key=aws_credentials.secret_access_key,
            session_token=aws_credentials.session_token,
            environment=aws_credentials.environment,
        )
        res = temp_client.list_assignments_for_hit(HITId=hit_id)
        answers = [parse_assignment(assignment) for assignment in res["Assignments"]]
        if not answers:
            return False, "No assignments found for HIT"

        cids_to_check = set(completion_id_to_score.keys())
        # example of answers variable
        # [{'Answer': [{'taskAnswers': [{'a6f41ad1-d8a8-4bf3-a698-1b431bf2edac': 5.68, 'f95cae4d-38ed-4911-b97a-f92a0c3bad9a': 7.49}]}], 'HITId': '3MG8450X3U3J7MRIYLXCI5SO1CIUPJ'}]
        for answer in answers:
            task_answers = answer.get("Answer", [])
            for task_answer in task_answers:
                for task_key, score in task_answer.get("taskAnswers", {}).items():
                    completion_id = MTurkUtils.decode_task_key(task_key)
                    if completion_id not in cids_to_check:
                        bt.logging.warning(
                            f"Completion ID {completion_id} found in MTurk task answers is not in the expected set."
                        )
                        return (
                            False,
                            f"Unexpected completion ID {completion_id} in MTurk task answers.",
                        )
                    elif completion_id_to_score[completion_id] != score:
                        bt.logging.warning(
                            f"Score mismatch for completion ID {completion_id}: expected {completion_id_to_score[completion_id]}, got {score} from MTurk task answers."
                        )
                        return (
                            False,
                            f"Score mismatch for completion ID {completion_id}.",
                        )

        return True, "All checks passed"

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

            is_verified, reason = self.verify_mturk_task(
                synapse.aws_credentials,
                synapse.mturk_hit_id,
                synapse.completion_id_to_score,
            )
            if not is_verified:
                bt.logging.error(
                    f"MTurk task verification failed due to reason:{reason}"
                )
                return

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
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        if not axons:
            bt.logging.warning("No axons to send consensus to... skipping")
        else:
            bt.logging.debug(
                f"Sending back consensus to miners for request id: {synapse.request_id}"
            )

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

        bt.logging.info(
            f"Looping through {len(data)} requests to calculate miner classification accuracy"
        )
        for d in data:
            for r in d.responses:
                participant = r.axon.hotkey
                if (
                    participant in self.hotkey_to_accuracy
                    and self.hotkey_to_accuracy[participant] > 0
                ):
                    bt.logging.debug(
                        f"Participant {participant} already has an accuracy score of {self.hotkey_to_accuracy[participant]} skipping"
                    )
                    continue

                if r.scoring_method not in [method for method in ScoringMethod]:
                    bt.logging.error(
                        f"Unrecognized scoring method: {r.scoring_method} for participant {participant}"
                    )
                    continue

                # TODO @dev ensure that different miners get the same data
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

    async def process_request(self, synapse: RankingRequest) -> RankingRequest:
        """Process request from miners, specifically filling out the ranks fields"""
        bt.logging.debug(
            f"Processing miner's request for scoring method: {synapse.scoring_method}"
        )

        if synapse.scoring_method == ScoringMethod.HF_MODEL:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    ModelUtils.hf_score_text,
                    synapse.model_config.model_name,
                    synapse.prompt,
                    completion.text,
                )
                for completion in synapse.completions
            ]
            scores = await asyncio.gather(*tasks)
            for completion, score in zip(synapse.completions, scores):
                synapse.ranks.append(
                    Rank(
                        cid=completion.cid,
                        score=score,
                    )
                )

        elif synapse.scoring_method == ScoringMethod.LLM_API:
            llm_provider = synapse.model_config.provider
            model_name = synapse.model_config.model_name
            scores_response = await ModelUtils.llm_api_score_text(
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
                if not matching_score_item:
                    continue

                synapse.ranks.append(
                    Rank(
                        cid=completion.cid,
                        score=matching_score_item.score,
                    )
                )
        else:
            bt.logging.error("Unrecognized scoring method!")

        return synapse

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

        current_time = get_epoch_time()
        # allow enough time for human feedback
        SECONDS_IN_4H = 4 * 3600
        filtered_data = [
            d
            for d in data
            if (current_time - d.request.epoch_timestamp) >= SECONDS_IN_4H
        ]
        bt.logging.info(
            f"Got {len(filtered_data)} requests past deadline and ready to score"
        )
        if not filtered_data:
            bt.logging.warning(
                "Skipping scoring as no ranking data has been persisted for at least 8 hours."
            )
            return

        consumed_responses = []
        for d in filtered_data:
            for i in range(len(d.responses)):
                d.responses[i] = await self.process_request(d.responses[i])

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

            completions = d.request.completions
            consensus_scores = [None] * len(completions)
            best_completion = None
            max_consensus_score = float("-inf")
            for i in range(len(completions)):
                cid = completions[i].cid
                consensus_score = ranking_result.cid_to_consensus.get(cid, None)
                if not consensus_score:
                    bt.logging.warning(
                        f"Missing consensus score for request id ({d.request.request_id}) completion id ({cid})"
                    )
                    continue
                if consensus_score > max_consensus_score:
                    max_consensus_score = consensus_score
                    best_completion = completions[i].text

                consensus_scores[i] = consensus_score
            wandb_data = jsonable_encoder(
                {
                    "modality": "text",
                    "prompt": d.request.prompt,
                    "completions": d.request.completions,
                    "num_completions": len(d.request.completions),
                    "consensus_scores": consensus_scores,
                    "best_completion": best_completion,
                    "responses": d.responses,
                    "num_responses": len(d.responses),
                    "hotkey_to_accuracy": self.hotkey_to_accuracy,
                    "hotkey_to_score": ranking_result.hotkey_to_score,
                }
            )
            asyncio.create_task(wandb_log(wandb_data))

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
                modality=Modality.TEXT,
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
            if response.scoring_method in [method for method in ScoringMethod]
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

    async def run(self):
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
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                await self.send_request()

                # # Check if we should exit.
                if self.should_exit:
                    bt.logging.info("Validator should stop...")
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1
                await asyncio.sleep(24)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

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

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        normalized_weights = F.normalize(self.scores.cpu(), p=1, dim=0)

        bt.logging.debug(f"Raw scores: {self.scores}")
        bt.logging.debug(f"normalized weights: {normalized_weights}")
        bt.logging.debug(f"normalized weights uids: {self.metagraph.uids}")

        if torch.count_nonzero(normalized_weights).item() == 0:
            bt.logging.error("All weights are zero, therefore no valid weights to set")
            return

        bt.logging.success("Some weights are non zero.... time to set them!")

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=normalized_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug(f"processed weights {processed_weights}")
        bt.logging.debug(f"processed weights uids {processed_weight_uids}")

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

    def update_scores(self, hotkey_to_scores):
        """Performs exponential moving average on the scores based on the rewards received from the miners,
        after setting the self.scores variable here, `set_weights` will be called to set the weights on chain."""

        nan_value_indices = np.isnan(list(hotkey_to_scores.values()))
        if nan_value_indices.any():
            bt.logging.warning(f"NaN values detected in rewards: {hotkey_to_scores}")

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # scores dimensions might have been updated after resyncing... len(uids) != len(self.scores)
        rewards = torch.zeros((len(self.metagraph.axons),))
        neuron_hotkeys: List[str] = [neuron.hotkey for neuron in self.metagraph.neurons]
        for index, (key, value) in enumerate(hotkey_to_scores.items()):
            # handle nan values
            if nan_value_indices[index]:
                rewards[key] = 0.0
            # search metagraph for hotkey and grab uid
            try:
                uid = neuron_hotkeys.index(key)
            except ValueError:
                bt.logging.warning(
                    "Old hotkey found from previous metagraph, skip setting weights"
                )
                continue

            # multiply by the classification accuracy
            accuracy = self.hotkey_to_accuracy[key]
            if accuracy == 0:
                raise ValueError(
                    f"Accuracy for hotkey {key} is 0, waiting for accuracy to be calculated first."
                )
            bt.logging.debug(f"Accuracy for hotkey {key} is {accuracy}")
            bt.logging.debug(f"Pre-multiplier score for hotkey {key} is {value}")
            rewards[uid] = value * accuracy
        bt.logging.debug(f"Rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * rewards + (1 - alpha) * self.scores.to(
            self.device
        )
        bt.logging.debug(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            DataManager.validator_save(self.scores, self.hotkey_to_accuracy)
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        loop = asyncio.get_event_loop()
        success, scores, hotkey_to_accuracy = loop.run_until_complete(
            DataManager.validator_load()
        )
        if success:
            self.scores = scores
            self.hotkey_to_accuracy = hotkey_to_accuracy


async def log_validator_status():
    while True:
        bt.logging.info(f"Validator running... {time.time()}")
        await asyncio.sleep(20)
