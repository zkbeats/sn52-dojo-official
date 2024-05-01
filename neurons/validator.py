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
from commons.dataset.synthetic import build_prompt_responses_pair
from commons.human_feedback.aws_mturk import MTurkUtils, parse_assignment
from commons.human_feedback.dojo import DojoAPI
from commons.logging.wandb_logging import wandb_log
from commons.scoring import Scoring
from commons.utils import get_epoch_time, init_wandb
from template.base.neuron import BaseNeuron
from template.protocol import (
    SCORING_METHOD_PRIORITY,
    AWSCredentials,
    Completion,
    CriteriaType,
    DendriteQueryResponse,
    MTurkResponse,
    # Rank,
    FeedbackRequest,
    ScoringResult,
    ScoringMethod,
    TaskType,
)
from template.utils.config import get_config
from template.utils.uids import (
    MinerUidSelector,
    extract_miner_uids,
    is_miner,
)


def _filter_valid_responses(responses: List[FeedbackRequest]) -> List[FeedbackRequest]:
    return [response for response in responses if len(response.ranks) > 0]


class DojoTaskTracker:
    _instance = None
    # request id to miner hotkey to task id
    _rid_to_mhotkey_to_task_id: Dict[str, Dict[str, str]] = defaultdict(
        lambda: defaultdict(str)
    )
    _lock = asyncio.Lock()
    _background_tasks = set()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DojoTaskTracker, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def filter_dojo_responses(
        responses: List[FeedbackRequest],
    ) -> List[FeedbackRequest]:
        return list(filter(lambda r: r.scoring_method == ScoringMethod.DOJO, responses))

    @staticmethod
    def _parse_dojo_task_results(results: List[Dict]):
        """Given a list of task results from Dojo API, calculate the average across multiple task results
        to determine a final ranking"""
        parsed_results = defaultdict(int)

        for result in results:
            for rank, model_hash in result.items():
                rank_int = int(rank)
                parsed_results[model_hash] += rank_int

        sorted_model_hash_count = dict(
            sorted(parsed_results.items(), key=lambda item: item[1])
        )
        # TODO handle edge case where total ranks for all models are the same
        # TODO handle edge case where same ranks sum for 2 different model hashes
        return {
            model_id: rank
            for rank, model_id in enumerate(sorted_model_hash_count.keys(), start=1)
        }

    @classmethod
    async def update_task_map(cls, responses: List[FeedbackRequest]):
        dojo_responses = DojoTaskTracker.filter_dojo_responses(responses)
        async with cls._lock:
            valid_responses = list(
                filter(
                    lambda r: r.request_id and r.axon.hotkey and r.dojo_task_id,
                    dojo_responses,
                )
            )
            bt.logging.info(
                f"Validated N={len(dojo_responses)} responses into {len(valid_responses)} valid responses for request id: {responses[0].request_id}"
            )

            for r in valid_responses:
                cls._rid_to_mhotkey_to_task_id[r.request_id][r.axon.hotkey] = (
                    r.dojo_task_id
                )
        return

    @classmethod
    async def monitor_task_completions(cls):
        SLEEP_SECONDS = 10
        while True:
            bt.logging.info(f"Monitoring Dojo Task completions... {get_epoch_time()}")
            async with cls._lock:
                if not cls._rid_to_mhotkey_to_task_id:
                    bt.logging.warning("No Dojo tasks to monitor")
                    await asyncio.sleep(SLEEP_SECONDS)
                    continue

                for (
                    request_id,
                    miner_to_task_id,
                ) in cls._rid_to_mhotkey_to_task_id.items():
                    for miner_hotkey, task_id in miner_to_task_id.items():
                        if not task_id:
                            bt.logging.warning(
                                f"No task ID found for miner hotkey: {miner_hotkey}"
                            )
                            continue

                        task_results = await DojoAPI.get_task_and_results(task_id)
                        if not task_results:
                            bt.logging.warning(
                                f"Task ID: {task_id} by miner: {miner_hotkey} has not been completed yet or no task results."
                            )
                            continue

                        data = await DataManager.get_by_request_id(request_id)
                        if not data.request:
                            bt.logging.error(
                                f"No request on disk found for request id: {request_id}"
                            )
                            continue

                        model_id_to_rank = DojoTaskTracker._parse_dojo_task_results(
                            task_results
                        )
                        for completion in data.request.completions:
                            if completion.model_id in model_id_to_rank:
                                completion.rank_id = model_id_to_rank[
                                    completion.model_id
                                ]

                        parsed_request = FeedbackRequest(
                            request_id=request_id,
                            completions=data.request.completions,
                        )
                        await DataManager.append_responses(request_id, [parsed_request])
            await asyncio.sleep(SLEEP_SECONDS)


class Validator(BaseNeuron):
    def __init__(self):
        super(Validator, self).__init__()
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

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

    async def send_scores(self, synapse: ScoringResult, hotkeys: List[str]):
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

    def validate_response(self, synapse: FeedbackRequest) -> bool:
        """Process request from miners, specifically filling out the ranks fields"""
        bt.logging.debug(
            f"Processing miner's request for scoring method: {synapse.scoring_method}"
        )
        if CriteriaType.PREFERENCE_RANKING in [synapse.criteria_types]:
            is_missing_ranks = any(
                completion.rank_id is None for completion in synapse.completions
            )
            if is_missing_ranks:
                bt.logging.warning("One or more completions are missing rank IDs.")
                return False
        return True

    async def update_score_and_send_feedback(self):
        """While this function is triggered every X time period in AsyncIOScheduler,
        only relevant data that has passed the deadline of 8 hours will be scored and sent feedback.
        """
        while True:
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
            if not filtered_data:
                bt.logging.warning(
                    "Skipping scoring as no ranking data has been persisted for at least 8 hours."
                )
                return

            bt.logging.info(
                f"Got {len(filtered_data)} requests past deadline and ready to score"
            )
            for d in filtered_data:
                criteria_to_miner_scores = Scoring.calculate_score(
                    criteria_types=d.request.criteria_types,
                    request=d.request,
                    responses=d.responses,
                )

                # calculate mean across all criteria
                mean_miner_scores = (
                    torch.stack(
                        [
                            miner_scores
                            for miner_scores in criteria_to_miner_scores.values()
                        ]
                    )
                    .mean(dim=0)
                    .tolist()
                )
                bt.logging.trace(
                    f"mean miner scores across differerent criteria: {mean_miner_scores.shape} {mean_miner_scores}"
                )

                # craft hotkey to score
                assert len(d.responses) == len(mean_miner_scores)
                hotkey_to_scores = {
                    r.axon.hotkey: mean_miner_scores[i]
                    for i, r in enumerate(d.responses)
                }
                # update the scores based on the rewards
                self.update_scores(hotkey_to_scores=hotkey_to_scores)
                await self.send_scores(
                    synapse=ScoringResult(
                        request_id=d.request.request_id,
                        hotkey_to_scores=hotkey_to_scores,
                    ),
                    hotkeys=list(hotkey_to_scores.keys()),
                )

                wandb_data = jsonable_encoder(
                    {
                        "task": d.request.task_type,
                        "criteria": d.request.criteria_types,
                        "prompt": d.request.prompt,
                        "completions": jsonable_encoder(d.request.completions),
                        "num_completions": len(d.request.completions),
                        "scores": hotkey_to_scores,
                        "num_responses": len(d.responses),
                        "avg_miner_scores": hotkey_to_scores,
                        "miner_scores_by_criteria": criteria_to_miner_scores,
                    }
                )
                asyncio.create_task(wandb_log(wandb_data))

                # once we have scored a response, just remove it
                await DataManager.remove_responses(d)

            await asyncio.sleep(3600)

    async def send_request(
        self,
        synapse: FeedbackRequest = None,
    ):
        # typically the request may come from an external source however,
        # initially will seed it with some data for miners to get started
        if synapse is None:
            data = await build_prompt_responses_pair()
            synapse = FeedbackRequest(
                task_type=TaskType.CODE_GENERATION,
                criteria_types=[CriteriaType.PREFERENCE_RANKING],
                prompt=data["prompt"],
                completions=[Completion.parse_obj(d) for d in data["responses"]],
            )
            bt.logging.info(f"Sending synapse: {synapse.dict()} off to miners")

        all_miner_uids = extract_miner_uids(metagraph=self.metagraph)
        sel_miner_uids = MinerUidSelector(all_miner_uids).get_target_uids(
            key=synapse.request_id, k=get_config().neuron.sample_size
        )
        axons = [
            self.metagraph.axons[uid]
            for uid in sel_miner_uids
            if self.metagraph.axons[uid].hotkey.casefold()
            != self.wallet.hotkey.ss58_address.casefold()
        ]
        if not len(axons):
            bt.logging.warning("No axons to query ... skipping")
            return

        responses: List[FeedbackRequest] = await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=24
        )

        dojo_responses = DojoTaskTracker.filter_dojo_responses(responses)
        await DojoTaskTracker().update_task_map(dojo_responses)
        non_dojo_responses = list(filter(lambda r: r not in dojo_responses, responses))

        response_data = DendriteQueryResponse(
            request=synapse,
            responses=non_dojo_responses,
        )
        await DataManager.save_response(response=response_data)
        return

    async def run(self):
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
                await asyncio.sleep(12)

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

            bt.logging.trace(f"Score for hotkey {key} is {value}")
            rewards[uid] = value

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
        loop.run_until_complete(DataManager.validator_save(self.scores))

    def load_state(self):
        """Loads the state of the validator from a file."""
        loop = asyncio.get_event_loop()
        success, scores, hotkey_to_accuracy = loop.run_until_complete(
            DataManager.validator_load()
        )
        if success:
            self.scores = scores


async def log_validator_status():
    while True:
        bt.logging.info(f"Validator running... {time.time()}")
        await asyncio.sleep(20)


if __name__ == "__main__":

    async def main():
        validator = Validator()
        await asyncio.gather(
            validator.run(),
            DojoTaskTracker().monitor_task_completions(),
            log_validator_status(),
        )

    asyncio.run(main())
