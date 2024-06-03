import asyncio
import copy
import threading
import time
import traceback
from collections import defaultdict
from traceback import print_exception
from typing import Dict, List, Tuple

import bittensor as bt
import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from loguru import logger
from torch.nn import functional as F

import wandb
from commons.data_manager import DataManager, ValidatorStateKeys
from commons.dataset.synthetic import SyntheticAPI
from commons.human_feedback.aws_mturk import MTurkUtils, parse_assignment
from commons.human_feedback.dojo import DojoAPI
from commons.scoring import Scoring
from commons.utils import get_epoch_time
from template.base.neuron import BaseNeuron
from template.protocol import (
    AWSCredentials,
    CriteriaTypeEnum,
    DendriteQueryResponse,
    FeedbackRequest,
    MTurkResponse,
    MultiScoreCriteria,
    RankingCriteria,
    ScoringMethod,
    ScoringResult,
    SyntheticQA,
    TaskType,
)
from template.utils.config import get_config
from template.utils.uids import (
    MinerUidSelector,
    extract_miner_uids,
    is_miner,
)


class DojoTaskTracker:
    _instance = None
    _rid_to_mhotkey_to_task_id: Dict[str, Dict[str, str]] = defaultdict(
        lambda: defaultdict(str)
    )
    _lock = asyncio.Lock()
    _should_exit: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DojoTaskTracker, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def filter_dojo_responses(
        responses: List[FeedbackRequest],
    ) -> List[FeedbackRequest]:
        return list(filter(lambda r: r.scoring_method == ScoringMethod.DOJO, responses))

    @classmethod
    async def update_task_map(cls, dojo_responses: List[FeedbackRequest]):
        if not dojo_responses:
            bt.logging.warning("No Dojo responses found")
            return

        async with cls._lock:
            valid_responses = list(
                filter(
                    lambda r: r.request_id and r.axon.hotkey and r.dojo_task_id,
                    dojo_responses,
                )
            )
            bt.logging.info(
                f"Got {len(valid_responses)} valid Dojo responses to update task tracker"
            )

            for r in valid_responses:
                if r.request_id not in cls._rid_to_mhotkey_to_task_id:
                    cls._rid_to_mhotkey_to_task_id[r.request_id] = {}

                cls._rid_to_mhotkey_to_task_id[r.request_id][r.axon.hotkey] = (
                    r.dojo_task_id
                )
        return

    @classmethod
    async def monitor_task_completions(cls):
        SLEEP_SECONDS = 30
        while not cls._should_exit:
            try:
                logger.info(f"Monitoring Dojo Task completions... {get_epoch_time()}")
                async with cls._lock:
                    if not cls._rid_to_mhotkey_to_task_id:
                        bt.logging.warning("No Dojo tasks to monitor")
                        await asyncio.sleep(SLEEP_SECONDS)
                        continue
                    else:
                        logger.info(
                            f"Monitoring Dojo tasks: {cls._rid_to_mhotkey_to_task_id}"
                        )

                    for (
                        request_id,
                        miner_to_task_id,
                    ) in cls._rid_to_mhotkey_to_task_id.items():
                        for miner_hotkey, task_id in miner_to_task_id.items():
                            logger.info(
                                f"Miner hotkey: {miner_hotkey}, task_id: {task_id}, request id: {request_id}"
                            )
                            if not task_id:
                                bt.logging.warning(
                                    f"No task ID found for miner hotkey: {miner_hotkey}"
                                )
                                continue

                            task_results = await DojoAPI.get_task_results_by_task_id(
                                task_id
                            )
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

                            # calculate average rank/scores across a single miner's workers
                            model_id_to_avg_rank = defaultdict(float)
                            model_id_to_avg_score = defaultdict(float)
                            num_ranks, num_scores = 0, 0
                            for result in task_results:
                                for result_data in result["result_data"]:
                                    type = result_data["type"]
                                    value = result_data["value"]
                                    if type == CriteriaTypeEnum.RANKING_CRITERIA:
                                        for rank, model_id in value.items():
                                            model_id_to_avg_rank[model_id] += rank
                                            num_ranks += 1
                                    elif type == CriteriaTypeEnum.MULTI_SCORE:
                                        for model_id, score in value.items():
                                            model_id_to_avg_score[model_id] += score
                                            num_scores += 1

                            # dvide all sums by the number of ranks and scores
                            for model_id in model_id_to_avg_rank:
                                model_id_to_avg_rank[model_id] /= num_ranks
                            for model_id in model_id_to_avg_score:
                                model_id_to_avg_score[model_id] /= num_scores

                            # mimic miners responding to the dendrite call
                            miner_response = copy.deepcopy(data.request)
                            miner_response.axon = bt.TerminalInfo(
                                hotkey=miner_hotkey,
                            )
                            for completion in miner_response.responses:
                                model_id = completion.model
                                if RankingCriteria in miner_response.criteria_types:
                                    completion.rank_id = model_id_to_avg_rank[model_id]

                                if MultiScoreCriteria in miner_response.criteria_types:
                                    completion.score = model_id_to_avg_score[model_id]

                            if model_id_to_avg_rank:
                                logger.info(
                                    f"Parsed request with ranks data: {model_id_to_avg_rank}"
                                )
                            if model_id_to_avg_score:
                                logger.info(
                                    f"Parsed request with scores data: {model_id_to_avg_score}"
                                )

                            logger.info(
                                f"Appending Dojo task results for request id: {request_id}"
                            )
                            await DataManager.append_responses(
                                request_id, [miner_response]
                            )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during Dojo task monitoring {str(e)}")
                pass
            await asyncio.sleep(SLEEP_SECONDS)


class Validator(BaseNeuron):
    _should_exit: bool = False
    _is_running: bool = False
    _thread: threading.Thread = None
    _lock = asyncio.Lock()

    def __init__(self):
        super(Validator, self).__init__()

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        self.scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)
        self.load_state()

        # manually always register and always sync metagraph when application starts
        self.check_registered()
        self.resync_metagraph()
        # init_wandb(config=self.config, my_uid=self.uid, wallet=self.wallet)

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

    # async def forward_mturk_response(self, synapse: MTurkResponse):
    #     """Receives MTurk responses from miners after delayed response to allow for human feedback loop"""
    #     # 1. check request from RankingRequest
    #     # 2. append completion scores to request
    #     # 3. persist on disk via DataManager

    #     path = DataManager.get_ranking_data_filepath()
    #     data = await DataManager.load(path=path)
    #     miner_hotkey = synapse.dendrite.hotkey
    #     if not data:
    #         bt.logging.error(
    #             f"Received MTurk response from miner {miner_hotkey} but no requests persisted on disk."
    #         )
    #         return

    #     mturk_cids = set(synapse.completion_id_to_score.keys())

    #     for d in data:
    #         request_cids = set([completion.cid for completion in d.request.completions])

    #         if (found_cids := request_cids.intersection(mturk_cids)) and not found_cids:
    #             bt.logging.warning(
    #                 f"Received mturk response with {mturk_cids=}, but no found requests with those matching CIDs."
    #             )
    #             continue

    #         bt.logging.info(
    #             f"Miner {miner_hotkey} sent human feedback for {d.request.request_id}"
    #         )

    #         existing_responses = {r.axon.hotkey: r for r in d.responses}
    #         if miner_hotkey in existing_responses:
    #             existing_method = ScoringMethod(
    #                 existing_responses[miner_hotkey].scoring_method
    #             )
    #             new_method = ScoringMethod.AWS_MTURK
    #             if (
    #                 SCORING_METHOD_PRIORITY[new_method]
    #                 > SCORING_METHOD_PRIORITY[existing_method]
    #             ):
    #                 bt.logging.info(
    #                     f"Replacing {existing_method} with higher priority {new_method} for request id: {d.request.request_id}"
    #                 )
    #                 d.responses.remove(existing_responses[miner_hotkey])
    #             else:
    #                 bt.logging.warning(
    #                     f"Miner {miner_hotkey} already sent response with equal or higher priority for request id: {d.request.request_id}, skipping..."
    #                 )
    #                 continue

    #         is_verified, reason = self.verify_mturk_task(
    #             synapse.aws_credentials,
    #             synapse.mturk_hit_id,
    #             synapse.completion_id_to_score,
    #         )
    #         if not is_verified:
    #             bt.logging.error(
    #                 f"MTurk task verification failed due to reason:{reason}"
    #             )
    #             return

    #         request_copy = copy.deepcopy(d.request)
    #         for cid in found_cids:
    #             # similar to scoring method in Miner.forward(...)
    #             request_copy.ranks.append(
    #                 Rank(
    #                     cid=cid,
    #                     score=synapse.completion_id_to_score[cid],
    #                     scoring_method=ScoringMethod.AWS_MTURK,
    #                 )
    #             )
    #             # ensure axon.hotkey has miner's hotkey because our Consensus._spearman_correlation
    #             # function expects axon.hotkey to be the miner's hotkey
    #             request_copy.axon.hotkey = miner_hotkey
    #             d.responses.append(request_copy)

    #         d.responses = _filter_valid_responses(d.responses)
    #     await DataManager.save(path, data)
    #     return

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

    async def update_score_and_send_feedback(self):
        """While this function is triggered every X time period in AsyncIOScheduler,
        only relevant data that has passed the deadline of 8 hours will be scored and sent feedback.
        """
        while True:
            bt.logging.debug(
                f"Scheduled update score and send feedback triggered at time: {time.time()}"
            )
            data = await DataManager.load(path=DataManager.get_requests_data_path())
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
                criteria_to_miner_score = Scoring.calculate_score(
                    criteria_types=d.request.criteria_types,
                    request=d.request,
                    miner_responses=d.miner_responses,
                )

                GT_WEIGHT = 0.4
                CONSENSUS_WEIGHT = 0.6
                # log_data is score by each miner
                score_data = {}
                assert len(criteria_to_miner_score.keys()) == 1
                for criteria in criteria_to_miner_score:
                    miner_scores = criteria_to_miner_score[criteria]
                    weighted_gt = GT_WEIGHT * miner_scores.ground_truth
                    weighted_consensus = CONSENSUS_WEIGHT * miner_scores.consensus
                    # update weighted scores
                    criteria_to_miner_score[
                        criteria
                    ].weighted_consensus = weighted_consensus
                    criteria_to_miner_score[
                        criteria
                    ].weighted_ground_truth = weighted_gt

                    hotkey_to_score = {
                        r.axon.hotkey: criteria_to_miner_score[criteria][i]
                        for i, r in enumerate(d.miner_responses)
                    }

                    self.update_scores(hotkey_to_scores=hotkey_to_score)
                    await self.send_scores(
                        synapse=ScoringResult(
                            request_id=d.request.request_id,
                            hotkey_to_scores=hotkey_to_score,
                        ),
                        hotkeys=list(hotkey_to_score.keys()),
                    )

                    # calculate mean across all criteria
                    mean_weighted_consensus_scores = (
                        torch.stack(
                            [
                                miner_scores.consensus
                                for miner_scores in criteria_to_miner_score.values()
                            ]
                        )
                        .mean(dim=0)
                        .tolist()
                    )
                    mean_weighted_gt_scores = (
                        torch.stack(
                            [
                                miner_scores.ground_truth
                                for miner_scores in criteria_to_miner_score.values()
                            ]
                        )
                        .mean(dim=0)
                        .tolist()
                    )

                    bt.logging.info(
                        f"mean miner scores across differerent criteria: consensus shape{mean_weighted_consensus_scores.shape}, gt shape:{mean_weighted_gt_scores.shape}"
                    )

                    # craft hotkey to score
                    assert len(d.miner_responses) == len(mean_weighted_consensus_scores)
                    assert len(d.miner_responses) == len(mean_weighted_consensus_scores)
                    # update the scores based on the rewards
                    score_data["scores_by_hotkey"] = {
                        hotkey: score.dict()
                        for hotkey, score in hotkey_to_score.items()
                    }
                    score_data["mean"] = {
                        "consensus": mean_weighted_consensus_scores,
                        "ground_truth": mean_weighted_gt_scores,
                    }
                    self.update_scores(hotkey_to_scores=hotkey_to_score)
                    await self.send_scores(
                        synapse=ScoringResult(
                            request_id=d.request.request_id,
                            hotkey_to_scores=hotkey_to_score,
                        ),
                        hotkeys=list(hotkey_to_score.keys()),
                    )

                wandb_data = jsonable_encoder(
                    {
                        "task": d.request.task_type,
                        "criteria": d.request.criteria_types,
                        "prompt": d.request.prompt,
                        "completions": jsonable_encoder(d.request.responses),
                        "num_completions": len(d.request.responses),
                        "scores": score_data,
                        "num_responses": len(d.miner_responses),
                    }
                )

                async def _log_wandb(wandb_data: Dict):
                    loop = asyncio.get_running_loop()

                    def log_wandb(data: dict):
                        wandb.log(data, sync=False)

                    await loop.run_in_executor(None, log_wandb, wandb_data)

                asyncio.create_task(_log_wandb(wandb_data))

                # once we have scored a response, just remove it
                await DataManager.remove_responses(d)

            await asyncio.sleep(3600)

    async def send_request(
        self,
        synapse: FeedbackRequest = None,
        data: SyntheticQA = None,
    ):
        start = get_epoch_time()
        # typically the request may come from an external source however,
        # initially will seed it with some data for miners to get started
        if synapse is None:
            if not data:
                logger.error("Failed to generate data from synthetic gen API")
                return

            synapse = FeedbackRequest(
                task_type=str(TaskType.CODE_GENERATION),
                criteria_types=[
                    MultiScoreCriteria(
                        options=[completion.model for completion in data.responses],
                        min=1.0,
                        max=100.0,
                    ),
                ],
                prompt=data.prompt,
                responses=data.responses,
            )
            bt.logging.info(
                f"Sending synapse request id: {synapse.request_id} off to miners"
            )

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
        await DojoTaskTracker.update_task_map(dojo_responses)
        non_dojo_responses = list(filter(lambda r: r not in dojo_responses, responses))

        response_data = DendriteQueryResponse(
            request=synapse,
            miner_responses=non_dojo_responses,
        )
        await DataManager.save_response(response=response_data)
        bt.logging.info(
            f"Sending request to miners & processing took {get_epoch_time() - start}"
        )
        return

    async def run(self):
        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                synthetic_data = await SyntheticAPI.get_qa()
                await self.send_request(data=synthetic_data)

                # # Check if we should exit.
                if self._should_exit:
                    bt.logging.info("Validator should stop...")
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1
                await asyncio.sleep(60)

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
            bt.logging.warning("All weights are zero, skipping...")
            return

        bt.logging.success("Attempting to set weights")

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
            return

        # check for preserved hotkeys based on prev metagraph...
        preserved_uids = [
            uid
            for uid, hotkey in enumerate(previous_metagraph.hotkeys)
            if hotkey == self.metagraph.hotkeys[uid]
        ]

        new_uids = list(range(previous_metagraph.n.item(), self.metagraph.n.item()))
        new_axons = [
            axon
            for axon in self.metagraph.axons
            if axon not in previous_metagraph.axons
        ]
        bt.logging.info(
            f"Metagraph updated, new uids: {new_uids}, new axons: {new_axons}"
        )

        # create a new score tensor since metagraph size is different
        updated_scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)

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
        rewards = np.zeros((len(self.metagraph.axons),))
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
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                DataManager.validator_save(
                    self.scores, DojoTaskTracker._rid_to_mhotkey_to_task_id
                )
            )
        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")
            pass

    def load_state(self):
        """Loads the state of the validator from a file."""
        loop = asyncio.get_event_loop()
        state_data = loop.run_until_complete(DataManager.validator_load())
        if state_data is None:
            logger.error("Failed to load validator state data")
            return

        logger.success("Loaded validator state successfully")
        self.scores = state_data[ValidatorStateKeys.SCORES]
        DojoTaskTracker._rid_to_mhotkey_to_task_id = state_data[
            ValidatorStateKeys.DOJO_TASKS_TO_TRACK
        ]

        logger.info(f"Scores state: {self.scores}")
        logger.info(
            f"Dojo Tasks to track: {DojoTaskTracker._rid_to_mhotkey_to_task_id}"
        )

    @classmethod
    async def log_validator_status(cls):
        while not cls._should_exit:
            bt.logging.info(f"Validator running... {time.time()}")
            await asyncio.sleep(20)
