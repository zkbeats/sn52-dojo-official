import asyncio
import copy
import threading
import time
import traceback
from collections import defaultdict
from traceback import print_exception
from typing import Dict, List

import numpy as np
import torch
import wandb
from commons.data_manager import DataManager, ValidatorStateKeys
from commons.dataset.synthetic import SyntheticAPI
from commons.human_feedback.dojo import DojoAPI
from commons.objects import ObjectManager
from commons.scoring import Scoring
from commons.utils import get_epoch_time, get_new_uuid
from fastapi.encoders import jsonable_encoder
from loguru import logger
from template.base.neuron import BaseNeuron
from template.protocol import (
    CriteriaTypeEnum,
    DendriteQueryResponse,
    FeedbackRequest,
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
)
from torch.nn import functional as F

import bittensor as bt


class DojoTaskTracker:
    _instance = None
    # request id -> miner hotkey -> task id
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
            logger.warning("No Dojo responses found")
            return

        logger.debug("update_task_map attempting to acquire lock")
        async with cls._lock:
            valid_responses = list(
                filter(
                    lambda r: r.request_id and r.axon.hotkey and r.dojo_task_id,
                    dojo_responses,
                )
            )
            logger.info(
                f"Got {len(valid_responses)} valid Dojo responses to update task tracker"
            )

            for r in valid_responses:
                if r.request_id not in cls._rid_to_mhotkey_to_task_id:
                    cls._rid_to_mhotkey_to_task_id[r.request_id] = {}

                cls._rid_to_mhotkey_to_task_id[r.request_id][r.axon.hotkey] = (
                    r.dojo_task_id
                )
        logger.debug("released lock for task tracker")
        return

    @classmethod
    async def monitor_task_completions(cls):
        SLEEP_SECONDS = 30
        await asyncio.sleep(60)
        while not cls._should_exit:
            try:
                logger.info(
                    f"Monitoring Dojo Task completions... {get_epoch_time()} for {len(cls._rid_to_mhotkey_to_task_id)} requests"
                )
                if not cls._rid_to_mhotkey_to_task_id:
                    await asyncio.sleep(SLEEP_SECONDS)
                    continue

                for request_id in list(cls._rid_to_mhotkey_to_task_id.keys()):
                    miner_to_task_id = cls._rid_to_mhotkey_to_task_id[request_id]
                    processed_hotkeys = set()

                    data = await DataManager.get_by_request_id(request_id)
                    if not data or not data.request:
                        logger.error(
                            f"No request on disk found for request id: {request_id}"
                        )
                        # del cls._rid_to_mhotkey_to_task_id[request_id]
                        continue

                    for miner_hotkey, task_id in miner_to_task_id.items():
                        if not task_id:
                            logger.warning(
                                f"No task ID found for miner hotkey: {miner_hotkey}"
                            )
                            continue

                        task_results = await DojoAPI.get_task_results_by_task_id(
                            task_id
                        )
                        if not task_results:
                            logger.warning(
                                f"Task ID: {task_id} by miner: {miner_hotkey} has not been completed yet or no task results."
                            )
                            continue

                        logger.info(
                            f"Request id: {request_id}, miner hotkey: {miner_hotkey}, task id: {task_id}"
                        )

                        # calculate average rank/scores across a single miner's workers
                        model_id_to_avg_rank = defaultdict(float)
                        model_id_to_avg_score = defaultdict(float)
                        # keep track so we average across the miner's worker pool
                        num_ranks_by_workers, num_scores_by_workers = 0, 0
                        for result in task_results:
                            for result_data in result["result_data"]:
                                type = result_data["type"]
                                value = result_data["value"]
                                if type == CriteriaTypeEnum.RANKING_CRITERIA:
                                    for rank, model_id in value.items():
                                        model_id_to_avg_rank[model_id] += rank
                                    num_ranks_by_workers += 1
                                elif type == CriteriaTypeEnum.MULTI_SCORE:
                                    for model_id, score in value.items():
                                        model_id_to_avg_score[model_id] += score
                                    num_scores_by_workers += 1

                        # dvide all sums by the number of ranks and scores
                        for model_id in model_id_to_avg_rank:
                            model_id_to_avg_rank[model_id] /= num_ranks_by_workers
                        for model_id in model_id_to_avg_score:
                            model_id_to_avg_score[model_id] /= num_scores_by_workers

                        # mimic miners responding to the dendrite call
                        miner_response = copy.deepcopy(data.request)
                        miner_response.axon = bt.TerminalInfo(
                            hotkey=miner_hotkey,
                        )
                        for completion in miner_response.responses:
                            model_id = completion.model

                            for criteria in miner_response.criteria_types:
                                if isinstance(criteria, RankingCriteria):
                                    completion.rank_id = model_id_to_avg_rank[model_id]
                                elif isinstance(criteria, MultiScoreCriteria):
                                    completion.score = model_id_to_avg_score[model_id]

                        if model_id_to_avg_rank:
                            logger.info(
                                f"Parsed request with ranks data: {model_id_to_avg_rank}"
                            )
                        if model_id_to_avg_score:
                            logger.info(
                                f"Parsed request with scores data: {model_id_to_avg_score}"
                            )

                        # miner would have originally responded with the right task id
                        found_response = next(
                            (
                                r
                                for r in data.miner_responses
                                if r.axon.hotkey == miner_hotkey
                            ),
                            None,
                        )
                        if not found_response:
                            logger.warning(
                                "Miner response not found in data, this should never happen"
                            )
                            data.miner_responses.append(miner_response)
                        else:
                            data.miner_responses.remove(found_response)
                            data.miner_responses.append(miner_response)

                        status = (
                            await DataManager.overwrite_miner_responses_by_request_id(
                                request_id, data.miner_responses
                            )
                        )
                        logger.info(
                            f"Appending Dojo task results for request id: {request_id}, was successful? {status}"
                        )
                        if status:
                            processed_hotkeys.add(miner_hotkey)

                    # determine if we should completely remove the request from the tracker
                    async with cls._lock:
                        if processed_hotkeys == set(miner_to_task_id.keys()):
                            logger.info(
                                f"All hotkeys processed for request id {request_id}, removing from tracker"
                            )
                            del cls._rid_to_mhotkey_to_task_id[request_id]
                        else:
                            for hotkey in processed_hotkeys:
                                del cls._rid_to_mhotkey_to_task_id[request_id][hotkey]

                    ObjectManager.get_validator().save_state()

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during Dojo task monitoring {str(e)}")
                pass
            await asyncio.sleep(SLEEP_SECONDS)

    @classmethod
    async def remove_by_request_id(cls, request_id: str):
        if cls._rid_to_mhotkey_to_task_id.get(request_id, None) is None:
            return
        async with cls._lock:
            del cls._rid_to_mhotkey_to_task_id[request_id]
        return


class Validator(BaseNeuron):
    _should_exit: bool = False
    _is_running: bool = False
    _thread: threading.Thread = None
    _lock = asyncio.Lock()

    def __init__(self):
        super(Validator, self).__init__()

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        self.scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)
        self.load_state()

        # manually always register and always sync metagraph when application starts
        self.check_registered()
        self.resync_metagraph()
        # init_wandb(config=self.config, my_uid=self.uid, wallet=self.wallet)

    async def send_scores(self, synapse: ScoringResult, hotkeys: List[str]):
        """Send consensus score back to miners who participated in the request."""
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        if not axons:
            logger.warning("No axons to send consensus to... skipping")
        else:
            logger.debug(
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
            await asyncio.sleep(60)
            try:
                logger.debug(
                    f"Scheduled update score and send feedback triggered at time: {time.time()}"
                )
                data = await DataManager.load(path=DataManager.get_requests_data_path())
                if not data:
                    logger.debug(
                        "Skipping scoring as no feedback data found, this means either all have been processed or you are running the validator for the first time."
                    )
                    continue

                current_time = get_epoch_time()
                # allow enough time for human feedback
                TASK_DEADLINE = 5 * 60
                filtered_data = [
                    d
                    for d in data
                    if (current_time - d.request.epoch_timestamp) >= TASK_DEADLINE
                ]
                if not filtered_data:
                    logger.warning(
                        "Skipping scoring as no feedback data is due for scoring."
                    )
                    continue

                logger.info(
                    f"Got {len(filtered_data)} requests past deadline and ready to score"
                )
                for d in filtered_data:
                    criteria_to_miner_score, hotkey_to_score = Scoring.calculate_score(
                        criteria_types=d.request.criteria_types,
                        request=d.request,
                        miner_responses=d.miner_responses,
                    )
                    logger.debug(f"Got hotkey to score: {hotkey_to_score}")

                    if not hotkey_to_score:
                        await DataManager.remove_responses([d])
                        continue

                    logger.debug(
                        f"Initiailly had {len(d.miner_responses)} responses from miners, but only {len(hotkey_to_score.keys())} valid responses"
                    )

                    self.update_scores(hotkey_to_scores=hotkey_to_score)
                    await self.send_scores(
                        synapse=ScoringResult(
                            request_id=d.request.request_id,
                            hotkey_to_scores=hotkey_to_score,
                        ),
                        hotkeys=list(hotkey_to_score.keys()),
                    )

                    # TODO fix why after 5 mins this halts validator.run()
                    async def log_wandb():
                        # calculate mean across all criteria
                        mean_weighted_consensus_scores = (
                            torch.stack(
                                [
                                    miner_scores.consensus.score
                                    for miner_scores in criteria_to_miner_score.values()
                                ]
                            )
                            .mean(dim=0)
                            .tolist()
                        )
                        mean_weighted_gt_scores = (
                            torch.stack(
                                [
                                    miner_scores.ground_truth.score
                                    for miner_scores in criteria_to_miner_score.values()
                                ]
                            )
                            .mean(dim=0)
                            .tolist()
                        )

                        logger.info(
                            f"mean miner scores across differerent criteria: consensus shape{mean_weighted_consensus_scores.shape}, gt shape:{mean_weighted_gt_scores.shape}"
                        )

                        score_data = {}
                        # update the scores based on the rewards
                        score_data["scores_by_hotkey"] = {
                            hotkey: score.dict()
                            for hotkey, score in hotkey_to_score.items()
                        }
                        score_data["mean"] = {
                            "consensus": mean_weighted_consensus_scores,
                            "ground_truth": mean_weighted_gt_scores,
                        }

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

                        wandb.log(wandb_data, commit=True)

                    # loop = asyncio.get_running_loop()
                    # await loop.run_in_executor(None, log_wandb)

                    # once we have scored a response, just remove it
                    await DataManager.remove_responses([d])

            except Exception:
                traceback.print_exc()
                pass

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

            obfuscated_model_to_model = {}
            for completion in data.responses:
                new_uuid = get_new_uuid()
                obfuscated_model_to_model[new_uuid] = completion.model
                completion.model = new_uuid

            synapse = FeedbackRequest(
                task_type=str(TaskType.CODE_GENERATION),
                criteria_types=[
                    MultiScoreCriteria(
                        options=list(obfuscated_model_to_model.keys()),
                        min=1.0,
                        max=100.0,
                    ),
                ],
                prompt=data.prompt,
                responses=data.responses,
            )

        all_miner_uids = extract_miner_uids(metagraph=self.metagraph)
        sel_miner_uids = MinerUidSelector(nodes=all_miner_uids).get_target_uids(
            key=synapse.request_id, k=get_config().neuron.sample_size
        )
        logger.info(
            f"Sending synapse off to miners, request id: {synapse.request_id}, miner uids: {sel_miner_uids}"
        )
        axons = [
            self.metagraph.axons[uid]
            for uid in sel_miner_uids
            if self.metagraph.axons[uid].hotkey.casefold()
            != self.wallet.hotkey.ss58_address.casefold()
        ]
        if not len(axons):
            logger.warning("No axons to query ... skipping")
            return

        miner_responses: List[FeedbackRequest] = []
        miner_responses: List[FeedbackRequest] = await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=24
        )
        # try:
        #     # let asyncio timeout take priority
        #     miner_responses = await asyncio.wait_for(
        #         self.dendrite.forward(
        #             axons=axons, synapse=synapse, deserialize=False, timeout=24
        #         ),
        #         timeout=24,
        #     )
        # except asyncio.TimeoutError:
        #     logger.debug("Timeout deadline reached for miners to respond.")
        #     return

        # map obfuscated model names back to the original model names
        logger.debug(f"Obfuscated model map: {obfuscated_model_to_model}")
        valid_miner_responses = []
        try:
            for miner_response in miner_responses:
                completions = [
                    obfuscated_model_to_model.get(completion.model, None)
                    for completion in miner_response.responses
                ]
                if any(c is None for c in completions):
                    logger.warning("Failed to map obfuscated model to original model")
                    continue

                logger.debug(
                    f"Successfully mapped obfuscated model names for {miner_response.axon.hotkey}"
                )

                valid_miner_responses.append(miner_response)
        except Exception as e:
            logger.error(f"Failed to map obfuscated model to original model: {e}")
            pass

        dojo_responses = DojoTaskTracker.filter_dojo_responses(miner_responses)
        logger.debug("Attempting to update task map")
        # TODO FIX DEADLOCK OCCURRING INSIDE HERE
        await DojoTaskTracker.update_task_map(dojo_responses)
        response_data = DendriteQueryResponse(
            request=synapse,
            miner_responses=miner_responses,
        )
        # saving response
        success = await DataManager.save_dendrite_response(response=response_data)
        logger.info(
            f"Saved dendrite response for request id: {response_data.request.request_id}, success: {success}"
        )
        logger.info(
            f"Sending request to miners & processing took {get_epoch_time() - start}"
        )
        return

    async def run(self):
        logger.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        logger.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                try:
                    synthetic_data = await SyntheticAPI.get_qa()
                    await self.send_request(data=synthetic_data)

                    # # Check if we should exit.
                    if self._should_exit:
                        logger.info("Validator should stop...")
                        break

                    # Sync metagraph and potentially set weights.
                    self.sync()

                    self.step += 1
                except Exception as e:
                    logger.error(f"Error during validator run: {e}")
                    pass
                await asyncio.sleep(60)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            logger.error("Error during validation", str(err))
            logger.debug(print_exception(type(err), err, err.__traceback__))

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners.
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        normalized_weights = F.normalize(self.scores.cpu(), p=1, dim=0)

        logger.debug(f"Raw scores: {self.scores}")
        logger.debug(f"normalized weights: {normalized_weights}")
        logger.debug(f"normalized weights uids: {self.metagraph.uids}")

        if torch.count_nonzero(normalized_weights).item() == 0:
            logger.warning("All weights are zero, skipping...")
            return

        logger.info("Attempting to set weights")

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
        logger.debug(f"processed weights {processed_weights}")
        logger.debug(f"processed weights uids {processed_weight_uids}")

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
            logger.success("Validator set weights on chain successfully!")
        else:
            logger.error("set_weights failed")
        return result

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        logger.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(previous_metagraph.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(previous_metagraph.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(previous_metagraph.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

    def update_scores(self, hotkey_to_scores):
        """Performs exponential moving average on the scores based on the rewards received from the miners,
        after setting the self.scores variable here, `set_weights` will be called to set the weights on chain."""

        nan_value_indices = np.isnan(list(hotkey_to_scores.values()))
        if nan_value_indices.any():
            logger.warning(f"NaN values detected in rewards: {hotkey_to_scores}")
            return

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
                logger.warning(
                    "Old hotkey found from previous metagraph, skip setting weights"
                )
                continue

            logger.trace(f"Score for hotkey {key} is {value}")
            rewards[uid] = value

        logger.debug(f"Rewards: {rewards}")
        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * rewards + (1 - alpha) * self.scores
        logger.debug(f"Updated scores: {self.scores}")

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
            logger.info(f"Validator running... {time.time()}")
            await asyncio.sleep(20)
