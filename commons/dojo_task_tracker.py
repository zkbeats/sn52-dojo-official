import asyncio
import copy
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict

import bittensor as bt
from bittensor.btlogging import logging as logger

import template
from commons.data_manager import DataManager
from commons.objects import ObjectManager
from commons.utils import get_epoch_time
from database.prisma.models import Feedback_Request_Model, Miner_Response_Model
from template.protocol import (
    CriteriaTypeEnum,
    MultiScoreCriteria,
    RankingCriteria,
    RidToHotKeyToTaskId,
    RidToModelMap,
    TaskExpiryDict,
    TaskResult,
    TaskResultRequest,
)


class DojoTaskTracker:
    _instance = None
    # request id -> miner hotkey -> task id
    _rid_to_mhotkey_to_task_id: RidToHotKeyToTaskId = defaultdict(
        lambda: defaultdict(str)
    )
    _rid_to_model_map: RidToModelMap = defaultdict(lambda: defaultdict(str))
    _task_to_expiry: TaskExpiryDict = defaultdict(str)
    _lock = asyncio.Lock()
    _should_exit: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def update_task_map(
        cls,
        request_id: str,
        fb_request_model: Feedback_Request_Model,
        obfuscated_model_to_model: Dict,
    ):
        dojo_responses = fb_request_model.miner_responses
        if dojo_responses is None or len(dojo_responses) == 0:
            logger.warning("No Dojo responses found")
            return

        logger.debug("update_task_map attempting to acquire lock")
        async with cls._lock:
            valid_responses: list[Miner_Response_Model] = list(
                filter(
                    lambda r: r.request_id == request_id
                    and r.miner_hotkey
                    and r.dojo_task_id,
                    dojo_responses,
                )
            )
            logger.debug(
                f"Got {len(valid_responses)} valid Dojo responses to update task tracker"
            )

            if request_id not in cls._rid_to_mhotkey_to_task_id:
                cls._rid_to_mhotkey_to_task_id[request_id] = {}

            for r in valid_responses:
                cls._rid_to_mhotkey_to_task_id[request_id][r.miner_hotkey] = (
                    r.dojo_task_id
                )

                cls._task_to_expiry[r.dojo_task_id] = r.expire_at

            cls._rid_to_model_map[request_id] = obfuscated_model_to_model
        logger.debug("released lock for task tracker")
        return

    @classmethod
    async def remove_expired_tasks(cls):
        # Identify expired tasks
        current_time = datetime.now(timezone.utc)
        expired_tasks = [
            task_id
            for task_id, expiry_time in cls._task_to_expiry.items()
            if datetime.fromisoformat(expiry_time) < current_time
        ]

        async with cls._lock:
            for task_id in expired_tasks:
                # Remove from _rid_to_mhotkey_to_task_id
                for request_id, hotkeys in list(cls._rid_to_mhotkey_to_task_id.items()):
                    for hotkey, t_id in list(hotkeys.items()):
                        if t_id == task_id:
                            del cls._rid_to_mhotkey_to_task_id[request_id][hotkey]
                    if not cls._rid_to_mhotkey_to_task_id[
                        request_id
                    ]:  # This means no more hotkeys for this request, so remove the request
                        del cls._rid_to_mhotkey_to_task_id[request_id]
                # Remove from _task_to_expiry
                del cls._task_to_expiry[task_id]

        logger.info(f"Removed {len(expired_tasks)} expired tasks from DojoTaskTracker.")

    @classmethod
    async def get_task_results_from_miner(
        cls, miner_hotkey: str, task_id: str
    ) -> list[TaskResult]:
        """Fetch task results from the miner's Axon using Dendrite."""
        try:
            logger.info(
                f"Fetching task result from miner {miner_hotkey} for task {task_id}"
            )

            validator = ObjectManager.get_validator()

            dendrite: bt.dendrite = validator.dendrite
            metagraph = validator.metagraph

            if not dendrite:
                raise ValueError("Dendrite not initialized")

            # Prepare the synapse (data request) that will be sent via Dendrite
            task_synapse = TaskResultRequest(task_id=task_id)

            # Use Dendrite to communicate with the Axon
            miner_axon = metagraph.axons[metagraph.hotkeys.index(miner_hotkey)]
            if not miner_axon:
                raise ValueError(f"Miner Axon not found for hotkey: {miner_hotkey}")

            # Send the request via Dendrite and get the response
            response = await dendrite.forward(
                axons=[miner_axon], synapse=task_synapse, deserialize=False
            )

            logger.debug(f"TaskResult Response from miner {miner_hotkey}: {response}")

            if response and response[0]:
                logger.info(
                    f"Received task result from miner {miner_hotkey} for task {task_id}"
                )
                return response[0].task_results
            else:
                logger.warning(
                    f"No task results found from miner {miner_hotkey} for task {task_id}"
                )
                return []

        except Exception as e:
            logger.error(f"Error fetching task result from miner {miner_hotkey}: {e}")
            return []

    @classmethod
    async def monitor_task_completions(cls):
        SLEEP_SECONDS = 30
        await asyncio.sleep(template.DOJO_TASK_MONITORING)

        while not cls._should_exit:
            try:
                logger.info(
                    f"Monitoring Dojo Task completions... {get_epoch_time()} for {len(cls._rid_to_mhotkey_to_task_id)} requests"
                )

                # Clean up expired tasks before processing
                await cls.remove_expired_tasks()
                await DataManager.remove_expired_tasks_from_storage()

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
                        continue

                    for miner_hotkey, task_id in miner_to_task_id.items():
                        if not task_id:
                            logger.warning(
                                f"No task ID found for miner hotkey: {miner_hotkey}"
                            )
                            continue

                        task_results = await cls.get_task_results_from_miner(
                            miner_hotkey, task_id
                        )

                        if not task_results and not len(task_results) > 0:
                            logger.warning(
                                f"Task ID: {task_id} by miner: {miner_hotkey} has not been completed yet or no task results."
                            )
                            continue

                        logger.trace(
                            f"Request id: {request_id}, miner hotkey: {miner_hotkey}, task id: {task_id}"
                        )

                        # calculate average rank/scores across a single miner's workers
                        model_id_to_avg_rank = defaultdict(float)
                        model_id_to_avg_score = defaultdict(float)
                        # keep track so we average across the miner's worker pool
                        num_ranks_by_workers, num_scores_by_workers = 0, 0
                        for result in task_results:
                            for result_data in result.result_data:
                                type = result_data.type
                                value = result_data.value
                                if type == CriteriaTypeEnum.RANKING_CRITERIA:
                                    for model_id, rank in value.items():
                                        real_model_id = cls._rid_to_model_map.get(
                                            request_id
                                        ).get(model_id)
                                        model_id_to_avg_rank[real_model_id] += rank
                                    num_ranks_by_workers += 1
                                elif type == CriteriaTypeEnum.MULTI_SCORE:
                                    for model_id, score in value.items():
                                        real_model_id = cls._rid_to_model_map.get(
                                            request_id
                                        ).get(model_id)
                                        model_id_to_avg_score[real_model_id] += score
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
                            logger.trace(
                                f"Parsed request with ranks data: {model_id_to_avg_rank}"
                            )
                        if model_id_to_avg_score:
                            logger.trace(
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
                        logger.trace(
                            f"Appending Dojo task results for request id: {request_id}, was successful? {status}"
                        )
                        if status:
                            processed_hotkeys.add(miner_hotkey)

                    # determine if we should completely remove the request from the tracker
                    async with cls._lock:
                        if processed_hotkeys == set(miner_to_task_id.keys()):
                            del cls._rid_to_mhotkey_to_task_id[request_id]
                            del cls._rid_to_model_map[request_id]
                        else:
                            for hotkey in processed_hotkeys:
                                del cls._rid_to_mhotkey_to_task_id[request_id][hotkey]

                    ObjectManager.get_validator().save_state()

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during Dojo task monitoring {str(e)}")
                pass
            await asyncio.sleep(SLEEP_SECONDS)
