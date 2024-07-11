import asyncio
import copy
import traceback
from collections import defaultdict
from typing import Dict, List

import bittensor as bt
from loguru import logger

import template
from commons.data_manager import DataManager
from commons.human_feedback.dojo import DojoAPI
from commons.objects import ObjectManager
from commons.utils import (
    get_current_utc_time_iso,
    get_epoch_time,
    is_valid_expiry,
    set_expire_time,
)
from template.protocol import (
    CriteriaTypeEnum,
    FeedbackRequest,
    MultiScoreCriteria,
    RankingCriteria,
    ScoringMethod,
)


class DojoTaskTracker:
    _instance = None
    # request id -> miner hotkey -> task id
    _rid_to_mhotkey_to_task_id: Dict[str, Dict[str, str]] = defaultdict(
        lambda: defaultdict(str)
    )
    _rid_to_model_map: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(str))
    _task_to_expiry: Dict[str, str] = defaultdict(str)
    _lock = asyncio.Lock()
    _should_exit: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def filter_dojo_responses(
        responses: List[FeedbackRequest],
    ) -> List[FeedbackRequest]:
        return list(filter(lambda r: r.scoring_method == ScoringMethod.DOJO, responses))

    @classmethod
    async def update_task_map(
        cls,
        request_id: str,
        dojo_responses: List[FeedbackRequest],
        obfuscated_model_to_model: Dict,
    ):
        if not dojo_responses:
            logger.warning("No Dojo responses found")
            return

        logger.debug("update_task_map attempting to acquire lock")
        async with cls._lock:
            valid_responses: List[FeedbackRequest] = list(
                filter(
                    lambda r: r.request_id == request_id
                    and r.axon.hotkey
                    and r.dojo_task_id,
                    dojo_responses,
                )
            )
            logger.info(
                f"Got {len(valid_responses)} valid Dojo responses to update task tracker"
            )

            if request_id not in cls._rid_to_mhotkey_to_task_id:
                cls._rid_to_mhotkey_to_task_id[request_id] = {}

            for r in valid_responses:
                cls._rid_to_mhotkey_to_task_id[request_id][r.axon.hotkey] = (
                    r.dojo_task_id
                )

                # Ensure expire_at is set and is reasonable
                expire_at = r.expire_at
                if expire_at is None or is_valid_expiry(expire_at) is not True:
                    expire_at = set_expire_time(template.TASK_DEADLINE)

                cls._task_to_expiry[r.dojo_task_id] = expire_at

            cls._rid_to_model_map[request_id] = obfuscated_model_to_model
        logger.debug("released lock for task tracker")
        return

    @classmethod
    async def remove_expired_tasks(cls):
        # Identify expired tasks
        current_time = get_current_utc_time_iso()
        expired_tasks = [
            task_id
            for task_id, expiry_time in cls._task_to_expiry.items()
            if expiry_time < current_time
        ]

        async with cls._lock:
            for task_id in expired_tasks:
                # Remove from _rid_to_mhotkey_to_task_id
                for request_id, hotkeys in list(cls._rid_to_mhotkey_to_task_id.items()):
                    for hotkey, t_id in list(hotkeys.items()):
                        if t_id == task_id:
                            del cls._rid_to_mhotkey_to_task_id[request_id][hotkey]
                    if not cls._rid_to_mhotkey_to_task_id[request_id]:
                        del cls._rid_to_mhotkey_to_task_id[request_id]
                # Remove from _task_to_expiry
                del cls._task_to_expiry[task_id]

        logger.info(f"Removed {len(expired_tasks)} expired tasks from DojoTaskTracker.")

    @classmethod
    async def monitor_task_completions(cls):
        SLEEP_SECONDS = 30
        await asyncio.sleep(60)

        while not cls._should_exit:
            try:
                logger.info(
                    f"Monitoring Dojo Task completions... {get_epoch_time()} for {len(cls._rid_to_mhotkey_to_task_id)} requests"
                )
                logger.info(f"print task_to_expiry {cls._task_to_expiry}")

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
