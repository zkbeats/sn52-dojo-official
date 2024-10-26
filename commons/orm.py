import asyncio
import gc
import json
import math
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, List

import torch
from bittensor.btlogging import logging as logger

from commons.exceptions import (
    InvalidCompletion,
    InvalidMinerResponse,
    InvalidTask,
    NoNewUnexpiredTasksYet,
    UnexpiredTasksAlreadyProcessed,
)
from commons.utils import datetime_as_utc
from database.client import prisma, transaction
from database.mappers import (
    map_child_feedback_request_to_model,
    map_completion_response_to_model,
    map_criteria_type_to_model,
    map_feedback_request_model_to_feedback_request,
    map_parent_feedback_request_to_model,
)
from database.prisma import Json
from database.prisma.errors import PrismaError
from database.prisma.models import (
    Feedback_Request_Model,
    Ground_Truth_Model,
    Score_Model,
)
from database.prisma.types import (
    Completion_Response_ModelWhereInput,
    Completion_Response_ModelWhereUniqueInput,
    Feedback_Request_ModelInclude,
    Feedback_Request_ModelWhereInput,
    Ground_Truth_ModelCreateInput,
    Score_ModelCreateInput,
    Score_ModelUpdateInput,
)
from dojo import TASK_DEADLINE
from dojo.protocol import (
    CodeAnswer,
    CompletionResponses,
    DendriteQueryResponse,
    FeedbackRequest,
)


class ORM:
    @staticmethod
    async def get_last_expire_at_cutoff(
        validator_hotkeys: list[str],
        expire_at: datetime = datetime_as_utc(
            datetime.now(timezone.utc) - timedelta(seconds=TASK_DEADLINE)
        ),
    ) -> datetime:
        """
        Get the expire at cutoff for the query to `get_unexpired_tasks`
        We use 1.5 * TASK_DEADLINE to overlap with the `expire_at` field in the
        database so we don't miss out any data.

        Args:
            validator_hotkeys (list[str]): List of validator hotkeys.
            expire_at (datetime, optional): _description_. Defaults to datetime_as_utc( datetime.now(timezone.utc) - 1.5 * timedelta(seconds=TASK_DEADLINE) ).

        Raises:
            ValueError: Unable to determine expire at cutoff

        Returns:
            datetime: Expire at cutoff
        """
        logger.debug(f"Expire at cutoff: {expire_at}")
        vali_where_query_unprocessed = Feedback_Request_ModelWhereInput(
            {
                "hotkey": {"in": validator_hotkeys, "mode": "insensitive"},
                "child_requests": {"some": {}},
                "expire_at": {"gt": expire_at},
                "is_processed": {"equals": False},
            }
        )

        found = await Feedback_Request_Model.prisma().find_first(
            where=vali_where_query_unprocessed,
            order={"expire_at": "asc"},
        )
        if found:
            return datetime_as_utc(found.expire_at)

        raise ValueError("Unable to determine expire at cutoff")

    @staticmethod
    async def get_expired_tasks(
        validator_hotkeys: list[str],
        batch_size: int = 10,
        expire_at: datetime | None = None,
    ) -> AsyncGenerator[tuple[List[DendriteQueryResponse], bool], None]:
        """Returns a batch of Feedback_Request_Model and a boolean indicating if there are more batches.
        Depending on the `expire_at` provided, it will return different results.

        YOUR LOGIC ON WHETHER TASKS ARE EXPIRED OR NON-EXPIRED SHOULD BE HANDLED BY SETTING EXPIRE_AT YOURSELF.

        Args:
            validator_hotkeys (list[str]): List of validator hotkeys.
            batch_size (int, optional): Number of tasks to return in a batch. Defaults to 10.

            1 task == 1 validator request, N miner responses
            expire_at: (datetime | None) If provided, only tasks with expire_at after the provided datetime will be returned.
            You must determine the `expire_at` cutoff yourself, otherwise it defaults to current time UTC.

        Raises:
            NoNewUnexpiredTasksYet: If no unexpired tasks are found for processing.
            UnexpiredTasksAlreadyProcessed: If all unexpired tasks have already been processed.

        Yields:
            Iterator[AsyncGenerator[tuple[List[DendriteQueryResponse], bool], None]]:
                Returns a batch of DendriteQueryResponse and a boolean indicating if there are more batches

        """

        # find all validator requests first
        include_query = Feedback_Request_ModelInclude(
            {
                "completions": True,
                "criteria_types": True,
                "ground_truths": True,
                "parent_request": True,
            }
        )

        now = datetime_as_utc(datetime.now(timezone.utc))
        if expire_at:
            now = expire_at

        vali_where_query_unprocessed = Feedback_Request_ModelWhereInput(
            {
                "hotkey": {"in": validator_hotkeys, "mode": "insensitive"},
                "child_requests": {"some": {}},
                # only check for expire at since miner may lie
                "expire_at": {
                    "gt": now,
                },
                "is_processed": {"equals": False},
            }
        )

        vali_where_query_processed = Feedback_Request_ModelWhereInput(
            {
                "hotkey": {"in": validator_hotkeys, "mode": "insensitive"},
                "child_requests": {"some": {}},
                # only check for expire at since miner may lie
                "expire_at": {
                    "gt": now,
                },
                "is_processed": {"equals": True},
            }
        )

        # count first total including non
        task_count_unprocessed = await Feedback_Request_Model.prisma().count(
            where=vali_where_query_unprocessed,
        )

        task_count_processed = await Feedback_Request_Model.prisma().count(
            where=vali_where_query_processed,
        )

        logger.debug(f"Count of unprocessed tasks: {task_count_unprocessed}")
        logger.debug(f"Count of processed tasks: {task_count_processed}")

        logger.debug(
            f"Count of unprocessed tasks: {task_count_unprocessed}, count of processed tasks: {task_count_processed}"
        )

        if not task_count_unprocessed:
            if task_count_processed:
                raise UnexpiredTasksAlreadyProcessed(
                    f"No remaining unexpired tasks found for processing, but don't worry as you have processed {task_count_processed} tasks."
                )
            else:
                raise NoNewUnexpiredTasksYet(
                    f"No unexpired tasks found for processing, please wait for tasks to pass the task deadline of {TASK_DEADLINE} seconds."
                )

        for i in range(0, task_count_unprocessed, batch_size):
            # find all validator requests
            validator_requests = await Feedback_Request_Model.prisma().find_many(
                include=include_query,
                where=vali_where_query_unprocessed,
                order={"created_at": "desc"},
                skip=i,
                take=batch_size,
            )

            # find all miner responses
            validator_request_ids = [r.id for r in validator_requests]

            miner_responses = await Feedback_Request_Model.prisma().find_many(
                include=include_query,
                where={
                    "parent_id": {"in": validator_request_ids},
                    "is_processed": {"equals": False},
                },
                order={"created_at": "desc"},
            )

            responses: list[DendriteQueryResponse] = []
            for validator_request in validator_requests:
                vali_request = map_feedback_request_model_to_feedback_request(
                    validator_request
                )

                m_responses = list(
                    map(
                        lambda x: map_feedback_request_model_to_feedback_request(
                            x, is_miner=True
                        ),
                        [
                            m
                            for m in miner_responses
                            if m.parent_id == validator_request.id
                        ],
                    )
                )

                responses.append(
                    DendriteQueryResponse(
                        request=vali_request, miner_responses=m_responses
                    )
                )

            # yield responses, so caller can do something
            has_more_batches = True
            yield responses, has_more_batches

        yield [], False

    @staticmethod
    async def get_real_model_ids(request_id: str) -> dict[str, str]:
        """Fetches a mapping of obfuscated model IDs to real model IDs for a given request ID."""
        ground_truths = await Ground_Truth_Model.prisma().find_many(
            where={"request_id": request_id}
        )
        return {gt.obfuscated_model_id: gt.real_model_id for gt in ground_truths}

    @staticmethod
    async def mark_tasks_processed_by_request_ids(request_ids: list[str]) -> None:
        """Mark records associated with validator's request and miner's responses as processed.

        Args:
            request_ids (list[str]): List of request ids.
        """
        if not request_ids:
            logger.error("No request ids provided to mark as processed")
            return

        try:
            async with transaction() as tx:
                num_updated = await tx.feedback_request_model.update_many(
                    data={"is_processed": True},
                    where={"request_id": {"in": request_ids}},
                )
                logger.success(
                    f"Marked {num_updated} records associated to {len(request_ids)} tasks as processed"
                )
        except PrismaError as exc:
            logger.error(f"Prisma error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

    @staticmethod
    async def get_task_by_request_id(request_id: str) -> DendriteQueryResponse | None:
        try:
            # find the parent id first
            include_query = Feedback_Request_ModelInclude(
                {
                    "completions": True,
                    "criteria_types": True,
                    "ground_truths": True,
                    "parent_request": True,
                    "child_requests": True,
                }
            )
            all_requests = await Feedback_Request_Model.prisma().find_many(
                where={
                    "request_id": request_id,
                },
                include=include_query,
            )

            validator_requests = [r for r in all_requests if r.parent_id is None]
            assert len(validator_requests) == 1, "Expected only one validator request"
            validator_request = validator_requests[0]
            if not validator_request.child_requests:
                raise InvalidTask(
                    f"Validator request {validator_request.id} must have child requests"
                )

            miner_responses = [
                map_feedback_request_model_to_feedback_request(r, is_miner=True)
                for r in validator_request.child_requests
            ]
            return DendriteQueryResponse(
                request=map_feedback_request_model_to_feedback_request(
                    model=validator_request, is_miner=False
                ),
                miner_responses=miner_responses,
            )

        except Exception as e:
            logger.error(f"Failed to get feedback request by request_id: {e}")
            return None

    @staticmethod
    async def get_num_processed_tasks() -> int:
        return await Feedback_Request_Model.prisma().count(
            where={"is_processed": True, "parent_id": None}
        )

    @staticmethod
    async def update_miner_completions_by_request_id(
        request_id: str, miner_responses: List[FeedbackRequest], batch_size: int = 10
    ) -> tuple[bool, list[int]]:
        """Update the miner's provided rank_id / scores etc. for a given request id that it is responding to validator. This exists because over the course of a task, a miner may recruit multiple workers and we
        need to recalculate the average score / rank_id etc. across all workers.
        """
        if not len(miner_responses):
            logger.debug("Updating completion responses: nothing to update, skipping.")
            return True, []

        num_batches = math.ceil(len(miner_responses) / batch_size)
        num_failed_batches = 0
        failed_batch_indices = []
        batch_id = 0
        for i in range(0, len(miner_responses), batch_size):
            safe_lim = min(len(miner_responses), i + batch_size)
            try:
                batch_responses = miner_responses[i:safe_lim]

                # NOTE: @dev we must nest the transaction so after __aexit__ is
                # called (when no exceptions occur) then tx.commit() is called
                async with prisma.tx(timeout=timedelta(seconds=10)) as tx:
                    # find the feedback request ids
                    miner_hotkeys = []
                    for miner_response in batch_responses:
                        if not miner_response.axon or not miner_response.axon.hotkey:
                            raise InvalidMinerResponse(
                                f"Miner response {miner_response} must have a hotkey"
                            )
                        miner_hotkeys.append(miner_response.axon.hotkey)

                    # update the completion_responses data
                    for miner_response in batch_responses:
                        # find the particular request
                        hotkey = miner_response.axon.hotkey  # type: ignore
                        curr_miner_response = (
                            await tx.feedback_request_model.find_first(
                                where=Feedback_Request_ModelWhereInput(
                                    request_id=request_id,
                                    hotkey=hotkey,  # type: ignore
                                )
                            )
                        )

                        if not curr_miner_response:
                            raise ValueError("Miner response not found")

                        # the actual completion_ids NOT the ids of the records in DB
                        completion_ids = [
                            c.completion_id for c in miner_response.completion_responses
                        ]

                        # find completion ids of the miner response
                        completion_records = (
                            await tx.completion_response_model.find_many(
                                where=Completion_Response_ModelWhereInput(
                                    feedback_request_id=curr_miner_response.id,
                                    completion_id={"in": completion_ids},
                                )
                            )
                        )

                        # actual completion_id TO id of record in DB
                        completion_id_record_id = {
                            c.completion_id: c.id for c in completion_records
                        }

                        # update the completions
                        for completion in miner_response.completion_responses:
                            await tx.completion_response_model.update(
                                data={
                                    "score": completion.score,
                                    "rank_id": completion.rank_id,
                                },
                                where=Completion_Response_ModelWhereUniqueInput(
                                    id=completion_id_record_id[
                                        completion.completion_id
                                    ],
                                ),
                            )

                logger.debug(
                    f"Updating completion responses: updated batch {batch_id+1}/{num_batches} "
                )
            except Exception as e:
                logger.error(
                    f"Updating completion responses: failed for batch {batch_id+1}/{num_batches}, error: {e}"
                )
                num_failed_batches += 1
                failed_batch_indices.extend(range(i, safe_lim))
            finally:
                batch_id += 1

            await asyncio.sleep(0.1)

        if num_failed_batches == 0:
            logger.success(
                f"Updating completion responses: successfully in {num_batches} batches for {len(miner_responses)} responses"
            )
            gc.collect()
            return True, []

        return False, failed_batch_indices

    @staticmethod
    async def save_task(
        validator_request: FeedbackRequest,
        miner_responses: List[FeedbackRequest],
        ground_truth: dict[str, int],
    ) -> Feedback_Request_Model | None:
        """Saves a task, which consists of both the validator's request and the miners' responses.

        Args:
            validator_request (FeedbackRequest): The request made by the validator.
            miner_responses (List[FeedbackRequest]): The responses made by the miners.
            ground_truth (dict[str, str]): The ground truth for the task, where dict

        Returns:
            Feedback_Request_Model | None: Only validator's feedback request model, or None if failed.
        """
        try:
            feedback_request_model: Feedback_Request_Model | None = None
            async with transaction() as tx:
                logger.trace("Starting transaction for saving task.")

                feedback_request_model = await tx.feedback_request_model.create(
                    data=map_parent_feedback_request_to_model(validator_request)
                )

                # Create related criteria types
                criteria_create_input = [
                    map_criteria_type_to_model(criteria, feedback_request_model.id)
                    for criteria in validator_request.criteria_types
                ]
                await tx.criteria_type_model.create_many(criteria_create_input)

                # Create related miner responses (child) and their completion responses
                created_miner_models: list[Feedback_Request_Model] = []
                for miner_response in miner_responses:
                    try:
                        create_miner_model_input = map_child_feedback_request_to_model(
                            miner_response,
                            feedback_request_model.id,
                            expire_at=feedback_request_model.expire_at,
                        )

                        created_miner_model = await tx.feedback_request_model.create(
                            data=create_miner_model_input
                        )

                        created_miner_models.append(created_miner_model)

                        criteria_create_input = [
                            map_criteria_type_to_model(criteria, created_miner_model.id)
                            for criteria in miner_response.criteria_types
                        ]
                        await tx.criteria_type_model.create_many(criteria_create_input)

                        # Create related completions for miner responses
                        for completion in miner_response.completion_responses:
                            # remove the completion field, since the miner receives an obfuscated completion_response anyways
                            # therefore it is useless for training
                            try:
                                completion_copy = completion.model_dump()
                                completion_copy["completion"] = CodeAnswer(files=[])
                            except KeyError:
                                pass
                            completion_input = map_completion_response_to_model(
                                CompletionResponses.model_validate(completion_copy),
                                created_miner_model.id,
                            )
                            await tx.completion_response_model.create(
                                data=completion_input
                            )
                            logger.trace(
                                f"Created completion response: {completion_input}"
                            )

                    # we catch exceptions here because whether a miner responds well should not affect other miners
                    except InvalidMinerResponse as e:
                        miner_hotkey = (
                            miner_response.axon.hotkey if miner_response.axon else "??"
                        )
                        logger.debug(
                            f"Miner response from hotkey: {miner_hotkey} is invalid: {e}"
                        )
                    except InvalidCompletion as e:
                        miner_hotkey = (
                            miner_response.axon.hotkey if miner_response.axon else "??"
                        )
                        logger.debug(
                            f"Completion response from hotkey: {miner_hotkey} is invalid: {e}"
                        )

                if len(created_miner_models) == 0:
                    raise InvalidTask(
                        "A task must consist of at least one miner response, along with validator's request"
                    )

                # this is dependent on how we obfuscate in `validator.send_request`
                for completion_id, rank_id in ground_truth.items():
                    gt_create_input = {
                        "rank_id": rank_id,
                        "obfuscated_model_id": completion_id,
                        "request_id": validator_request.request_id,
                        "real_model_id": completion_id,
                        "feedback_request_id": feedback_request_model.id,
                    }
                    await tx.ground_truth_model.create(
                        data=Ground_Truth_ModelCreateInput(**gt_create_input)
                    )
                for vali_completion in validator_request.completion_responses:
                    vali_completion_input = map_completion_response_to_model(
                        vali_completion,
                        feedback_request_model.id,
                    )
                    await tx.completion_response_model.create(
                        data=vali_completion_input
                    )

                feedback_request_model.child_requests = created_miner_models
            return feedback_request_model
        except Exception as e:
            logger.error(f"Failed to save dendrite query response: {e}")
            return None

    @staticmethod
    async def create_or_update_validator_score(scores: torch.Tensor) -> None:
        # Save scores as a single record
        score_model = await Score_Model.prisma().find_first()
        scores_list = scores.tolist()
        if score_model:
            await Score_Model.prisma().update(
                where={"id": score_model.id},
                data=Score_ModelUpdateInput(score=Json(json.dumps(scores_list))),
            )
        else:
            await Score_Model.prisma().create(
                data=Score_ModelCreateInput(
                    score=Json(json.dumps(scores_list)),
                )
            )

    @staticmethod
    async def get_validator_score() -> torch.Tensor | None:
        score_record = await Score_Model.prisma().find_first(
            order={"created_at": "desc"}
        )
        if not score_record:
            return None

        return torch.tensor(json.loads(score_record.score))

    @staticmethod
    async def get_scores_and_ground_truth_by_dojo_task_id(
        dojo_task_id: str,
    ) -> dict[str, dict[str, float | int | None]]:
        """
        Fetch the scores, model IDs from Completion_Response_Model for a given Dojo task ID.
        Also fetches rank IDs from Ground_Truth_Model for the given Dojo task ID.

        Args:
            dojo_task_id (str): The Dojo task ID to search for.

        Returns:
            dict[str, dict[str, float | int | None]]: A dictionary mapping model ID to a dict containing score and rank_id.
        """
        try:
            # First, find the Feedback_Request_Model with the given dojo_task_id
            feedback_request = await Feedback_Request_Model.prisma().find_first(
                where=Feedback_Request_ModelWhereInput(dojo_task_id=dojo_task_id),
                include={
                    "completions": True,
                    "parent_request": {"include": {"ground_truths": True}},
                },
            )

            if not feedback_request:
                logger.warning(
                    f"No Feedback_Request_Model found for dojo_task_id: {dojo_task_id}"
                )
                return {}

            parent_request = feedback_request.parent_request
            if not parent_request:
                logger.warning(
                    f"No parent request found for dojo_task_id: {dojo_task_id}"
                )
                return {}

            rank_id_map = {
                gt.obfuscated_model_id: gt.rank_id
                for gt in parent_request.ground_truths
            }

            # Extract scores from the completions
            scores_and_gts = {
                completion.model: {
                    "score": completion.score,
                    "ground_truth_rank_id": rank_id_map.get(completion.completion_id),
                }
                for completion in feedback_request.completions
            }

            logger.debug(
                f"Found {len(scores_and_gts)} scores and ground truths for dojo_task_id: {dojo_task_id}"
            )
            return scores_and_gts

        except Exception as e:
            logger.error(
                f"Error fetching completion scores and ground truths for dojo_task_id {dojo_task_id}: {e}"
            )
            return {}
