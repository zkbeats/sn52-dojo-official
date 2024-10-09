import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

import torch
from bittensor.btlogging import logging as logger
from strenum import StrEnum

from database.client import transaction
from database.mappers import (
    map_completion_response_to_model,
    map_criteria_type_to_model,
    map_feedback_request_to_model,
    map_miner_response_to_model,
    map_model_to_dendrite_query_response,
)
from database.prisma._fields import Json
from database.prisma.models import (
    Feedback_Request_Model,
    Miner_Response_Model,
    Score_Model,
    Validator_State_Model,
)
from database.prisma.types import (
    Score_ModelCreateInput,
    Score_ModelUpdateInput,
    Validator_State_ModelCreateInput,
)
from template.protocol import (
    DendriteQueryResponse,
    FeedbackRequest,
    RidToHotKeyToTaskId,
    RidToModelMap,
    TaskExpiryDict,
)


class ValidatorStateKeys(StrEnum):
    SCORES = "scores"
    DOJO_TASKS_TO_TRACK = "dojo_tasks_to_track"
    MODEL_MAP = "model_map"
    TASK_TO_EXPIRY = "task_to_expiry"


class DataManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def load(cls) -> List[DendriteQueryResponse] | None:
        try:
            feedback_requests = await Feedback_Request_Model.prisma().find_many(
                include={
                    "criteria_types": True,
                    "miner_responses": {"include": {"completions": True}},
                }
            )

            logger.info(f"Loaded feedback requests: {len(feedback_requests)}")

            if not feedback_requests or len(feedback_requests) == 0:
                logger.error("No Feedback_Request_Model data found.")
                return None

            result = [
                map_model_to_dendrite_query_response(r) for r in feedback_requests
            ]

            return result

        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return None

    @classmethod
    async def save_dendrite_response(
        cls, response: DendriteQueryResponse
    ) -> Feedback_Request_Model | None:
        try:
            async with transaction() as tx:
                logger.info(
                    f"Saving dendrite query response for request_id: {response.request.request_id}"
                )
                # Create the main feedback request record
                feedback_request_model: Feedback_Request_Model = (
                    await tx.feedback_request_model.create(
                        data=map_feedback_request_to_model(response.request)
                    )
                )

                # Create related criteria types
                for criteria in response.request.criteria_types:
                    criteria_model = map_criteria_type_to_model(
                        criteria, feedback_request_model.request_id
                    )
                    await tx.criteria_type_model.create(data=criteria_model)

                miner_responses: list[Miner_Response_Model] = []
                # Create related miner responses and their completion responses
                for miner_response in response.miner_responses:
                    miner_response_data = map_miner_response_to_model(
                        miner_response,
                        feedback_request_model.request_id,  # Use feedback_request_model.request_id
                    )
                    if not miner_response_data.get("dojo_task_id"):
                        logger.error("Dojo task id is required")
                        raise ValueError("Dojo task id is required")

                    miner_response_model = await tx.miner_response_model.create(
                        data=miner_response_data
                    )

                    miner_responses.append(miner_response_model)

                    # Create related completions for miner responses
                    for completion in miner_response.completion_responses:
                        await tx.completion_response_model.create(
                            data=map_completion_response_to_model(
                                completion, miner_response_model.id
                            )
                        )
                feedback_request_model.miner_responses = miner_responses
                return feedback_request_model
        except Exception as e:
            logger.error(f"Failed to save dendrite query response: {e}")
            return None

    @classmethod
    async def overwrite_miner_responses_by_request_id(
        cls, request_id: str, miner_responses: List[FeedbackRequest]
    ) -> bool:
        try:
            # TODO can improve this
            async with transaction() as tx:
                # Delete existing completion responses for the given request_id
                await tx.completion_response_model.delete_many(
                    where={"miner_response": {"is": {"request_id": request_id}}}
                )

                # Delete existing miner responses for the given request_id
                await tx.miner_response_model.delete_many(
                    where={"request_id": request_id}
                )

                # Create new miner responses
                for miner_response in miner_responses:
                    miner_response_model = await tx.miner_response_model.create(
                        data=map_miner_response_to_model(miner_response, request_id)
                    )

                    # Create related completions for miner responses
                    for completion in miner_response.completion_responses:
                        await tx.completion_response_model.create(
                            data=map_completion_response_to_model(
                                completion, miner_response_model.id
                            )
                        )

                logger.success(
                    f"Overwritten miner responses for requestId: {request_id}"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to overwrite miner responses: {e}")
            return False

    @classmethod
    async def get_by_request_id(cls, request_id: str) -> DendriteQueryResponse | None:
        try:
            feedback_request = await Feedback_Request_Model.prisma().find_first(
                where={"request_id": request_id},
                include={
                    "criteria_types": True,
                    "miner_responses": {"include": {"completions": True}},
                },
            )
            if feedback_request:
                return map_model_to_dendrite_query_response(feedback_request)
            return None
        except Exception as e:
            logger.error(f"Failed to get feedback request by request_id: {e}")
            return None

    @classmethod
    async def remove_responses(cls, responses: List[DendriteQueryResponse]) -> bool:
        try:
            async with transaction() as tx:
                for response in responses:
                    request_id = response.request.request_id

                    # Delete completion responses associated with the miner responses
                    await tx.completion_response_model.delete_many(
                        where={"miner_response": {"is": {"request_id": request_id}}}
                    )

                    # Delete miner responses associated with the feedback request
                    await tx.miner_response_model.delete_many(
                        where={"request_id": request_id}
                    )

                    # Delete criteria types associated with the feedback request
                    await tx.criteria_type_model.delete_many(
                        where={"request_id": request_id}
                    )

                    # Delete the feedback request itself
                    await tx.feedback_request_model.delete_many(
                        where={"request_id": request_id}
                    )

                logger.success("Successfully removed specified responses.")
                return True
        except Exception as e:
            logger.error(f"Failed to remove responses: {e}")
            return False

    @classmethod
    async def validator_save(
        cls,
        scores: torch.Tensor,
        requestid_to_mhotkey_to_task_id: RidToHotKeyToTaskId,
        model_map: RidToModelMap,
        task_to_expiry: TaskExpiryDict,
    ):
        """Saves the state of the validator to the database."""
        logger.debug("Attempting to save validator state.")
        try:
            dojo_task_data = json.loads(json.dumps(requestid_to_mhotkey_to_task_id))
            if not dojo_task_data and torch.count_nonzero(scores).item() == 0:
                raise ValueError("Dojo task data and scores are empty. Skipping save.")

            logger.trace(f"Saving validator dojo_task_data: {dojo_task_data}")
            logger.trace(f"Saving validator score: {scores}")

            # Convert tensors to lists for JSON serialization
            scores_list = scores.tolist()

            # Prepare nested data for creating the validator state
            validator_state_data: list[Validator_State_ModelCreateInput] = [
                {
                    "request_id": request_id,
                    "miner_hotkey": miner_hotkey,
                    "task_id": task_id,
                    "expire_at": task_to_expiry[task_id],
                    "obfuscated_model": obfuscated_model,
                    "real_model": real_model,
                }
                for request_id, hotkey_to_task in dojo_task_data.items()
                for miner_hotkey, task_id in hotkey_to_task.items()
                for obfuscated_model, real_model in model_map[request_id].items()
            ]

            # Save the validator state
            await Validator_State_Model.prisma().create_many(
                data=validator_state_data, skip_duplicates=True
            )

            # Save scores as a single record
            score_model = await Score_Model.prisma().find_first()

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

            logger.success(
                f"📦 Saved validator state with scores: {scores}, and for {len(dojo_task_data)} requests"
            )
        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")

    @classmethod
    async def validator_load(cls) -> dict | None:
        try:
            # Query the latest validator state
            states: List[
                Validator_State_Model
            ] = await Validator_State_Model.prisma().find_many()

            if not states:
                logger.error("Validator state not found.")
                return None

            # Query the scores
            score_record = await Score_Model.prisma().find_first(
                order={"created_at": "desc"}
            )

            if not score_record:
                logger.error("Score record not found.")
                return None

            # Deserialize the data
            scores: torch.Tensor = torch.tensor(json.loads(score_record.score))

            # Initialize the dictionaries with the correct types and default factories
            dojo_tasks_to_track: RidToHotKeyToTaskId = defaultdict(
                lambda: defaultdict(str)
            )
            model_map: RidToModelMap = defaultdict(dict)
            task_to_expiry: TaskExpiryDict = defaultdict(str)

            for state in states:
                if (
                    state.request_id not in dojo_tasks_to_track
                ):  # might not need to check
                    dojo_tasks_to_track[state.request_id] = {}
                dojo_tasks_to_track[state.request_id][state.miner_hotkey] = (
                    state.task_id
                )

                if state.request_id not in model_map:
                    model_map[state.request_id] = {}
                model_map[state.request_id][state.obfuscated_model] = state.real_model

                task_to_expiry[state.task_id] = state.expire_at

            return {
                "scores": scores,
                "dojo_tasks_to_track": dojo_tasks_to_track,
                "model_map": model_map,
                "task_to_expiry": task_to_expiry,
            }

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    @staticmethod
    async def remove_expired_tasks_from_storage():
        try:
            # Load the current state
            state_data = await DataManager.validator_load()
            if not state_data:
                logger.error("Failed to load validator state data for cleanup.")
                return

            # Identify expired tasks
            current_time = datetime.now(timezone.utc)
            task_to_expiry = state_data.get(ValidatorStateKeys.TASK_TO_EXPIRY, {})
            expired_tasks = [
                task_id
                for task_id, expiry_time in task_to_expiry.items()
                if datetime.fromisoformat(expiry_time) < current_time
            ]

            # Remove expired tasks from the database
            for task_id in expired_tasks:
                await Validator_State_Model.prisma().delete_many(
                    where={"task_id": task_id}
                )

            # Update the in-memory state
            for task_id in expired_tasks:
                for request_id, hotkeys in list(
                    state_data[ValidatorStateKeys.DOJO_TASKS_TO_TRACK].items()
                ):
                    for hotkey, t_id in list(hotkeys.items()):
                        if t_id == task_id:
                            del state_data[ValidatorStateKeys.DOJO_TASKS_TO_TRACK][
                                request_id
                            ][hotkey]
                    if not state_data[ValidatorStateKeys.DOJO_TASKS_TO_TRACK][
                        request_id
                    ]:
                        del state_data[ValidatorStateKeys.DOJO_TASKS_TO_TRACK][
                            request_id
                        ]
                del task_to_expiry[task_id]

            # Save the updated state
            state_data[ValidatorStateKeys.TASK_TO_EXPIRY] = task_to_expiry
            await DataManager.validator_save(
                state_data[ValidatorStateKeys.SCORES],
                state_data[ValidatorStateKeys.DOJO_TASKS_TO_TRACK],
                state_data[ValidatorStateKeys.MODEL_MAP],
                task_to_expiry,
            )

            logger.info(f"Removed {len(expired_tasks)} expired tasks from DataManager.")
        except Exception as e:
            logger.error(f"Failed to remove expired tasks: {e}")
