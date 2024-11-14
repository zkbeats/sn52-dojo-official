import asyncio
import json
import os
import random
from typing import Dict, List

import httpx
from bittensor.btlogging import logging as logger

import dojo
from commons.exceptions import CreateTaskFailed
from commons.utils import loaddotenv, set_expire_time
from dojo import get_dojo_api_base_url
from dojo.protocol import FeedbackRequest, MultiScoreCriteria, RankingCriteria

DOJO_API_BASE_URL = get_dojo_api_base_url()
# to be able to get the curlify requests
# DEBUG = False


def _get_max_results_param() -> int:
    max_results = os.getenv("TASK_MAX_RESULTS")
    if not max_results:
        logger.warning("TASK_MAX_RESULTS is not set, defaulting to 1")
        max_results = 1
    return int(max_results)


class DojoAPI:
    _http_client = httpx.AsyncClient()

    @classmethod
    async def _get_task_by_id(cls, task_id: str):
        """Gets task by task id and checks completion status"""
        url = f"{DOJO_API_BASE_URL}/api/v1/tasks/{task_id}"
        response = await cls._http_client.get(url)
        response.raise_for_status()
        return response.json()

    @classmethod
    async def _get_task_results_by_task_id(cls, task_id: str):
        """Gets task results from task id"""
        url = f"{DOJO_API_BASE_URL}/api/v1/tasks/task-result/{task_id}"
        response = await cls._http_client.get(url)
        response.raise_for_status()
        return response.json()

    @classmethod
    async def get_task_results_by_task_id(cls, task_id: str) -> List[Dict] | None:
        """Gets task results from task id to prepare for scoring later on"""
        # task_response = await cls._get_task_by_id(task_id)
        # task_status = task_response.get("body", {}).get("status", None)
        # is_completed = task_status and task_status.lower() == "completed"
        # if is_completed is None:
        #     logger.error(f"Failed to read status field for task_id: {task_id}")
        #     return

        # if is_completed is False:
        #     return

        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                task_results_response = await cls._get_task_results_by_task_id(task_id)
                task_results = task_results_response.get("body", {}).get("taskResults")
                if task_results is None or not task_results:
                    return None
                return task_results
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * 2**attempt + random.uniform(0, 1)
                    logger.warning(
                        f"Error occurred while getting task results for task_id {task_id}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to get task results for task_id {task_id} after {max_retries} attempts: {e}"
                    )
                    return None

        logger.error(
            f"Failed to get task results for task_id {task_id} after {max_retries} retries"
        )
        return None

    @staticmethod
    def serialize_feedback_request(data: FeedbackRequest):
        output = dict(
            prompt=data.prompt,
            responses=[],
            task=str(data.task_type).upper(),
            criteria=[],
        )
        for c in data.completion_responses:
            completion_dict = {}
            completion_dict["model"] = c.model
            if isinstance(c.completion, list):  # handling the case for DIALOGUE
                completion_dict["completion"] = [i.model_dump() for i in c.completion]
            elif isinstance(c.completion, str):  # handling the case for TEXT_TO_IMAGE
                completion_dict["completion"] = c.completion
            else:  # if not DIALOGUE or TEXT_TO_IMAGE, then it is CODE_GENERATION
                completion_dict["completion"] = c.completion.model_dump()

            output["responses"].append(completion_dict)

        return output

    @classmethod
    async def create_task(
        cls,
        feedback_request: FeedbackRequest,
    ):
        response_text = ""
        response_json = {}
        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                path = f"{DOJO_API_BASE_URL}/api/v1/tasks/create-tasks"
                taskData = cls.serialize_feedback_request(feedback_request)
                for criteria_type in feedback_request.criteria_types:
                    if isinstance(criteria_type, RankingCriteria) or isinstance(
                        criteria_type, MultiScoreCriteria
                    ):
                        taskData["criteria"].append(
                            {
                                **criteria_type.model_dump(),
                                "options": [
                                    option
                                    for option in criteria_type.model_dump().get(
                                        "options", []
                                    )
                                ],
                            }
                        )
                    else:
                        logger.error(
                            f"Unrecognized criteria type: {type(criteria_type)}"
                        )

                expire_at = set_expire_time(dojo.TASK_DEADLINE)

                max_results = _get_max_results_param()
                form_body = {
                    "title": ("", "LLM Code Generation Task"),
                    "body": ("", feedback_request.prompt),
                    "expireAt": ("", expire_at),
                    "taskData": ("", json.dumps([taskData])),
                    "maxResults": ("", str(max_results)),
                }

                payload_size = sum(len(str(v[1])) for v in form_body.values())
                logger.info(f"Payload size: {payload_size} bytes")

                DOJO_API_KEY = loaddotenv("DOJO_API_KEY")

                response = await cls._http_client.post(
                    path,
                    files=form_body,
                    headers={
                        "x-api-key": DOJO_API_KEY,
                    },
                    timeout=15.0,
                )

                response_text = response.text
                response_json = response.json()

                task_ids = []
                if response.status_code == 200:
                    task_ids = response.json()["body"]
                    logger.success(
                        f"Successfully created task with\ntask ids:{task_ids}"
                    )
                else:
                    logger.error(
                        f"Error occurred when trying to create task\nErr:{response.json()['error']}"
                    )
                response.raise_for_status()
                return task_ids
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * 2**attempt + random.uniform(0, 1)
                    logger.warning(
                        f"Error occurred: {e}. Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Error occurred after {max_retries} attempts: {e}")
                    if isinstance(e, httpx.HTTPStatusError | httpx.RequestError):
                        raise CreateTaskFailed(
                            f"Failed to create task after {max_retries} attempts due to HTTP error: {e}"
                        )
                    elif isinstance(e, json.JSONDecodeError):
                        raise CreateTaskFailed(
                            f"Failed to create task due to JSON decode error: {e}, response_text: {response_text}"
                        )
                    else:
                        raise CreateTaskFailed(
                            f"Failed to create task due to unexpected error: {e}, response_text: {response_text}, response_json: {response_json}"
                        )
        raise CreateTaskFailed(f"Failed to create task after {max_retries} retries")
