import datetime
import json
from typing import Dict, List

import httpx
from loguru import logger

import template
from commons.utils import loaddotenv
from template import DOJO_API_BASE_URL
from template.protocol import (
    FeedbackRequest,
    MultiScoreCriteria,
    RankingCriteria,
)

# Set to True to enable debug mode
# TODO could possible setup with logger to enable debug mode
DEBUG = False


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
        task_response = await cls._get_task_by_id(task_id)
        task_status = task_response.get("body", {}).get("status", None)
        is_completed = task_status and task_status.lower() == "completed"
        if is_completed is None:
            logger.error(f"Failed to read status field for task_id: {task_id}")
            return

        if is_completed is False:
            return
        task_results_response = await cls._get_task_results_by_task_id(task_id)
        task_results = task_results_response.get("body", {}).get("taskResults")
        if task_results is None:
            logger.error(f"Failed to read task results for task_id: {task_id}")
            return

        if not task_results:
            return

        return task_results

    @classmethod
    async def create_task(
        cls,
        ranking_request: FeedbackRequest,
    ):
        path = f"{DOJO_API_BASE_URL}/api/v1/tasks/create-tasks"
        taskData = {
            "prompt": ranking_request.prompt,
            "responses": [
                {"model": c.model, "completion": c.completion.model_dump()}
                for c in ranking_request.responses
            ],
            "task": str(ranking_request.task_type).upper(),
            "criteria": [],
        }
        for criteria_type in ranking_request.criteria_types:
            if isinstance(criteria_type, RankingCriteria) or isinstance(
                criteria_type, MultiScoreCriteria
            ):
                taskData["criteria"].append(
                    {
                        **criteria_type.model_dump(),
                        "options": [
                            option
                            for option in criteria_type.model_dump().get("options", [])
                        ],
                    }
                )
            else:
                logger.error(f"Unrecognized criteria type: {type(criteria_type)}")

        json_body = {
            "title": ("", "LLM Code Generation Task"),
            "body": ("", ranking_request.prompt),
            "expireAt": (
                "",
                (
                    datetime.datetime.utcnow()
                    + datetime.timedelta(seconds=template.TASK_DEADLINE)
                )
                .replace(microsecond=0, tzinfo=datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            ),
            "taskData": ("", json.dumps([taskData])),
            "maxResults": ("", "1"),
        }

        DOJO_API_KEY = loaddotenv("DOJO_API_KEY")

        response = await cls._http_client.post(
            path,
            files=json_body,
            headers={
                "x-api-key": DOJO_API_KEY,
            },
            timeout=15.0,
        )

        if DEBUG is True:
            try:
                from curlify2 import Curlify

                curl_req = Curlify(response.request)
                print("CURL REQUEST >>> ")
                print(curl_req.to_curl())
            except ImportError:
                print("Curlify not installed")
            except Exception as e:
                print("Tried to export create task request as curl, but failed.")
                print(f"Exception: {e}")

        task_ids = []
        if response.status_code == 200:
            task_ids = response.json()["body"]
            logger.success(f"Successfully created task with\ntask ids:{task_ids}")
        else:
            logger.error(
                f"Error occurred when trying to create task\nErr:{response.json()['error']}"
            )
        response.raise_for_status()
        return task_ids
