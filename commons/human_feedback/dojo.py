import asyncio
import datetime
import os
from typing import Dict, List
from fastapi.encoders import jsonable_encoder
import httpx
from strenum import StrEnum

from template.protocol import FeedbackRequest, TaskType, CriteriaType
from dotenv import load_dotenv

load_dotenv()


# TODO @dev change to URL before launch
DOJO_API_BASE_URL = os.getenv("DOJO_API_BASE_URL")
if not DOJO_API_BASE_URL:
    raise ValueError("DOJO_API_BASE_URL is not set")


def get_dojo_api_key():
    key = os.getenv("DOJO_API_KEY")
    if key is None:
        raise ValueError("DOJO_API_KEY is not set")
    return key


def _extract_ranking_result_data(result_data: List[Dict]) -> Dict[str, str]:
    # sample data: [{'type': 'ranking', 'value': {'1': 'Code 2', '2': 'Code 1'}]
    ranking_results = list(
        filter(lambda x: x["type"] == CriteriaType.PREFERENCE_RANKING, result_data)
    )
    if len(ranking_results) == 1:
        return ranking_results[0].get("value", {})
    return {}


def parse_task_results(response_json: Dict) -> List[Dict[str, str]]:
    task_results = response_json.get("body", {}).get("taskResults", [])
    parsed_results = []
    for t in task_results:
        res = _extract_ranking_result_data(t.get("result_data", []))
        if not res:
            continue
        parsed_results.append(res)
    return parsed_results


def check_task_completion_status(response_json: Dict):
    status = response_json.get("body", {}).get("status")
    return status and status.lower() == "completed"


class DojoAPI:
    # _api_key = get_dojo_api_key()
    # _http_client = httpx.AsyncClient(headers={"Authorization": f"Bearer {_api_key}"})
    _http_client = httpx.AsyncClient()

    @classmethod
    async def get_task_by_id(cls, task_id: str):
        """Gets task by task id and checks completion status"""
        url = f"{DOJO_API_BASE_URL}/api/v1/tasks/{task_id}"
        async with cls._http_client as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @classmethod
    async def get_task_results_by_task_id(cls, task_id: str):
        """Gets ranking task results from task id"""
        url = f"{DOJO_API_BASE_URL}/api/v1/tasks/get-results/{task_id}"
        async with cls._http_client as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @classmethod
    async def get_task_and_results(cls, task_id: str):
        """Returns optional [{'1': 'model hash 1', '2': 'model hash 2'}],
        where '1' and '2' are the explicit rank integers"""
        completion_status = check_task_completion_status(
            await cls.get_task_by_id(task_id)
        )
        if not completion_status:
            return None
        ranking_results = parse_task_results(
            await cls.get_task_results_by_task_id(task_id)
        )
        if not ranking_results:
            return None
        return ranking_results

    @classmethod
    async def create_task(
        cls,
        ranking_request: FeedbackRequest,
    ):
        path = f"{DOJO_API_BASE_URL}/api/v1/tasks/create-task"
        async with cls._http_client as client:
            taskData = {
                "prompt": ranking_request.prompt,
                "responses": [
                    {"model": c.model, "completion": c.completion.dict()}
                    for c in ranking_request.responses
                ],
                "task": str(ranking_request.task_type).upper(),
                "criteria": [],
            }
            if CriteriaType.PREFERENCE_RANKING in ranking_request.criteria_types:
                taskData["criteria"].append(
                    {
                        "type": "ranking",
                        "options": [
                            f"Model {completion.model}"
                            for _, completion in enumerate(ranking_request.responses)
                        ],
                    }
                )
            body = {
                "title": "LLM Code Generation Task",
                "body": ranking_request.prompt,
                "expireAt": (datetime.datetime.utcnow() + datetime.timedelta(hours=24))
                .replace(microsecond=0)
                .isoformat(),
                "taskData": taskData,
                "maxResults": 10,
            }
            response = await client.post(path, json=jsonable_encoder(body))
            if response.status_code == 200:
                task_ids = response.json()["body"]
            response.raise_for_status()
            return task_ids


if __name__ == "__main__":

    async def main():
        print(
            await DojoAPI.get_task_results_by_task_id(
                "bdb56d72-dd98-40c0-a42f-312d018a0a1e"
            )
        )

    asyncio.run(main())
