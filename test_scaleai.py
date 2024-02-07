import asyncio
from distutils.command import build
from enum import unique
import os
import time
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import requests

import scaleapi
from dotenv import load_dotenv
from scaleapi.exceptions import ScaleDuplicateResource
from scaleapi.tasks import TaskType
from commons.llm.openai_proxy import Provider
from commons.objects import ScoreItem, ScoresResponse

from commons.reward_model.models import ModelUtils
from template.protocol import Completion

load_dotenv()

import json

scale_api_key = os.getenv("SCALE_API_KEY")
client = scaleapi.ScaleClient(api_key=scale_api_key)

url = "https://api.scale.com/v1/projects"


def create_project(project_name: str) -> Optional[bool]:
    """Returns True or False if project was successfully created, and None when project already exists."""
    if project_name in [p.name for p in client.get_projects()]:
        print(f"Project {project_name} already exists... skipping creation")
        return None
    project_payload = {
        "task_type": TaskType.TextCollection,
        "project_name": project_name,
        "rapid": False,
        "studio": True,
        "pipeline": "standard_task",
        "params": {
            "instruction": "# Instructions\n\nScore each prompt and completions, where 1 is the lowest score and 10 is the highest score.",
            # # specify this so for studio projects
            "fields": [
                {
                    "type": "number",
                    "field_id": "Quality Score",
                    "title": "Quality Score",
                    "required": True,
                    "use_slider": True,
                    "min": 1,
                    "max": 10,
                    "prefix": "lowest quality",
                    "suffix": "highest quality",
                    # prompt that human labellers will see
                    "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
                    # "hint": None,
                },
            ],
        },
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic bGl2ZV84YjIxOWMzNDM5MmI0NjVlYTQwZDU1MzQ3ODNjYjVmZTo=",
    }
    try:
        response = requests.post(
            url, headers=headers, json=jsonable_encoder(project_payload)
        )
        print(response.status_code)
        print(response.json())
        # project = client.create_project(**project_payload)
    except ScaleDuplicateResource as e:
        # for some reason will get duplicate resource error although project was created
        pass
    is_project_created = project_name in [p.name for p in client.get_projects()]
    print(f"Successfully created project? {is_project_created}")
    return is_project_created


def _ensure_project_setup_completed(project_name: str):
    project = client.get_project(project_name)
    print(f"Before: {project=}")
    payload = {
        "patch": True,
        "pipelineName": "standard_task",
        "numReviews": 0,
        # "params": {
        #     "pipeline": "standard_task",
        #     "fields": [
        #         {
        #             "type": "number",
        #             "field_id": "Quality Score",
        #             "title": "Quality Score",
        #             "required": True,
        #             "use_slider": True,
        #             "min": 1,
        #             "max": 10,
        #             "prefix": "lowest quality",
        #             "suffix": "highest quality",
        #             "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
        #         },
        #     ],
        # },
    }
    client.update_project(project_name, **payload)

    project = client.get_project(project_name)
    print(f"After: {project=}")


def dedupe_dataset():
    with open("./seed_datasets/dataset_sn18_duplicated.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        seen_responses = set()
        deduplicated_data = []
        # each row represents a row of prompt to completions
        for d in data:
            responses = d["responses"]
            unique_responses = []
            for response_obj in responses:
                response = response_obj["response"]
                if response not in seen_responses:
                    seen_responses.add(response)
                    unique_responses.append(response_obj)
            deduplicated_data.append(
                {
                    "prompt": d["prompt"],
                    "responses": unique_responses,
                    "num_completions": len(unique_responses),
                }
            )

    # save list of dict to new jsonl file
    with open("./seed_datasets/dataset_sn18_deduplicated.jsonl", "w") as f_out:
        for item in deduplicated_data:
            f_out.write(json.dumps(item) + "\n")

    return deduplicated_data


def build_batch_name(project_name: str, is_calibration) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    # TODO remove after testing
    batch_name = f"{project_name}_{current_date}_2"
    if not is_calibration:
        return batch_name
    return batch_name + "_calibration"


def create_task(project_name: str):
    batch = None
    batch_name = build_batch_name(project_name, False)
    try:
        batch = client.create_batch(project=project_name, batch_name=batch_name)
    except ScaleDuplicateResource as e:
        print("Batch already exists... skipping creation")
        batch = client.get_batch(batch_name)
        pass

    print(f"Batch json: {batch._json}")

    # # convert  completions into attachments for human to view
    # def format_prompt_and_completion(prompt: str, completion: Completion) -> dict:
    #     return {
    #         "type": "text",
    #         "content": f"# Prompt: {prompt} # Completion: {completion.text}",
    #     }

    # attachments = [{"type": "text", "content": c.text} for c in completions]
    payload = {
        "instruction": "**Instructions:** Please annotate all the things",
        "attachments": [
            {
                "type": "text",
                "content": "# Prompt:\nwhat is your name?\n\n# Completion:\n my name is Alice, nice to meet you!",
            },
        ],
        "responses_required": 1,
        "priority": 30,
        "project": project_name,
        "batch": batch_name,
        "callback_url": "https://webhook.site/#!/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
        "title": "title",
        "description": "desc",
        "unique_id": str(uuid4()),
        "fields": [
            {
                "type": "number",
                "field_id": "Quality Score",
                "title": "Quality Score",
                "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
                "required": False,
                "use_slider": True,
                "min": 1,
                "max": 10,
                # "prefix": "lowest quality",
                # "suffix": "highest quality",
                # prompt that human labellers will see
            },
        ],
        "isProcessed": False,
    }

    task_payload = {
        "project": project_name,
        "batch": batch_name,
        "callback_url": "https://webhook.site/#!/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
        "attachments": [
            {
                "type": "text",
                "content": "# Prompt:what is your name?\n\n# Completion: my name is Alice.",
            },
        ],
        "description": "This project blah blah description",
        "responses_required": 1,
        "instruction": "# Instructions\n\nScore each prompt and completions, where 1 is the lowest score and 10 is the highest score.",
        "fields": [
            {
                "type": "number",
                "field_id": "Quality Score",
                "title": "Quality Score",
                "required": True,
                "use_slider": True,
                "min": 1,
                "max": 10,
                "prefix": "lowest quality",
                "suffix": "highest quality",
                # prompt that human labellers will see
                "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
                # "hint": None,
            },
        ],
        "priority": 30,
        # "params": {
        #     # "fields": [
        #     #     {
        #     #         "type": "number",
        #     #         "field_id": "Quality Score",
        #     #         "title": "Quality Score",
        #     #         "required": True,
        #     #         "use_slider": True,
        #     #         "min": 1,
        #     #         "max": 10,
        #     #         "prefix": "lowest quality",
        #     #         "suffix": "highest quality",
        #     #         # prompt that human labellers will see
        #     #         "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
        #     #         # "hint": None,
        #     #     },
        #     # ],
        #     # specify this so that we can use num_reviews =
        #     # "pipeline": "standard_task",
        # },
    }
    # task = client.create_evaluation_task(TaskType.TextCollection, **task_payload)
    # task = client.create_task(TaskType.TextCollection, **task_payload)
    task = client.create_task(TaskType.TextCollection, **payload)
    batch.finalize()


def create_eval_tasks(project_name):
    eval_data = []
    with open("./seed_datasets/dataset_sn18_deduplicated_scored.jsonl", "r") as f:
        for line in f:
            eval_data.append(json.loads(line))

    batch_name = build_batch_name(project_name, True)
    batch = None
    try:
        batch = client.create_batch(
            project=project_name,
            batch_name=batch_name,
            calibration_batch=True,
        )
        print("Created calibration batch...")
    except Exception as e:
        print(f"Error occured while creating calibration batch, exception: {str(e)}")
        print("Calibration batch already exists... skipping creation")
        pass

    if batch is None:
        batch = client.get_batch(batch_name)

    count = 0
    for row in eval_data:
        for response in row["responses"]:
            if count > 10:
                break

            attachments = [
                {
                    "type": "text",
                    "content": f"# Prompt: {row['prompt']} # Completion: {response['response']}",
                }
            ]
            task_payload = {
                "project": project_name,
                "batch": batch_name,
                "callback_url": "https://webhook.site/#!/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
                "attachments": attachments,
                "expected_response": {
                    "Quality Score": {
                        "type": "number",
                        "field_id": "Quality Score",
                        "response": 9.5,
                    },
                },
                "fields": [
                    {
                        "type": "number",
                        "field_id": "Quality Score",
                        "title": "Quality Score",
                        "required": True,
                        "use_slider": True,
                        "min": 1,
                        "max": 10,
                        "prefix": "lowest quality",
                        "suffix": "highest quality",
                        # prompt that human labellers will see
                        "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
                        # "hint": None,
                    },
                ],
            }
            task = client.create_evaluation_task(
                TaskType.TextCollection, **task_payload
            )
            print(f"Created task... for {batch_name}")
            count += 1
    # finalised_res = batch.finalize()
    # print(f"Finalised batch, response: {finalised_res}")


async def add_scores_to_dataset():
    scored_data = []
    with open("./seed_datasets/dataset_sn18_deduplicated_scored.jsonl", "r") as f:
        all_data = [json.loads(line) for line in f]
        for row_data in all_data:
            completions = [
                Completion(text=r["response"]) for r in row_data["responses"]
            ]
            scores_response: ScoresResponse = await ModelUtils._llm_api_score(
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                provider=Provider.TOGETHER_AI,
                completions=completions,
                prompt=row_data["prompt"],
            )

            for r in row_data["responses"]:
                response = r["response"]
                found_completion: Completion = next(
                    (c for c in completions if c.text == response), None
                )
                if found_completion is None:
                    continue
                # find matching cid in response
                found_score_item: Optional[ScoreItem] = next(
                    (
                        sr
                        for sr in scores_response.scores
                        if sr.completion_id == found_completion.cid
                    ),
                    None,
                )
                if found_score_item is None:
                    continue
                # scale it back to 1 to 10
                score = found_score_item.score * 10
                r["score"] = float(score)
            scored_data.append(row_data)
        # create a new jsonl file with scored data
        with open("./seed_datasets/scaleai_eval_dataset.jsonl", "w") as f_out:
            for item in scored_data:
                f_out.write(json.dumps(item) + "\n")


def one_time_finalise(project_name):
    batch = None
    batch_name = build_batch_name(project_name, False)
    try:
        batch = client.create_batch(project=project_name, batch_name=batch_name)
    except ScaleDuplicateResource as e:
        print("Batch already exists... skipping creation")
        batch = client.get_batch(batch_name)

    assert batch is not None
    finalised_res = batch.finalize()
    print(f"batch finalised res: {finalised_res}")


if __name__ == "__main__":
    project_name = "human_feedback23"
    # print(client.get_projects())
    create_project(project_name)
    # _ensure_project_setup_completed(project_name)
    # dedupe_dataset()
    # asyncio.run(add_scores_to_dataset())
    # create_eval_tasks(project_name)
    # create_task(project_name)

    # one_time_finalise(project_name)
    # print(client.get_projects())

    # print(client.get_task(task_id="65c23c07451f97ec7876c4ca"))
