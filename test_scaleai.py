import asyncio
from distutils.command import build
from enum import unique
import os
import time
from typing import List, Optional
from uuid import uuid4
from datetime import datetime

import scaleapi
from dotenv import load_dotenv
from scaleapi.exceptions import ScaleDuplicateResource, ScaleInvalidRequest
from scaleapi.tasks import TaskType
from commons.llm.openai_proxy import Provider
from commons.llm.prompts import ScoreRange
from commons.objects import ScoreItem, ScoresResponse

from commons.reward_model.models import ModelUtils
from template.protocol import Completion

load_dotenv()

import json

scale_api_key = os.getenv("SCALE_API_KEY")
client = scaleapi.ScaleClient(api_key=scale_api_key)

import textwrap


score_range = ScoreRange(lower=1, upper=10)


def _build_project_level_instruction(score_range):
    instruction = f"""
    # Instructions
    Score each prompt and completions, where {score_range.lower} is the lowest score and {score_range.upper} is the highest score.
    """
    return textwrap.dedent(instruction)


def _build_task_level_instruction(score_range):
    instruction = f"""
    # Instructions
    Score each prompt and completions, where {score_range.lower} is the lowest score and {score_range.upper} is the highest score.
    """
    return textwrap.dedent(instruction)


def _build_fields_dict(score_range):
    return (
        {
            "type": "number",
            "field_id": "Quality Score",
            "title": "Quality Score",
            "required": True,
            "use_slider": True,
            "min": score_range.lower,
            "max": score_range.upper,
            # text that labellers will see at "min" part of slider
            "prefix": "lowest quality",
            # text that labellers will see at "max" part of slider
            "suffix": "highest quality",
            # prompt above slider
            "description": f"Rate the completions quality compared to the prompt from {score_range.lower} (lowest quality) to {score_range.upper} (highest quality)",
            # "hint": None,
        },
    )


def create_project(project_name: str) -> Optional[bool]:
    """Returns True or False if project was successfully created, and None when project already exists."""
    if project_name in [p.name for p in client.get_projects()]:
        print(f"Project {project_name} already exists... skipping creation")
        return None
    project_payload = {
        "task_type": TaskType.TextCollection,
        "project_name": project_name,
        "studio": True,
        "params": {
            "instruction": _build_project_level_instruction(score_range),
            "fields": [_build_fields_dict(score_range)],
        },
    }
    try:
        _ = client.create_project(**project_payload)
    except ScaleDuplicateResource as e:
        # may get duplicate resource error although project was created properly...
        pass

    if is_project_created := project_name in [p.name for p in client.get_projects()]:
        batch_name = build_batch_name(project_name)
        try:
            # need to create first batch in order to "complete setup"
            _ = client.create_batch(project=project_name, batch_name=batch_name)
        except Exception as e:
            pass

    print(f"Successfully created project? {is_project_created}")
    return is_project_created


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


def build_batch_name(project_name: str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    batch_name = f"{project_name}_{current_date}"
    return batch_name


def create_task(project_name: str):
    batch = None
    batch_name = build_batch_name(project_name)
    try:
        batch = client.create_batch(project=project_name, batch_name=batch_name)
        print("Created batch successfully")
    except ScaleDuplicateResource as e:
        print("Batch already exists... skipping creation")
        batch = client.get_batch(batch_name)
        pass

    payload = {
        "instruction": _build_task_level_instruction(score_range),
        # TODO use multiple prompt / completion pairs
        "attachments": [
            {
                "type": "text",
                "content": "# Prompt:\nwhat is your name?\n\n# Completion:\n my name is Alice, nice to meet you!",
            },
            {
                "type": "text",
                "content": "# Prompt:\nwhat is your name?\n\n# Completion:\n my name is Bob",
            },
        ],
        "responses_required": 1,
        "priority": 30,
        "project": project_name,
        "batch": batch_name,
        # TODO
        "callback_url": "https://webhook.site/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
        "title": "title",
        "description": "desc",
        "unique_id": str(uuid4()),
        "fields": [_build_fields_dict(score_range)],
    }

    task = client.create_task(TaskType.TextCollection, **payload)
    try:
        batch.finalize()
    except ScaleInvalidRequest as e:
        print(f"Error occured while finalising batch, exception: {str(e)}")
        pass


def create_eval_tasks(project_name):
    eval_data = []
    with open("./seed_datasets/dataset_sn18_deduplicated_scored.jsonl", "r") as f:
        for line in f:
            eval_data.append(json.loads(line))

    batch_name = build_batch_name(project_name) + "_calibration"
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
                "callback_url": "https://webhook.site/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
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
    finalised_res = batch.finalize()
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
    batch_name = build_batch_name(project_name)
    try:
        batch = client.create_batch(project=project_name, batch_name=batch_name)
    except ScaleDuplicateResource as e:
        print("Batch already exists... skipping creation")
        batch = client.get_batch(batch_name)

    assert batch is not None
    finalised_res = batch.finalize()
    print(f"batch finalised res: {finalised_res}")


if __name__ == "__main__":
    project_name = "human_feedback_PLEASE_WORK"
    create_project(project_name)
    create_task(project_name)
    # dedupe_dataset()
    # asyncio.run(add_scores_to_dataset())
    # create_eval_tasks(project_name)

    # one_time_finalise(project_name)
    # print(client.get_projects())

    # print(client.get_task(task_id="65c23c07451f97ec7876c4ca"))
