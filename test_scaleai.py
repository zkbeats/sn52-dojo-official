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


def update_project(project_name: str) -> Optional[bool]:
    """Returns True or False if project was successfully created, and None when project already exists."""
    existing_projects = client.get_projects()
    if project_name in [p.name for p in existing_projects]:
        print("Project already exists...")
        return
    try:
        project_payload = {
            "task_type": TaskType.TextCollection,
            "project_name": project_name,
            "rapid": True,
            "studio": False,
            "params": {
                "instruction": "# Instructions\n\nScore each prompt and completions, where 1 is the lowest score and 10 is the highest score.",
                # specify this so that we can use num_reviews =
                "pipeline": "standard_task",
                # "fields": [
                #     {
                #         "type": "number",
                #         "field_id": "Quality Score",
                #         "title": "Quality Score",
                #         "required": True,
                #         "use_slider": True,
                #         "min": 1,
                #         "max": 10,
                #         "prefix": "lowest quality",
                #         "suffix": "highest quality",
                #         # prompt that human labellers will see
                #         "description": "Rate the completions quality compared to the prompt from 1 (lowest quality) to 10 (highest quality)",
                #         # "hint": None,
                #     },
                # ],
            },
        }
        project = client.create_project(**project_payload)
    except ScaleDuplicateResource as e:
        is_existing_projects = project_name in [p.name for p in client.get_projects()]
        print(f"Successfully created project? {is_existing_projects}")
        return is_existing_projects


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
    batch_name = f"{project_name}_{current_date}_2"
    if not is_calibration:
        return batch_name
    return batch_name + "_calibration"


def create_task(project_name: str):
    batch_name = build_batch_name(project_name, False)
    try:
        _ = client.create_batch(project=project_name, batch_name=batch_name)
    except ScaleDuplicateResource as e:
        print("Batch already exists... skipping creation")
        pass

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


def create_eval_tasks(project_name):
    eval_data = []
    with open("./seed_datasets/dataset_sn18_deduplicated_scored.jsonl", "r") as f:
        for line in f:
            eval_data.append(json.loads(line))

    batch_name = build_batch_name(project_name, True)
    try:
        batch = client.create_batch(
            project=project_name,
            batch_name=batch_name,
            calibration_batch=True,
        )
        print("Created calibration batch...")
    except:
        print("Calibration batch already exists... skipping creation")
        pass

    for row in eval_data:
        for response in row["responses"]:
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
                    "annotations": {
                        "Quality Score": {
                            "type": "number",
                            "field_id": "Quality Score",
                            "response": [9.5],
                        },
                    }
                },
                "initial_response": {
                    "annotations": {
                        "Quality Score": {
                            "type": "number",
                            "field_id": "Quality Score",
                            "response": [5.0],
                        },
                    }
                },
            }
            task = client.create_evaluation_task(
                TaskType.TextCollection, **task_payload
            )


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


if __name__ == "__main__":
    project_name = "human_feedback9"
    update_project(project_name)
    # dedupe_dataset()
    # asyncio.run(add_scores_to_dataset())
    create_task(project_name)
    # print(client.get_projects())
    # create_eval_tasks(project_name)

    # print(client.get_task(task_id="65c23c07451f97ec7876c4ca"))


######################### CREATE PROJECT TASKS
from scaleapi.tasks import TaskType
import json


# # try:
# #     client.create_task(TaskType.TextCollection, **payload)
# #     client.create_project()
# # except ScaleDuplicateResource as err:
# #     print(err.message)  # If unique_id is already used for a different task

######################### CREATE PROJECT
# import scaleapi
# from scaleapi.tasks import TaskType
# client = scaleapi.ScaleClient(scale_api_key)
# project = client.create_project(
#     project_name="Project_Param_Example",
#     task_type=TaskType.ImageAnnotation,
#     params={
#         "instruction": "Please label the kittens",
#         "geometries": {"box": {"objects_to_annotate": ["Kittens", "Puppies"]}},
#         "annotation_attributes": {
#             "happiness_rating": {
#                 "type": "category",
#                 "description": "How happy is this animal?",
#                 "choices": [
#                     "Jumping for Joy",
#                     "Wagging Tail",
#                     "Content",
#                     "Less than happy",
#                 ],
#                 "allow_multiple": True,
#             }
#         },
#     },
# )


# uuid = uuid4()
# # payload = dict(
# #     project=f"Human Annotation Project {uuid}",
# #     callback_url="https://webhook.site/#!/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
# #     instruction="Draw a box around each baby cow and big cow.",
# #     attachment_type="image",
# #     attachment="http://i.imgur.com/v4cBreD.jpg",
# #     unique_id="c235d023af73",
# #     geometries={
# #         "box": {
# #             "objects_to_annotate": ["Baby Cow", "Big Cow"],
# #             "min_height": 10,
# #             "min_width": 10,
# #         }
# #     },
# # )
# payload = {
#     "project": f"Human Annotation Project {uuid}",
#     "callback_url": "https://webhook.site/#!/71d292d7-4ef0-41a6-b8d5-b4e1717da367",
#     "instruction": "# Instructions\n\nScore each prompt and completions, where 1 is the lowest score and 10 is the highest score.",
#     "title": "Task title",
#     "description": "This project blah blah description",
#     "fields": [
#         {
#             "type": "category",
#             "field_id": "category_field",
#             "title": "Please",
#             "choices": [
#                 {
#                     "type": "number",
#                     "field_id": "item_price",
#                     "title": "Item Price",
#                     "description": "Leave empty if not applicable.",
#                     "required": False,
#                     "use_slider": True,
#                     "min": 1,
#                     "max": 10,
#                 }
#             ],
#         }
#     ],
#     "responses_required": 1,
#     "priority": 30,
# }
