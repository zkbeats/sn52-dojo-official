from typing import List
import huggingface_hub
from huggingface_hub import CommitOperationAdd
from dotenv import load_dotenv
import os
import json
from pydantic import Field

from template.protocol import Completion, Rank


class DatasetItem(Completion):
    role: str = Field(description="Role of the response", example="ai")


load_dotenv()


# # NOTE login with your huggingface hub first
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = 1

# common huggingface repo id
HF_TOKEN = os.getenv("HF_TOKEN")
# NOTE this will be where our dataset gets collected over time
# TODO @dev change to actual name of dataset to store all our collected data
HF_REPO_ID = "prooompt/test_dataset"
hf_api = huggingface_hub.HfApi(token=HF_TOKEN)
repo_kwargs = {"repo_id": HF_REPO_ID, "repo_type": "dataset"}

for discussion in hf_api.get_repo_discussions(**repo_kwargs):
    print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")


class HuggingFaceUtils:
    _dataset_dir = "./hf_datasets"
    _dataset_filename = "dataset_sn18.jsonl"
    _full_path = f"{_dataset_dir}/{_dataset_filename}"

    @classmethod
    def download_dataset(cls):
        # download existing data from huggingface hub
        # download the dataset from the huggingface hub
        hf_api.snapshot_download(
            allow_patterns=["*.jsonl"],
            local_dir=cls._dataset_dir,
            revision="main",
            **repo_kwargs,
        )

    @classmethod
    def append_data(
        cls, prompt: str, dataset_items: List[DatasetItem], ranks: List[Rank]
    ):
        with open(cls._full_path, "r") as file:
            modified_data = [json.loads(line) for line in file]

        assert isinstance(modified_data, list)

        # Find the corresponding rank for each dataset item and add it to the new_data dictionary
        best_completion = None
        best_score = float("-inf")
        for item in dataset_items:
            for rank in ranks:
                if item.cid != rank.cid:
                    continue
                if rank.score > best_score:
                    best_score = rank.score
                    best_completion = item.text
                    break
        if best_completion is None:
            raise ValueError("No rank found for any of the completions")

        new_data = {
            "prompt": prompt,
            # each response: {'role': 'ai | human', 'response': '<response text>', 'metadata': {'model_name': ...}}
            "completions": dataset_items,
            "num_completions": len(dataset_items),
            "best_completion": best_completion,
        }

        # Append new_data to the existing_data list and write it back to the file
        modified_data.append(new_data)
        with open(cls._full_path, "w") as file:
            for entry in modified_data:
                file.write(json.dumps(entry) + "\n")
        return modified_data

    @classmethod
    def add_commit(cls, path: str, hotkey: str):
        operation = CommitOperationAdd(
            path_in_repo=cls._dataset_filename, path_or_fileobj=cls._full_path
        )
        # TODO @dev need some way of verifying dataset follows format, since
        # git lfs previews not available
        hf_api.create_commits_on_pr(
            addition_commits=[[operation]],
            deletion_commits=[],
            commit_message=f"Validator hotkey: {hotkey} adding data to dataset",
            merge_pr=False,
            **repo_kwargs,
        )


def verify_dataset():
    """Double check that our appending of data actually worked since we cannot
    preview git lfs file diffs"""
    from datasets import load_dataset

    dataset = load_dataset(path=HF_REPO_ID, split="train")
    all_data = []
    for item in dataset:
        all_data.append(item)

    output_file = "collected_data.jsonl"
    with open(output_file, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
