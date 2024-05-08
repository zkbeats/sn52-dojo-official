import json
import os
from typing import List, Union

import bittensor as bt
from fastapi.encoders import jsonable_encoder
import huggingface_hub
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd

from template.protocol import Response

load_dotenv()


os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

HF_TOKEN = os.getenv("HF_TOKEN")
# NOTE this will be where our dataset gets collected over time
HF_REPO_ID = "prooompt/test_dataset"
hf_api = huggingface_hub.HfApi(token=HF_TOKEN)
repo_kwargs = {"repo_id": HF_REPO_ID, "repo_type": "dataset"}


class HuggingFaceUtils:
    _dataset_dir = "./hf_datasets"
    _dataset_filename = "dataset_sn18.jsonl"
    _full_path = f"{_dataset_dir}/{_dataset_filename}"

    @classmethod
    def _download_dataset(cls):
        # download existing data from huggingface hub
        # download the dataset from the huggingface hub
        hf_api.snapshot_download(
            allow_patterns=["*.jsonl"],
            local_dir=cls._dataset_dir,
            revision="main",
            **repo_kwargs,
        )

    # @classmethod
    # def append_data(
    #     cls,
    #     prompt: str,
    #     completions: List[Response],
    #     scores: List[Union[float, None]],
    #     wallet: bt.wallet,
    # ):
    #     if not os.path.exists(cls._full_path):
    #         cls._download_dataset()

    #     with open(cls._full_path, "r") as file:
    #         modified_data = [json.loads(line) for line in file]

    #     assert isinstance(modified_data, list)

    #     best_completion = None
    #     best_score = float("-inf")
    #     for i in range(len(completions)):
    #         completion = completions[i]
    #         score = scores[i]
    #         if score is not None and score > best_score:
    #             best_score = score
    #             best_completion = completion

    #     if best_completion is None:
    #         raise ValueError("No best completion found")

    #     new_data = {
    #         "prompt": prompt,
    #         "completions": jsonable_encoder(completions),
    #         "num_completions": len(completions),
    #         "best_completion": str(best_completion.text),
    #     }

    #     # Append new_data to the existing_data list and write it back to the file
    #     modified_data.append(new_data)
    #     with open(cls._full_path, "w") as file:
    #         for entry in modified_data:
    #             file.write(json.dumps(entry) + "\n")

    #     hotkey = wallet.hotkey.ss58_address
    #     operation = CommitOperationAdd(
    #         path_in_repo=cls._dataset_filename, path_or_fileobj=cls._full_path
    #     )
    #     hf_api.create_commits_on_pr(
    #         addition_commits=[[operation]],
    #         deletion_commits=[],
    #         commit_message=f"Validator hotkey: {hotkey} adding data to dataset",
    #         merge_pr=False,
    #         **repo_kwargs,
    #     )

    #     return modified_data


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
