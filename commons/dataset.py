from enum import StrEnum
from typing import Dict, Iterable, List, Tuple
import random
import bittensor as bt

from datasets import load_dataset, interleave_datasets

from template.protocol import Completion, RankingRequest

seed = 42


class DatasetName(StrEnum):
    ANTHROPIC_HHRLHF = "Anthropic/hh-rlhf"
    STANFORD_SHP = "stanfordnlp/SHP"
    OPENAI_WEBGPT_COMPARISONS = "openai/webgpt_comparisons"
    # OPENASSISTANT_OASST1 = "OpenAssistant/oasst1"
    # OPENASSISTANT_OASST2 = "OpenAssistant/oasst2"
    YITINGXIE_RLHF_REWARD_DATASETS = "yitingxie/rlhf-reward-datasets"
    DAHOAS_SYNTHETIC_INSTRUCT_GPTJ_PAIRWISE = "Dahoas/synthetic-instruct-gptj-pairwise"


def get_anthropic_hhrlhf():
    return load_dataset(
        DatasetName.ANTHROPIC_HHRLHF,
        split="train",
        streaming=True,
    )


def get_stanford_shp():
    return (
        load_dataset(
            DatasetName.STANFORD_SHP,
            split="train",
            streaming=True,
        )
        .filter(lambda row: row["score_A"] != row["score_B"])
        .map(
            lambda row: {"chosen": row["human_ref_A"], "rejected": row["human_ref_B"]}
            if row["labels"] == 1
            else {"chosen": row["human_ref_B"], "rejected": row["human_ref_A"]}
        )
    )


def get_openai_webgpt_comparisons():
    return (
        load_dataset(
            DatasetName.OPENAI_WEBGPT_COMPARISONS,
            split="train",
            streaming=True,
        )
        .filter(lambda row: row["score_0"] != row["score_1"])
        .map(
            lambda row: {
                "chosen": row["answer_0"],
                "rejected": row["answer_1"],
                "context_chosen": row["quotes_0"],
                "context_rejected": row["quotes_1"],
            }
            if row["score_0"] > row["score_1"]
            else {
                "chosen": row["answer_1"],
                "rejected": row["answer_0"],
                "context_chosen": row["quotes_1"],
                "context_rejected": row["quotes_0"],
            }
        )
    )


eval_datasets = {
    DatasetName.ANTHROPIC_HHRLHF: iter(get_anthropic_hhrlhf()),
    DatasetName.STANFORD_SHP: iter(get_stanford_shp()),
    DatasetName.OPENAI_WEBGPT_COMPARISONS: iter(get_openai_webgpt_comparisons()),
    DatasetName.YITINGXIE_RLHF_REWARD_DATASETS: iter(
        load_dataset(
            DatasetName.YITINGXIE_RLHF_REWARD_DATASETS, split="train", streaming=True
        )
    ),
    DatasetName.DAHOAS_SYNTHETIC_INSTRUCT_GPTJ_PAIRWISE: iter(
        load_dataset(
            DatasetName.DAHOAS_SYNTHETIC_INSTRUCT_GPTJ_PAIRWISE,
            split="train",
            streaming=True,
        )
    ),
}


def next_circular(dataset_dict: Dict[str, Iterable], key: str):
    try:
        return next(dataset_dict[key])
    except StopIteration:
        if key == DatasetName.OPENAI_WEBGPT_COMPARISONS:
            dataset_dict[key] = iter(get_openai_webgpt_comparisons())
        elif key == DatasetName.STANFORD_SHP:
            dataset_dict[key] = iter(get_stanford_shp())
        elif key == DatasetName.ANTHROPIC_HHRLHF:
            dataset_dict[key] = iter(get_anthropic_hhrlhf())
        else:
            dataset_dict[key] = iter(load_dataset(key, split="train", streaming=True))

        return next(dataset_dict[key])


class EvalDatasetManager:
    @staticmethod
    def get_batch() -> List[Dict]:
        dataset_names = list(eval_datasets.keys())
        key = random.choice(dataset_names)
        bt.logging.info(f"Using dataset: {key}")
        batch_size = 32
        return [next_circular(eval_datasets, key) for _ in range(batch_size)]


# # NOTE this serves as a start for prompt/completion pairs to be generated because at first there will be no requests coming in
# oasst1 = load_dataset(
#     DatasetName.OPENASSISTANT_OASST1,
#     split="train",
#     streaming=True,
# )
# oasst2 = load_dataset(
#     DatasetName.OPENASSISTANT_OASST2,
#     split="train",
#     streaming=True,
# )


# def _is_oasst_prompt(row):
#     return not row["parent_id"] and row["role"] == "prompter"


# # ensure we only grab the 'text' fields from the dataset
# oasst1_prompts = oasst1.filter(lambda row: _is_oasst_prompt(row)).map(
#     lambda row: {"text": row["text"]}
# )
# oasst2_prompts = oasst2.filter(lambda row: _is_oasst_prompt(row)).map(
#     lambda row: {"text": row["text"]}
# )
# ALL_OASST_PROMPTS = "all_oasst_prompts"
# seed_datasets = {
#     ALL_OASST_PROMPTS: iter(
#         interleave_datasets(
#             [oasst1_prompts, oasst2_prompts],
#             probabilities=[0.5, 0.5],
#             seed=seed,
#             stopping_strategy="all_exhausted",
#         )
#     )
# }

# TODO change name to actual datset name
seed_dataset_name = "prooompt/test_dataset"


def get_seed_dataset():
    return load_dataset(
        seed_dataset_name,
        split="train",
        streaming=True,
    )


seed_dataset = iter(get_seed_dataset())


class SeedDataManager:
    @staticmethod
    def get_prompt_and_completions():
        global seed_dataset
        try:
            return SeedDataManager._map_seed_data(next(seed_dataset))
        except StopIteration:
            seed_dataset = iter(get_seed_dataset())
            return SeedDataManager._map_seed_data(next(seed_dataset))

    @staticmethod
    def _map_seed_data(row: Dict) -> Tuple[str, List[str]]:
        prompt = row["prompt"]
        completions = [c["response"] for c in row["responses"]]
        return prompt, completions
