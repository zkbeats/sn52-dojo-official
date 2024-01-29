import torch
from torch.utils.data import DataLoader
from commons.models import ModelName, get_cached_model, get_cached_tokenizer

from datasets import load_dataset
import bittensor as bt


def get_dataloader(hf_repo_id: str, **kwargs):
    dataset = load_dataset(hf_repo_id, streaming=True)
    dataset = dataset.with_format("torch")
    if (split := kwargs.get("split", None)) is not None:
        dataset = dataset[split]
    if "format" in kwargs:
        dataset = dataset.with_format(kwargs["format"])
    return DataLoader(dataset, batch_size=16)


class EvalUtils:
    @staticmethod
    def classification_accuracy(model_name: ModelName) -> float:
        split = "train"
        dataset = get_dataloader("Anthropic/hh-rlhf", split=split, format="torch")
        accuracy = calculate_classification_accuracy(dataset, model_name, split=split)
        return accuracy


def calculate_classification_accuracy(dataset, model_name):
    model = get_cached_model(model_name)
    tokenizer = get_cached_tokenizer(model_name)
    total_chosen = 0
    for idx, batch in enumerate(dataset):
        # TODO check if need to split the raw string into dialogues
        chosen = tokenizer(
            batch["chosen"], return_tensors="pt", padding=True, truncation=True
        )
        rejected = tokenizer(
            batch["rejected"], return_tensors="pt", padding=True, truncation=True
        )

        chosen_logits = model(**chosen).logits
        rejected_logits = model(**rejected).logits
        logits = torch.cat([chosen_logits, rejected_logits], dim=1)
        # check if chosen is more likely than rejected
        values, indices = torch.max(logits, dim=-1)
        num_chosen = sum(indices == 0)
        total_chosen += num_chosen

    accuracy = total_chosen / len(dataset)
    bt.logging.info(f"Accuracy: {accuracy}")
    return accuracy
