import torch
from commons.dataset import EvalDatasetManager
from commons.reward_model.models import (
    ModelName,
    ModelZoo,
    get_cached_model,
    get_cached_tokenizer,
)

import bittensor as bt

from commons.utils import get_device


class EvalUtils:
    @staticmethod
    def classification_accuracy(model_name: ModelName) -> float:
        total_accuracy = 0
        num_batches = 10
        device = get_device()
        model = get_cached_model(model_name).to(device)
        tokenizer = get_cached_tokenizer(model_name)
        for _ in range(num_batches):
            batch_human_preference = EvalDatasetManager.get_batch()
            batch_accuracy = calc_batch_accuracy(
                batch_human_preference, model, tokenizer, device
            )
            total_accuracy += batch_accuracy
        accuracy = total_accuracy / num_batches
        bt.logging.info(f"Average accuracy: {accuracy}")
        return accuracy


def calc_batch_accuracy(batch, model, tokenizer, device):
    """Instead of performing classification accuracy on the entire dataset, we
    perform it on a single batch"""
    bt.logging.debug("Running classification accuracy...")
    total_chosen = 0
    # could actually batch this but might encounter leaked semaphore due to RAM constraints
    for row in batch:
        chosen_out = tokenizer(
            row["chosen"], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        rejected_out = tokenizer(
            row["rejected"], return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            chosen_logits = model(**chosen_out).logits
            rejected_logits = model(**rejected_out).logits
        logits = torch.cat([chosen_logits, rejected_logits], dim=1)
        # check if chosen is more likely than rejected
        indices = torch.argmax(logits, dim=-1)
        num_chosen = sum(indices == 0)
        total_chosen += num_chosen

    accuracy = total_chosen / len(batch)
    bt.logging.info(f"Accuracy: {accuracy}")
    return accuracy


def llm_classification_accuracy(dataset, model_name):
    # TODO determine classification accuracy by calling LLM apis
    pass


if __name__ == "__main__":
    EvalUtils.classification_accuracy(ModelZoo.DEBERTA_V3_LARGE_V2)
