import asyncio
import torch
from commons.dataset import EvalDatasetManager
from commons.factory import Factory
from commons.llm.openai_proxy import Provider
from commons.reward_model.models import (
    ModelUtils,
    get_cached_model,
    get_cached_tokenizer,
)
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

import bittensor as bt

from commons.utils import get_device
from template.protocol import ModelConfig, ScoringMethod


class EvalUtils:
    @staticmethod
    async def classification_accuracy(
        scoring_method: ScoringMethod,
        model_config: ModelConfig = None,
    ) -> float:
        total_accuracy = 0
        num_batches = Factory.get_config().eval.num_batches
        for _ in range(num_batches):
            batch_human_preference = EvalDatasetManager.get_batch()
            if scoring_method == ScoringMethod.HF_MODEL:
                device = get_device()
                model = get_cached_model(model_config.model_name).to(device)
                tokenizer = get_cached_tokenizer(model_config.model_name)
                batch_accuracy = hf_classify_accuracy(
                    batch_human_preference, model, tokenizer, device
                )

            elif scoring_method == ScoringMethod.LLM_API:
                batch_accuracy = await llm_classify_accuracy(
                    batch_human_preference,
                    model_config=model_config,
                )
                pass

            total_accuracy += batch_accuracy
        accuracy = total_accuracy / num_batches
        bt.logging.info(f"Average accuracy: {accuracy}")
        return accuracy


def hf_classify_accuracy(batch, model, tokenizer, device):
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
        # checking for 0 since we cat chosen first
        num_chosen = sum(indices == 0)
        total_chosen += num_chosen

    accuracy = total_chosen / len(batch)
    bt.logging.info(f"Accuracy: {accuracy}")
    return accuracy


def log_retry_info(retry_state):
    """Meant to be used with tenaicty's before_sleep callback"""
    bt.logging.warning(
        f"Retry attempt {retry_state.attempt_number} failed with exception: {retry_state.outcome.exception()}",
    )


async def llm_classify_accuracy(
    batch,
    model_config: ModelConfig,
):
    """Instead of performing classification accuracy on the entire dataset, we
    perform it on a single batch"""
    bt.logging.debug("Running LLM classification accuracy...")
    total_chosen = 0
    # could actually batch this but might encounter leaked semaphore due to RAM constraints
    MAX_RETRIES = 5
    for row in batch:
        # TODO @dev handle rate limiting from the model provider
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(MAX_RETRIES),
                before_sleep=log_retry_info,
                wait=wait_exponential(multiplier=1, min=4, max=12),
            ):
                with attempt:
                    is_chosen = await ModelUtils.llm_eval_human_preference(
                        model_config=model_config,
                        chosen=row["chosen"],
                        rejected=row["rejected"],
                    )
                total_chosen += is_chosen

        except RetryError:
            bt.logging.error(
                f"Failed to generate completion after {MAX_RETRIES} attempts while evaluating human preference.",
            )

    accuracy = total_chosen / len(batch)
    bt.logging.info(
        f"Batch Accuracy: {accuracy}, with LLM Model Config: {model_config}"
    )
    return accuracy


async def main():
    batch_human_preference = EvalDatasetManager.get_batch()
    batch_accuracy = await llm_classify_accuracy(
        batch_human_preference,
        # TODO normally this would be stored inside of our Datamanager class
        ModelConfig(
            provider=Provider.TOGETHER_AI,
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
    )
    bt.logging.info(f"Batch accuracy: {batch_accuracy}")


if __name__ == "__main__":
    # EvalUtils.classification_accuracy(ModelZoo.DEBERTA_V3_LARGE_V2)
    asyncio.run(main())
