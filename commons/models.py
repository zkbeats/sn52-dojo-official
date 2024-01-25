from enum import StrEnum
from functools import lru_cache
from typing import Union

import bittensor as bt
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class ModelZoo(StrEnum):
    ELECTRA_LARGE_DISCRIMINATOR = (
        "OpenAssistant/reward-model-electra-large-discriminator"
    )
    DEBERTA_V3_LARGE_V2 = "OpenAssistant/reward-model-deberta-v3-large-v2"
    DEBERTA_V3_LARGE = "OpenAssistant/reward-model-deberta-v3-large"
    DEBERTA_V3_BASE = "OpenAssistant/reward-model-deberta-v3-base"


ModelName = Union[str, ModelZoo]


# adjust cache maxsize depending on however many models you have in your ModelZoo
@lru_cache(maxsize=10)
def get_cached_model(model_name: ModelName):
    return AutoModelForSequenceClassification.from_pretrained(model_name).eval()


@lru_cache(maxsize=10)
def get_cached_tokenizer(
    model_name: ModelName,
) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


class ModelUtils:
    @staticmethod
    def reward_model_score(
        model_name: ModelName,
        prompt,
        completion,
    ):
        model = get_cached_model(model_name)
        tokenizer = get_cached_tokenizer(model_name)
        question_ = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            tokenize=False,
            padding=True,
            truncation=True,
        )
        inputs = tokenizer(
            question_, completion, return_tensors="pt", padding=True, truncation=True
        )
        logits = model(**inputs).logits[0].cpu().detach()
        bt.logging.info(f"Raw logits: {logits}")
        # squish logits into range [0, 1]
        score = F.sigmoid(logits)
        bt.logging.info(f"Score: {score}, question: {prompt}, answer: {completion}")
        return score
