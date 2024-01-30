from enum import StrEnum
from functools import lru_cache
from typing import List, Union

import bittensor as bt
from dotenv import load_dotenv
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from commons.llm.openai_proxy import Provider, get_openai_client
from commons.llm.prompts import PromptBuilder, ScoreRange
from commons.objects import ScoresResponse
from commons.utils import PydanticUtils
from template.protocol import Completion

load_dotenv()


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
    def _hf_score(
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

    @staticmethod
    async def _llm_api_score(
        provider: Provider, model_name: str, prompt: str, completions: List[Completion]
    ):
        client = get_openai_client(provider)
        score_range = ScoreRange(lower=0, upper=1)
        system_prompt = PromptBuilder.build_system_score_completion_prompt(
            score_range=score_range
        )
        user_prompt = PromptBuilder.build_user_score_completion_prompt(
            prompt, completions
        )
        # TODO use instructor instead when compatible due to pydantic v2
        response = await client.chat.completions.create(
            model=model_name,
            temperature=0,
            stream=False,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=PydanticUtils.build_response_format(ScoresResponse),
        )
        try:
            content = response.choices[0].message.content
            response = ScoresResponse.parse_raw(content)
            # ensure that our scores are in the range [0, 1]
            for score_item in response.scores:
                score_item.score = score_item.score / score_range.upper

        except Exception as e:
            bt.logging.error(f"Error occurred while trying to parsing response: {e}")
        bt.logging.info(f"Response: {response}")
        return response
