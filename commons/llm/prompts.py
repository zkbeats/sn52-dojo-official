from typing import List, Optional
from pydantic import BaseModel, validator

from template.protocol import Response


system_score_completion_prompt = """
You are a helpful assistant that provides responses in JSON. Your task is to score the quality of each of the completions from a model with respect to a prompt, where your score must be in the range {range_lower} to {range_upper}, where {range_lower} is the lowest score and {range_upper} is the highest score, and a higher score represents a higher quality response. You must provide your answer in JSON format.
"""

user_score_completion_prompt = """
Prompt:
{prompt}

{completions_prompt}

Assistant:
"""

completion_item_prompt = """
Model #{idx}
{text}
"""

system_eval_human_preference_prompt = """
You are a helpful assistant that provides responses in JSON. Your task is to choose one of two pieces of texts in terms of your preference. You must provide your answer in JSON format.
"""

user_eval_human_preference_prompt = """
Text {chosen_idx}:
{chosen}

Text {rejected_idx}:
{rejected}
"""


# NOTE @miner here you should either use ranges [0,1] or [1,10], do not use [-1,1] as LLMs will perform poorly
class ScoreRange(BaseModel):
    lower: int
    upper: int

    @validator("lower")
    def lower_must_be_less_than_upper(cls, v, values, **kwargs):
        if "upper" in values and v >= values["upper"]:
            raise ValueError("lower must be less than upper")
        return v


class PromptBuilder:
    @staticmethod
    def build_user_score_completion_prompt(prompt: str, completions: List[Response]):
        if not len(completions):
            raise ValueError("Cannot build prompt without any completions")

        completion_prompts = [
            completion_item_prompt.format(idx=c.cid, text=c.json()) for c in completions
        ]
        formatted_prompt = user_score_completion_prompt.format(
            prompt=prompt,
            completions_prompt="\n".join(completion_prompts),
        )
        # bt.logging.debug(f"User prompt: {formatted_prompt}")
        return formatted_prompt

    @staticmethod
    def build_system_score_completion_prompt(score_range: ScoreRange):
        formatted_prompt = system_score_completion_prompt.format(
            range_lower=score_range.lower, range_upper=score_range.upper
        )
        # bt.logging.debug(f"System prompt: {formatted_prompt}")
        return formatted_prompt

    @staticmethod
    def build_system_eval_human_preference_prompt():
        return system_eval_human_preference_prompt

    @staticmethod
    def build_user_eval_human_preference_prompt(
        chosen: str,
        rejected: str,
        chosen_idx,
        rejected_idx,
        prompt: Optional[str] = None,
    ):
        formatted_prompt = user_eval_human_preference_prompt.format(
            chosen=chosen,
            rejected=rejected,
            chosen_idx=chosen_idx,
            rejected_idx=rejected_idx,
        )
        # bt.logging.debug(f"User prompt: {formatted_prompt}")
        return formatted_prompt
