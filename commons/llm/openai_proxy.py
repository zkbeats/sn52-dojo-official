from functools import lru_cache
import os
from enum import StrEnum

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
# import instructor

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_BASE_URL = "https://api.together.xyz/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE_URL = "https://api.openai.com/v1"


class Provider(StrEnum):
    TOGETHER_AI = "togetherai"
    OPENAI = "openai"


def get_openai_kwargs(provider: Provider):
    if provider == Provider.TOGETHER_AI:
        return {"api_key": TOGETHER_API_KEY, "base_url": TOGETHER_API_BASE_URL}
    elif provider == Provider.OPENAI:
        return {"api_key": OPENAI_API_KEY, "base_url": OPENAI_API_BASE_URL}
    raise ValueError(f"Unknown provider specified , provider: {provider}")


@lru_cache(maxsize=10)
def get_openai_client(provider: Provider):
    # known_providers = [provider.value for provider in Provider]
    # TODO @dev use instructor when bittensor migrates to pydantic v2
    # if provider in known_providers:
    #     return instructor.apatch(
    #         AsyncOpenAI(**get_openai_kwargs(provider)), mode=instructor.Mode.MD_JSON
    #     )
    kwargs = get_openai_kwargs(provider)
    return AsyncOpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"])
