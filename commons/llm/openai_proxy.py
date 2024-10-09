import os

from openai import AsyncOpenAI, OpenAI
from strenum import StrEnum

# import instructor

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_BASE_URL = "https://api.together.xyz/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class Provider(StrEnum):
    TOGETHER_AI = "togetherai"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


def get_openai_kwargs(provider: Provider):
    if provider == Provider.TOGETHER_AI:
        return {"api_key": TOGETHER_API_KEY, "base_url": TOGETHER_API_BASE_URL}
    elif provider == Provider.OPENAI:
        return {"api_key": OPENAI_API_KEY, "base_url": OPENAI_API_BASE_URL}
    elif provider == Provider.OPENROUTER:
        return {"api_key": OPENROUTER_API_KEY, "base_url": OPENROUTER_API_BASE_URL}
    raise ValueError(f"Unknown provider specified , provider: {provider}")


def get_openai_client(provider: Provider):
    # known_providers = [provider.value for provider in Provider]
    # TODO @dev use instructor when bittensor migrates to pydantic v2
    # if provider in known_providers:
    #     return instructor.apatch(
    #         AsyncOpenAI(**get_openai_kwargs(provider)), mode=instructor.Mode.MD_JSON
    #     )
    kwargs = get_openai_kwargs(provider)
    return AsyncOpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"])


def get_sync_openai_client(provider: Provider):
    # known_providers = [provider.value for provider in Provider]
    # TODO @dev use instructor when bittensor migrates to pydantic v2
    # if provider in known_providers:
    #     return instructor.apatch(
    #         AsyncOpenAI(**get_openai_kwargs(provider)), mode=instructor.Mode.MD_JSON
    #     )
    kwargs = get_openai_kwargs(provider)
    return OpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"])
