import os

import aiohttp
from bittensor.btlogging import logging as logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

from commons.utils import log_retry_info
from template.protocol import SyntheticQA

SYNTHETIC_API_BASE_URL = os.getenv("SYNTHETIC_API_URL")


def _map_synthetic_response(response: dict) -> SyntheticQA:
    # Create a new dictionary to store the mapped fields
    mapped_data = {
        "prompt": response["prompt"],
        "ground_truth": response["ground_truth"],
    }

    responses = list(
        map(
            lambda resp: {
                "model": resp["model"],
                "completion": resp["completion"],
                "completion_id": resp["cid"],
            },
            response["responses"],
        )
    )

    mapped_data["responses"] = responses

    return SyntheticQA.model_validate(mapped_data)


class SyntheticAPI:
    _session: aiohttp.ClientSession | None = None

    @classmethod
    async def init_session(cls):
        if cls._session is None:
            cls._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return

    @classmethod
    async def get_qa(cls) -> SyntheticQA | None:
        await cls.init_session()

        path = f"{SYNTHETIC_API_BASE_URL}/api/synthetic-gen"
        logger.debug(f"Generating synthetic QA from {path}.")

        MAX_RETRIES = 6
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(MAX_RETRIES),
                wait=wait_exponential(multiplier=1, max=30),
                before_sleep=log_retry_info,
            ):
                with attempt:
                    async with cls._session.get(path) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        if "body" not in response_json:
                            raise ValueError("Invalid response from the server.")
                        synthetic_qa = _map_synthetic_response(response_json["body"])
                        logger.info("Synthetic QA generated and parsed successfully")
                        return synthetic_qa
        except RetryError:
            logger.error(
                f"Failed to generate synthetic QA after {MAX_RETRIES} retries."
            )
            raise
        except Exception:
            raise
