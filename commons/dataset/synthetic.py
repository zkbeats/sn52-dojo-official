import asyncio
import functools
import os
import time
from typing import List
import aiohttp

import bittensor as bt
from dotenv import load_dotenv
from tenacity import AsyncRetrying, RetryError, stop_after_attempt
from commons.utils import log_retry_info

from template.protocol import SyntheticQA

load_dotenv()


SYNTHETIC_API_BASE_URL = os.getenv("SYNTHETIC_API_URL")


@functools.lru_cache
def get_client_session():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))


class SyntheticAPI:
    @classmethod
    async def get_qa(cls) -> List[SyntheticQA]:
        path = f"{SYNTHETIC_API_BASE_URL}/api/synthetic-gen"
        bt.logging.debug(f"Generating synthetic QA from {path}.")
        # Instantiate the aiohttp ClientSession outside the loop

        session = get_client_session()
        MAX_RETRIES = 3
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(MAX_RETRIES), before_sleep=log_retry_info
            ):
                with attempt:
                    async with session.get(path) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        if "body" not in response_json:
                            raise ValueError("Invalid response from the server.")
                        synthetic_QAs = [
                            SyntheticQA.parse_obj(item)
                            for item in response_json["body"]
                        ]
                        bt.logging.info(
                            f"{len(synthetic_QAs)} Synthetic QA generated successfully."
                        )
                        return synthetic_QAs
        except RetryError:
            bt.logging.error("Failed to generate synthetic QA after retries.")
            return None


if __name__ == "__main__":

    async def simulate_vali():
        while True:
            start_time = time.perf_counter()
            synthetic_qa = await SyntheticAPI.get_qa()
            elapsed_time = time.perf_counter() - start_time
            print(f"Time taken to get QA: {elapsed_time:.2f} seconds")

    async def main():
        await asyncio.gather(*[simulate_vali()])

    asyncio.run(main())
