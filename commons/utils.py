import copy
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Tuple, Type

import bittensor as bt
import jsonref
import requests
import torch
from pydantic import BaseModel
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential_jitter

import wandb


def get_new_uuid():
    return str(uuid.uuid4())


def init_wandb(config: bt.config, my_uid, wallet: bt.wallet):
    import template

    run_name = f"{config.neuron.type}-{my_uid}-{template.__version__}"
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = template.__version__

    # Initialize the wandb run for the single project
    kwargs = {
        "name": run_name,
        "project": "subnet",
        "entity": "tensorplex",
        "config": config,
        "dir": config.full_path,
        "reinit": True,
    }
    run = wandb.init(**kwargs)

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run with {kwargs=}")
    return run


def log_retry_info(retry_state):
    """Meant to be used with tenacity's before_sleep callback"""
    bt.logging.warning(
        f"Retry attempt {retry_state.attempt_number} failed with exception: {retry_state.outcome.exception()}",
    )


def serve_axon(
    subtensor: bt.subtensor, axon: bt.axon, config: bt.config, max_attempts: int = 10
) -> bool:
    """A wrapper around the underlying self.axon.serve(...) call with retries"""
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            before_sleep=log_retry_info,
            wait=wait_exponential_jitter(initial=6, max=24, jitter=1),
        ):
            with attempt:
                serve_success = subtensor.serve_axon(netuid=config.netuid, axon=axon)
                if serve_success:
                    return True

                raise Exception(
                    "Failed to serve axon, probaby due to many miners/validators trying to serve in this period."
                )
    except RetryError:
        bt.logging.error(f"Failed to serve axon after {max_attempts} attempts.")
        pass
    return False


def initialise(
    config: bt.config,
) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, bt.axon]:
    # Build Bittensor objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects....")
    # The wallet holds the cryptographic key pairs for the miner.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")
    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")
    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")
    # The axon handles request processing, allowing validators to send this miner requests.
    axon = bt.axon(wallet=wallet, port=config.axon.port)
    bt.logging.info(f"Axon: {axon}")
    return wallet, subtensor, metagraph, axon


def check_registered(subtensor, wallet, config):
    # --- Check for registration.
    if not subtensor.is_hotkey_registered(
        netuid=config.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    ):
        bt.logging.error(
            f"Wallet: {wallet} is not registered on netuid {config.netuid}."
            f" Please register the hotkey using `btcli s register` before trying again"
        )
        exit()


def should_resync_metagraph(subtensor, metagraph, wallet, config):
    """
    Check if enough blocks have elapsed since the last checkpoint to sync.
    """
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    return (
        ttl_get_block(subtensor) - metagraph.last_update[uid]
    ) > config.neuron.epoch_length


def get_external_ip() -> str:
    response = requests.get("https://ifconfig.me/ip")
    response.raise_for_status()
    return response.text.strip()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class DotDict(OrderedDict):
    """
    Quick and dirty implementation of a dot-able dict, which allows access and
    assignment via object properties rather than dict indexing.
    """

    def __init__(self, *args, **kwargs):
        # we could just call super(DotDict, self).__init__(*args, **kwargs)
        # but that won't get us nested dotdict objects
        od = OrderedDict(*args, **kwargs)
        for key, val in od.items():
            if isinstance(val, Mapping):
                value = DotDict(val)
            else:
                value = val
            self[key] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {name}") from ex

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {k}") from ex

    __setattr__ = OrderedDict.__setitem__


def remove_key(input_dict, key, depth=0):
    """Recursively remove a specified key from a nested dictionary, keeping track of depth."""
    for k, v in list(input_dict.items()):
        if k == key:
            del input_dict[k]
        elif isinstance(v, dict):
            remove_key(v, key, depth=depth + 1)
    return input_dict


def _resolve_references(json_str):
    return jsonref.loads(json_str)


class PydanticUtils:
    @staticmethod
    def build_response_format(model: BaseModel):
        """Build a response format for OpenAI API calls."""
        schema = model.schema_json()
        resolved_schema = copy.deepcopy(_resolve_references(schema))

        if "definitions" in resolved_schema:
            resolved_schema.pop("definitions")

        resolved_schema = remove_key(resolved_schema, "title")
        resolved_schema = remove_key(resolved_schema, "additionalProperties")
        required = resolved_schema.get("required", [])
        resolved_schema = remove_key(resolved_schema, "required")
        resolved_schema["required"] = required
        # return {"type": "json_object", "schema": resolved_schema}
        return resolved_schema

    @classmethod
    def build_minimal_json(cls, model: Type[BaseModel]):
        return {
            field_name: model.__fields__[field_name].field_info.description
            for field_name in model.__fields__
            if model.__fields__[field_name].field_info.description
        }

        # {'code': {'description': 'Code solution to the question', 'type': 'string'}, 'language': {'description': 'Programming language of the code', 'type': 'string'}, 'installation_commands': {'description': 'Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run', 'type': 'string'}, 'additional_notes': {'description': 'Any additional notes or comments about the code solution', 'type': 'string'}}, 'required': ['code', 'language', 'installation_commands']}


# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from functools import lru_cache, update_wrapper
from math import floor
from typing import Any, Callable


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(subtensor) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the miner or validator instance
    """
    return subtensor.get_current_block()
