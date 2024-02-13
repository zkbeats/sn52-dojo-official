import copy
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Tuple

import bittensor as bt
import jsonref
import requests
import torch
from pydantic import BaseModel


def initialise(
    config: bt.config,
) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, bt.axon]:
    # The wallet holds the cryptographic key pairs for the miner.
    wallet = bt.wallet(config=config)
    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    # The axon handles request processing, allowing validators to send this miner requests.
    axon = bt.axon(wallet=wallet, port=config.axon.port)
    return wallet, subtensor, metagraph, axon


def get_external_ip() -> str:
    response = requests.get("https://ifconfig.me/ip")
    response.raise_for_status()
    return response.text.strip()


def get_new_uuid():
    return str(uuid.uuid4())


def get_epoch_time():
    return time.time()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
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
        return {"type": "json_object", "schema": resolved_schema}
