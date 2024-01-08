import torch
import random
import bittensor as bt
from typing import List


def get_all_serving_uids(metagraph: bt.metagraph):
    uids = [uid for uid in range(metagraph.n.item()) if metagraph.axons[uid].is_serving]
    return uids


def check_uid_availability(
    metagraph: bt.metagraph, uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(
    metagraph: bt.metagraph, k: int, config: bt.config
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph."""
    avail_uids = []

    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(
            metagraph, uid, config.neuron.vpermit_tao_limit
        )

        if uid_is_available:
            avail_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    bt.logging.debug(f"available uids: {avail_uids}")
    if not len(avail_uids):
        return torch.tensor([])
    # If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    if len(avail_uids) < k:
        return torch.tensor(avail_uids)

    uids = torch.tensor(random.sample(avail_uids, k))
    return uids
