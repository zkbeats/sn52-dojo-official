import torch
import random
import bittensor as bt


def get_all_serving_uids(metagraph: bt.metagraph):
    uids = [uid for uid in range(metagraph.n.item()) if metagraph.axons[uid].is_serving]
    return uids


def is_uid_available(metagraph: bt.metagraph, uid: int) -> bool:
    """Check if uid is available."""
    # filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False

    return True


def get_random_uids(metagraph: bt.metagraph, k: int) -> torch.LongTensor:
    """Returns k available random uids from the metagraph."""
    avail_uids = []

    for uid in range(metagraph.n.item()):
        if is_uid_available(metagraph, uid):
            avail_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    bt.logging.info(f"available uids: {avail_uids}")
    if not len(avail_uids):
        return torch.tensor([])

    # If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    if len(avail_uids) < k:
        return torch.tensor(avail_uids)

    uids = torch.tensor(random.sample(avail_uids, k))
    return uids
