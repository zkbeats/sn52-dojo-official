from collections import defaultdict
from typing import List, Optional
import uuid
import torch
import random
import bittensor as bt
import bisect
from Crypto.Hash import keccak

from commons.utils import get_new_uuid


def get_all_serving_uids(metagraph: bt.metagraph):
    uids = [uid for uid in range(metagraph.n.item()) if metagraph.axons[uid].is_serving]
    return uids


def is_uid_available(metagraph: bt.metagraph, uid: int) -> bool:
    """Check if uid is available."""
    # filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    return True


def is_miner(metagraph: bt.metagraph, uid: int) -> bool:
    """Check if uid is a validator."""
    stakes = metagraph.S.tolist()
    from template import VALIDATOR_MIN_STAKE

    return stakes[uid] < VALIDATOR_MIN_STAKE


def get_random_miner_uids(metagraph: bt.metagraph, k: int) -> torch.LongTensor:
    """Returns k available random uids from the metagraph."""
    avail_uids = []

    for uid in range(metagraph.n.item()):
        if metagraph.axons[uid].is_serving and is_miner(metagraph, uid):
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


# TODO v2
class HashRing:
    _instance = None
    ring = []
    nodes_hash_map = {}
    VIRTUAL_NODES = 160

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HashRing, cls).__new__(cls)
            cls.ring = []
            cls.nodes_hash_map = {}
        return cls._instance

    @classmethod
    def __init__(cls, nodes: Optional[List[int]] = None):
        if not nodes:
            return

        cls.reset()
        for node in nodes:
            cls.add_uid(node)

    @classmethod
    def reset(cls):
        cls.ring = []
        cls.nodes_hash_map = {}

    @classmethod
    def hash_function(cls, key):
        def keccak256_hash(data):
            k = keccak.new(digest_bits=256)
            k.update(data.encode("utf-8"))
            return k.hexdigest()

        return int(keccak256_hash(key), 16)

    @classmethod
    def add_uid(cls, node: int):
        for vnode in range(cls.VIRTUAL_NODES):
            vnode_key = f"{node}#vnode{vnode}"
            hash_value = cls.hash_function(vnode_key)
            cls.ring.append(hash_value)
            cls.nodes_hash_map[hash_value] = node
        cls.ring.sort()

    @classmethod
    def remove_uid(cls, node: int):
        for vnode in range(cls.VIRTUAL_NODES):
            vnode_key = f"{node}#vnode{vnode}"
            hash_value = cls.hash_function(vnode_key)
            if hash_value in cls.ring:
                cls.ring.remove(hash_value)
                del cls.nodes_hash_map[hash_value]

    @classmethod
    def get_target_uids(cls, key, k: int = 4):
        if not cls.ring or k <= 0:
            return []
        hash_value = cls.hash_function(key)
        index = bisect.bisect_left(cls.ring, hash_value) % len(cls.ring)
        nodes = []
        for i in range(min(k, len(cls.ring))):
            node_index = (index + i) % len(cls.ring)
            node = cls.nodes_hash_map[cls.ring[node_index]]
            if node not in nodes:
                nodes.append(node)
        return nodes


nodes = list(range(1, 193))
ch = HashRing(nodes)
requests_per_node = defaultdict(int)
for _ in range(100_000):
    request_key = str(uuid.uuid4())
    target_nodes = ch.get_target_uids(request_key)
    for node in target_nodes:
        requests_per_node[node] += 1

for node, count in requests_per_node.items():
    print(f"Node {node} received {count} requests.")

import matplotlib.pyplot as plt

nodes = list(requests_per_node.keys())
requests = list(requests_per_node.values())

plt.figure(figsize=(10, 6))
plt.bar(nodes, requests, color="skyblue")
plt.xlabel("Node")
plt.ylabel("Number of Requests")
plt.title("Distribution of Requests per Node")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TODO v3
# from uhashring import HashRing

# requests_per_node = defaultdict(int)
# nodes = list(range(1, 193))
# # create a consistent hash ring of 3 nodes of weight 1
# hr = HashRing(nodes=[str(node) for node in nodes])

# # Generate 100 UUIDs and tabulate requests per node
# for _ in range(100_000):
#     request_key = str(uuid.uuid4())
#     target_nodes = hr.get_node(request_key)
#     requests_per_node[int(target_nodes)] += 1
#     # for node in target_nodes:
#     #     requests_per_node[node] += 1

# # Print the total number of requests per node
# for node, count in requests_per_node.items():
#     print(f"Node {node} received {count} requests.")

# import matplotlib.pyplot as plt

# # Convert requests_per_node to lists for plotting
# nodes = list(requests_per_node.keys())
# requests = list(requests_per_node.values())

# plt.figure(figsize=(10, 6))
# plt.bar(nodes, requests, color="skyblue")
# plt.xlabel("Node")
# plt.ylabel("Number of Requests")
# plt.title("Distribution of Requests per Node")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
