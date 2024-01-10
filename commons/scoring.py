import json
from collections import defaultdict
from typing import Dict, List

import bittensor as bt
import numpy as np
import scipy
from attr import define, field

from commons.custom_exceptions import InvalidScoreLength
from commons.utils import DotDict
from template.protocol import RankingRequest


@define(kw_only=True, frozen=True, slots=True)
class RankingResult:
    # Each request id has multiple completions, where each miner scores each of these completions.
    request_id: str
    hotkey_to_scores: Dict[str, float] = field(factory=dict)


class Scoring:
    @staticmethod
    def _map_responses_to_result(responses: List[RankingRequest]):
        if len(responses) == 0:
            raise ValueError("Responses cannot be empty")

        request_id = responses[0].request_id
        nested_dict: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for response in responses:
            hotkey = response.axon.hotkey
            for rank in response.ranks:
                nested_dict[rank.cid][hotkey] = rank.score

        data = json.loads(json.dumps(nested_dict))
        return DotDict({"request_id": request_id, "cid_to_hotkey_to_score": data})

    @staticmethod
    def _spearman_correlation(responses: List[RankingRequest]):
        responses = [r for r in responses if len(r.ranks) > 0]
        if not responses or all(len(r.ranks) == 0 for r in responses):
            return {}

        result = Scoring._map_responses_to_result(responses)
        averages = [
            np.mean(list(hotkey_scores.values()))
            for hotkey_scores in result.cid_to_hotkey_to_score.values()
        ]

        miner_scores = np.array(
            [
                [
                    hotkey_scores.get(r.axon.hotkey, 0)
                    for hotkey_scores in result.cid_to_hotkey_to_score.values()
                ]
                for r in responses
            ]
        )

        bt.logging.debug(f"Miner scores shape: {np.array(miner_scores).shape}")
        bt.logging.debug(f"Average scores shape: {np.array(averages).shape}")

        # compute spearman correlation and handle nan values
        correlations = [
            np.nan_to_num(scipy.stats.spearmanr(miner_score, averages)[0])
            for miner_score in miner_scores
        ]

        hotkeys = [r.axon.hotkey for r in responses]
        return hotkeys, correlations

    @staticmethod
    def score_responses(responses: List[RankingRequest]):
        """Given a list of responses, will only return a dict of hotkey to their normalized scores.
        e.g. if a miner failed to respond, its hotkey won't be a key in the dict.
        """
        # all_hotkeys = [r.axon.hotkey for r in responses]

        hotkeys, spearman_correlations = Scoring._spearman_correlation(responses)
        # scale values in the range [-1, 1] to [0, 1]
        spearman_correlations = 0.5 * (np.array(spearman_correlations) + 1)

        # ensure sum of scores == 1
        scores = spearman_correlations / spearman_correlations.sum()

        return dict(zip(hotkeys, scores.tolist()))
