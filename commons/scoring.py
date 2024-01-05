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
        nested_dict = defaultdict(lambda: defaultdict(float))
        for response in responses:
            hotkey = response.axon.hotkey
            for rank in response.ranks:
                nested_dict[rank.cid][hotkey] = rank.score

        data = json.loads(json.dumps(nested_dict))
        return DotDict({"request_id": request_id, "cid_to_hotkey_to_score": data})

    @staticmethod
    def _spearman_correlation(responses: List[RankingRequest]):
        responses = [r for r in responses if len(r.ranks) > 0]
        if len(responses) == 0:
            return {}

        miner_hotkeys = [r.axon.hotkey for r in responses]
        result = Scoring._map_responses_to_result(responses)

        # prepare for calculating spearman correlation
        averages = []
        for cid in result.cid_to_hotkey_to_score:
            scores = [score for _, score in result.cid_to_hotkey_to_score[cid].items()]
            averages.append(np.average(scores))

        miner_scores = np.zeros((len(responses), responses[0].n_completions))
        for i, cid in enumerate(result.cid_to_hotkey_to_score):
            hotkey_to_score = result.cid_to_hotkey_to_score[cid]
            for j, k in enumerate(hotkey_to_score):
                score = hotkey_to_score[k]
                miner_scores[j][i] = score

        bt.logging.debug(f"Miner scores shape: {np.array(miner_scores).shape}")
        bt.logging.debug(f"Average scores shape: {np.array(averages).shape}")

        correlations = []
        for miner_score in miner_scores:
            spearman_c, _ = scipy.stats.spearmanr(miner_score, averages)
            # handle nan values
            spearman_c = np.nan_to_num(spearman_c)
            correlations.append(spearman_c)

        if len(miner_hotkeys) != len(correlations):
            raise InvalidScoreLength(
                "Miner hotkeys and correlations must be the same length"
            )
        return {hotkey: score for hotkey, score in zip(miner_hotkeys, correlations)}

    @staticmethod
    def score_responses(responses: List[RankingRequest]):
        hotkey_to_scores = Scoring._spearman_correlation(responses)

        # for response in responses:
        #     hotkey = response.axon.hotkey
        #     if hotkey not in hotkey_to_scores:
        #         hotkey_to_scores[hotkey] = 0.0

        bt.logging.debug(f"Scored responses: {hotkey_to_scores}")
        return hotkey_to_scores
