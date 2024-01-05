from collections import defaultdict
from email.policy import default
import json
from typing import List

import numpy as np
from template.protocol import RankingRequest, RankingResult


class Scoring:
    @staticmethod
    def _map_responses_to_result(responses: List[RankingRequest]):
        if len(responses) == 0:
            return None

        request_id = responses[0].request_id
        nested_dict = defaultdict(lambda: defaultdict(float))
        for response in responses:
            hotkey = response.axon.hotkey
            for rank in response.ranks:
                nested_dict[rank.cid][hotkey] = rank.score

        data = json.loads(json.dumps(nested_dict))
        return RankingResult(request_id=request_id, cid_to_hotkey_to_score=data)

    @staticmethod
    def score_responses(responses: List[RankingRequest]):
        if len(responses) == 0:
            return None
        result = Scoring._map_responses_to_result(responses)

        # calculate average score for each completion id
        cid_to_average = {}
        for cid in result.cid_to_hotkey_to_score:
            scores = [score for _, score in result.cid_to_hotkey_to_score[cid].items()]
            cid_to_average[cid] = np.average(scores)

        # prepare for spearman correlation
