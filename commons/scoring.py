import asyncio
import json
from collections import defaultdict
from typing import Dict, List

import bittensor as bt
import numpy as np
import scipy
from attr import define, field
import torch
from torch.nn import functional as F

from template.protocol import (
    CriteriaType,
    FeedbackRequest,
    RankingResult,
    ScoringMethod,
)


@define(kw_only=True, frozen=True, slots=True)
class Result:
    # Each request id has multiple completions, where each miner scores each of these completions.
    request_id: str
    cid_to_hotkey_to_score: Dict[str, Dict[str, float]] = field(factory=dict)


class Scoring:
    @staticmethod
    def _spearman_scoring(responses: List[FeedbackRequest]):
        # TODO refactor for future scoring use
        # map results...
        request_id = responses[0].request_id
        nested_dict: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for response in responses:
            hotkey = response.axon.hotkey
            for completion in response.completions:
                nested_dict[completion.cid][hotkey] = completion.score

        data = json.loads(json.dumps(nested_dict))
        result = Result(request_id=request_id, cid_to_hotkey_to_score=data)
        cid_to_average = {
            cid: np.mean(list(hotkey_scores.values()))
            for cid, hotkey_scores in result.cid_to_hotkey_to_score.items()
        }
        averages = list(cid_to_average.values())

        # shape (num miners, num completions)
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
        spearman_corr = [
            np.nan_to_num(scipy.stats.spearmanr(miner_score, averages)[0])
            for miner_score in miner_scores
        ]

        for i, response in enumerate(responses):
            if not response.scoring_method:
                bt.logging.error(
                    "Scoring method not set... defaulting to normal weights"
                )
                continue

            if response.scoring_method == ScoringMethod.AWS_MTURK:
                spearman_corr[i] *= 1.2

        hotkeys = [r.axon.hotkey for r in responses]

        # scale values in the range [-1, 1] to [0, 1]
        spearman_corr = 0.5 * (np.array(spearman_corr) + 1)
        hotkey_to_score = dict(zip(hotkeys, spearman_corr.tolist()))

        scores = torch.tensor(list(hotkey_to_score.values()))
        # ensure sum == 1
        scores = F.softmax(scores, dim=0)
        hotkey_to_adjusted_score = dict(zip(hotkey_to_score.keys(), scores))
        # store in synapse to be forwarded to miners
        ranking_result = RankingResult(
            request_id=responses[0].request_id,
            cid_to_consensus=cid_to_average,
            hotkey_to_score=hotkey_to_adjusted_score,
        )
        return ranking_result

    @staticmethod
    def consensus_score(criteria: CriteriaType, responses: List[FeedbackRequest]):
        """Given a list of responses, will only return a dict of hotkey to their normalized scores.
        e.g. if a miner failed to respond, its hotkey won't be a key in the dict.
        """
        if not len(responses):
            raise ValueError("Responses cannot be empty")

        if criteria == CriteriaType.SCORING:
            return Scoring._spearman_scoring(responses)


def _calculate_average_rank_by_model(
    responses: List[FeedbackRequest],
) -> Dict[str, float]:
    model_id_to_average_rank = defaultdict(list)
    for request in responses:
        for completion in request.completions:
            # if completion.model_id not in model_id_to_average_rank:
            #     model_id_to_average_rank[completion.model_id] = []
            model_id_to_average_rank[completion.model_id].append(completion.rank_id)

    for model_id, ranks in model_id_to_average_rank.items():
        model_id_to_average_rank[model_id] = sum(ranks) / len(ranks)

    sorted_dict = dict(
        sorted(model_id_to_average_rank.items(), key=lambda item: item[1])
    )
    return sorted_dict


def calculate_average_rank_by_model(
    responses: List[FeedbackRequest], metagraph: bt.metagraph
):
    sorted_dict = _calculate_average_rank_by_model(responses)
    has_conflicting_values = len(set(sorted_dict.values())) != len(sorted_dict.keys())
    # TODO handle this edge case
    # if has_conflicting_values:
    return sorted_dict


async def main():
    from template.protocol import Completion, CriteriaType, TaskType
    import random

    test_requests = []
    for i in range(1, 11):
        ranks = list(range(1, 5))
        random.shuffle(ranks)
        test_requests.append(
            FeedbackRequest(
                request_id=f"req{i}",
                prompt=f"Prompt for request {i}",
                completions=[
                    Completion(
                        cid=f"c{i}1",
                        model_id="m1",
                        text=f"Text {i}1",
                        rank_id=ranks[0],
                        code="",
                        language="deez",
                        installation_commands="",
                    ),
                    Completion(
                        cid=f"c{i}2",
                        model_id="m2",
                        text=f"Text {i}2",
                        rank_id=ranks[1],
                        code="",
                        language="deez",
                        installation_commands="",
                    ),
                    Completion(
                        cid=f"c{i}3",
                        model_id="m3",
                        text=f"Text {i}3",
                        rank_id=ranks[2],
                        code="",
                        language="deez",
                        installation_commands="",
                    ),
                    Completion(
                        cid=f"c{i}4",
                        model_id="m4",
                        text=f"Text {i}4",
                        rank_id=ranks[3],
                        code="",
                        language="deez",
                        installation_commands="",
                    ),
                ],
                scoring_method="dojo_worker",
                task_type=TaskType.CODE_GENERATION,
                criteria_types=[CriteriaType.PREFERENCE_RANKING],
            )
        )
    print(_calculate_average_rank_by_model(test_requests))


if __name__ == "__main__":
    asyncio.run(main())
    print("Done!")
