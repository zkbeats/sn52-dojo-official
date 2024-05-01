import asyncio
import json
from collections import defaultdict
from typing import Dict, List

import bittensor as bt
import numpy as np
from pydantic import validate_arguments
import scipy
from attr import define, field
import torch
from torch.nn import functional as F
from sklearn.metrics import cohen_kappa_score
from commons.dataset.leaderboard import diff_gt, get_gt_ranks, get_leaderboard_scores
from commons.dataset.mock import MockData

from template.protocol import (
    CriteriaType,
    FeedbackRequest,
    ScoringResult,
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
        ranking_result = ScoringResult(
            request_id=responses[0].request_id,
            cid_to_consensus=cid_to_average,
            hotkey_to_scores=hotkey_to_adjusted_score,
        )
        return ranking_result

    @staticmethod
    def _process_for_ranking(responses: List[FeedbackRequest]):
        model_id_to_avg_rank = _calculate_average_rank_by_model(responses)
        # shape (num miners, num completions)
        # order ranks based on their order in the sorted dict
        all_miner_ranks = [
            [
                x.rank_id
                for x in sorted(
                    response.completions, key=lambda x: model_id_to_avg_rank[x.model_id]
                )
            ]
            for response in responses
        ]
        all_miner_ranks = np.array(all_miner_ranks)
        bt.logging.trace(all_miner_ranks.shape)

        # compute spearman correlation and handle nan values
        avg_ranks = np.array([i + 1 for i in range(len(model_id_to_avg_rank.keys()))])
        return avg_ranks, all_miner_ranks

    @staticmethod
    def consensus_score(criteria: CriteriaType, responses: List[FeedbackRequest]):
        """Given a list of responses, will only return a dict of hotkey to their normalized scores.
        e.g. if a miner failed to respond, its hotkey won't be a key in the dict.
        """
        if not len(responses):
            raise ValueError("Responses cannot be empty")

        if criteria == CriteriaType.SCORING:
            # TODO refactor to return tensor
            return Scoring._spearman_scoring(responses)
        elif criteria == CriteriaType.PREFERENCE_RANKING:
            avg_ranks, all_miner_ranks = Scoring._process_for_ranking(responses)
            spearman_corr = [
                scipy.stats.spearmanr(miner_ranks, avg_ranks).statistic
                for miner_ranks in all_miner_ranks
            ]
            cohen_kappa = [
                cohen_kappa_score(miner_ranks, avg_ranks)
                for miner_ranks in all_miner_ranks
            ]
            dist_penalty = -1 * torch.sum(
                torch.square(torch.tensor(avg_ranks - all_miner_ranks)), axis=1
            ).to(dtype=torch.float64)
            snorm = F.normalize(torch.tensor(spearman_corr), dim=0, p=2)
            cknorm = F.normalize(torch.tensor(cohen_kappa), dim=0, p=2)
            dnorm = F.normalize(dist_penalty, dim=0, p=2)
            bt.logging.trace(snorm)
            bt.logging.trace(cknorm)
            bt.logging.trace(dnorm)
            combined_sm = F.softmax(snorm + cknorm + 1.5 * dnorm, dim=0)
            bt.logging.trace(f"{combined_sm}")
            return combined_sm

    @staticmethod
    def cmp_ground_truth(
        criteria: CriteriaType,
        request: FeedbackRequest,
        responses: List[FeedbackRequest],
    ):
        if criteria == CriteriaType.SCORING:
            raise NotImplementedError("Not implemented yet")
        elif criteria == CriteriaType.PREFERENCE_RANKING:
            # already sorting according to score
            model_score_tuples = get_leaderboard_scores(
                [completion.model_id for completion in request.completions]
            )
            # ensure we have the same ordering by model id
            model_ids_sorted = [model[0] for model in model_score_tuples]
            all_miner_ranks = [
                [
                    completion.rank_id
                    for completion in sorted(
                        response.completions,
                        key=lambda x: model_ids_sorted.index(x.model_id),
                    )
                ]
                for response in responses
            ]

            gt_ranks = get_gt_ranks(models_with_scores=model_score_tuples)
            bt.logging.trace(f"{gt_ranks=}")
            """Calculate difference between a miner's ranks and the ground truth"""
            if isinstance(all_miner_ranks, list):
                all_miner_ranks = np.array(all_miner_ranks)
            if isinstance(gt_ranks, list):
                ground_truth_ranks = np.array(gt_ranks)

            diff_gt = -1 * np.linalg.norm(
                all_miner_ranks - ground_truth_ranks, ord=2, axis=1
            )
            bt.logging.trace(f"{diff_gt=}")
            diff_gt_sm = F.softmax(torch.tensor(diff_gt), dim=0)
            bt.logging.trace(f"{diff_gt_sm=}")
            return diff_gt_sm

    @staticmethod
    @validate_arguments
    def calculate_score(
        criteria_types: List[CriteriaType],
        request: FeedbackRequest,
        responses: List[FeedbackRequest],
    ) -> Dict[CriteriaType, torch.Tensor]:
        """Combines both consensus score and difference with ground truths scoring to output a final score per miner"""
        criteria_to_miner_scores = defaultdict(list)
        for criteria in criteria_types:
            gt_score = Scoring.cmp_ground_truth(criteria, request, responses)
            consensus_score = Scoring.consensus_score(criteria, responses)
            print(f"{gt_score}")
            print(f"{consensus_score}")
            criteria_to_miner_scores[criteria] = torch.tensor(
                0.65 * gt_score + 0.35 * consensus_score
            )
        return criteria_to_miner_scores


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


if __name__ == "__main__":
    test_requests = MockData.generate_test_data()[:5]
    for tr in test_requests:
        for completion in tr.completions:
            print((completion.model_id, completion.rank_id), end=" ")
        print()

    print(
        Scoring.cmp_ground_truth(
            CriteriaType.PREFERENCE_RANKING, test_requests[0], test_requests
        )
    )

    print(Scoring.consensus_score(CriteriaType.PREFERENCE_RANKING, test_requests))
    res = Scoring.calculate_score(
        [CriteriaType.PREFERENCE_RANKING], test_requests[0], test_requests
    )
    print(res.get(CriteriaType.PREFERENCE_RANKING).shape)
    res["scoring"] = res.get(CriteriaType.PREFERENCE_RANKING).clone().detach()
    # averages
    total_sum = torch.stack([miner_scores for miner_scores in res.values()])
    print(total_sum.shape)
    print(total_sum)
    print(total_sum.mean(dim=0).shape)
    print(total_sum.mean(dim=0))
    bt.logging.info(f"Total sum of scores across all criteria: {total_sum}")
    print(res)
