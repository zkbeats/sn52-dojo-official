from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pingouin as pg
import torch
from attr import define, field
from loguru import logger
from pydantic import BaseModel, Field
from torch.nn import functional as F

from commons.dataset.leaderboard import get_leaderboard_scores
from template.protocol import (
    CriteriaType,
    FeedbackRequest,
    MultiScoreCriteria,
    RankingCriteria,
    Response,
)


@define(kw_only=True, frozen=True, slots=True)
class Result:
    # Each request id has multiple completions, where each miner scores each of these completions.
    request_id: str
    cid_to_hotkey_to_score: Dict[str, Dict[str, float]] = field(factory=dict)


class GroundTruthScore(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    score: torch.Tensor
    raw_scores_by_miner: torch.Tensor


class ConsensusScore(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    score: torch.Tensor
    mse_by_miner: torch.Tensor
    icc_by_miner: torch.Tensor


class Score(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    ground_truth: GroundTruthScore = Field(description="Raw score from ground truth")
    consensus: ConsensusScore = Field(description="Raw score from ground truth")
    weighted_consensus: torch.Tensor | None = Field(
        default=None, description="Weighted score from consensus"
    )
    weighted_ground_truth: torch.Tensor | None = Field(
        default=None, description="Weighted score from ground truth"
    )


def _get_miner_response_by_criteria(criteria, response: Response):
    if isinstance(criteria, RankingCriteria):
        return response.rank_id
    elif isinstance(criteria, MultiScoreCriteria):
        return response.score


def _get_ground_truth_by_criteria(criteria, model_with_score_sorted):
    gt = []
    if isinstance(criteria, RankingCriteria):
        gt = [i + 1 for i in range(len(model_with_score_sorted))]
    elif isinstance(criteria, MultiScoreCriteria):
        gt = [score for _, score in model_with_score_sorted]
    return np.array(gt)


class Scoring:
    @staticmethod
    def consensus_score(
        criteria: CriteriaType,
        request: FeedbackRequest,
        miner_responses: List[FeedbackRequest],
    ):
        """Given a list of responses, will only return a dict of hotkey to their normalized scores.
        e.g. if a miner failed to respond, its hotkey won't be a key in the dict.
        """

        # depending on the criteria, this may be average ranks or average scores
        if not len(miner_responses):
            raise ValueError("Responses cannot be empty")

        # shape (num completions)
        avg = None
        # shape (num miners, num completions)
        miner_outputs = None
        icc_arr = []

        # for ordering based on criteria
        model_id_to_avg_rank = defaultdict(list)
        model_id_to_scores = defaultdict(list)

        if isinstance(criteria, RankingCriteria):
            logger.debug("consensus scoring for ranking criteria")
            for response in miner_responses:
                for completion in response.responses:
                    # if completion.model_id not in model_id_to_average_rank:
                    #     model_id_to_average_rank[completion.model_id] = []
                    model_id_to_avg_rank[completion.model].append(completion.rank_id)

            for model_id, ranks in model_id_to_avg_rank.items():
                model_id_to_avg_rank[model_id] = sum(ranks) / len(ranks)

            model_id_to_avg_rank = dict(
                sorted(model_id_to_avg_rank.items(), key=lambda item: item[1])
            )

            # shape (num miners, num completions)
            # order ranks based on their order in the sorted dict
            miner_outputs = [
                [
                    _get_miner_response_by_criteria(criteria, x)
                    for x in sorted(
                        response.responses, key=lambda x: model_id_to_avg_rank[x.model]
                    )
                ]
                for response in miner_responses
            ]
            miner_outputs = np.array(miner_outputs)
            avg = np.array([i + 1 for i in range(len(model_id_to_avg_rank.keys()))])

        elif isinstance(criteria, MultiScoreCriteria):
            logger.debug("consensus scoring for multi-score criteria")
            # calculate average score per model
            for response in miner_responses:
                for completion in response.responses:
                    model_id_to_scores[completion.model].append(completion.score)
            # for each model calculate the average score
            # USE DICT BECAUSE WE NEED TO ENSURE CORRECT ORDERING
            model_id_to_avg_score = {
                model: sum(scores) / len(scores)
                for model, scores in model_id_to_scores.items()
            }

            # shape (num miners, num completions)
            # collect all scores from each miner based on ordering in model_id_avg_score
            miner_outputs = np.array(
                [
                    [
                        completion.score
                        for completion in sorted(
                            response.responses,
                            key=lambda x: model_id_to_avg_score[x.model],
                        )
                    ]
                    for response in miner_responses
                ]
            )

            avg: np.ndarray = np.array([v for k, v in model_id_to_avg_score.items()])

        else:
            raise NotImplementedError(
                f"Consensus score for type {criteria} not implemented yet"
            )

        if avg is None or miner_outputs is None:
            raise ValueError("avg and miner_outputs cannot be None")

        logger.info(f"Average across all miners: {avg}")
        logger.info(f"Miner outputs {miner_outputs}")
        logger.info(f"Model id to avg {model_id_to_avg_score}")

        # create df with the original number of completions
        df = pd.DataFrame(
            {
                "subject": [i for i in range(len(request.responses))],
            }
        )
        # prepare dataframe for calculating ICC
        for response in miner_responses:
            rater_id = response.axon.hotkey
            ordered_scores = [
                x.score
                for x in sorted(
                    response.responses,
                    key=lambda x: model_id_to_avg_score[x.model]
                    if criteria == MultiScoreCriteria
                    else model_id_to_avg_rank[x.model],
                )
            ]
            # order scores based on order in model_id_to_avg_score
            df[rater_id] = ordered_scores
        rater_ids = list(df.columns)
        rater_ids.remove("subject")
        df["avg"] = df[rater_ids].mean(axis=1)

        # this works because we are calculating ICC for each rater VS the avg
        for rater_id in rater_ids:
            data_by_rater = df[["subject", rater_id, "avg"]]
            # only use the columns for the current rater and avg
            data_by_rater = data_by_rater.melt(
                id_vars=["subject"], var_name=rater_id, value_name="score"
            )
            icc = pg.intraclass_corr(
                data=data_by_rater,
                targets="subject",
                raters=rater_id,
                ratings="score",
            )

            # take ICC(2,1)
            icc2_value = icc[icc["Type"] == "ICC2"]["ICC"].iloc[0]
            icc_arr.append(icc2_value)
        # already in the range [0, 1]
        icc_arr: torch.Tensor = torch.tensor(np.array(icc_arr))

        # only use this for ordinal data
        # spearman = np.array(
        #     [
        #         spearmanr(miner_output, avg, nan_policy="propagate").statistic
        #         for miner_output in miner_outputs
        #     ]
        # )
        # num_nans = np.sum(np.isnan(spearman))

        mse = torch.tensor(np.mean(np.abs(miner_outputs - avg) ** 2, axis=1))
        logger.debug(f"MSE raw: {mse}")
        logger.info(f"ICC raw: {icc_arr}")
        if not np.isnan(icc_arr).any():
            return ConsensusScore(
                score=torch.tensor(icc_arr),
                mse_by_miner=mse,
                icc_by_miner=icc_arr,
            )

        logger.warning("ICC array contains NaN values, using just MSE instead")

        # use negative sign to penalize higher mse
        mse_reward = F.softmax(-1 * mse, dim=0)

        # # edge case where all miners provide the same rating
        # if torch.all(mse_reward_norm == 0):
        #     logger.warning("MSE reward normalization resulted in all zeros.")
        #     reward_per_miner = 1 / len(miner_outputs)
        #     mse_reward_norm = torch.full_like(mse_reward_norm, reward_per_miner)

        logger.debug(f"MSE reward: {mse_reward}")
        logger.debug(f"MSE normalized: {mse_reward}")

        return ConsensusScore(
            score=mse_reward,
            mse_by_miner=mse,
            icc_by_miner=icc_arr,
        )

    @staticmethod
    def cmp_ground_truth(
        criteria: CriteriaType,
        request: FeedbackRequest,
        miner_responses: List[FeedbackRequest],
    ):
        # determine the ground truth ordering based on request
        model_score_tuples = get_leaderboard_scores(
            [completion.model for completion in request.responses]
        )
        model_with_score_sorted = sorted(
            model_score_tuples, key=lambda x: (x[1] is not None, x[1]), reverse=True
        )
        model_ids_sorted = [model[0] for model in model_with_score_sorted]

        # sort miner outputs according to ground truth order
        # this may be scores or ranks
        # log miner models to check
        miner_models = []
        for r in miner_responses:
            for completion in r.responses:
                miner_models.append(completion.model)

        miner_outputs = []
        for response in miner_responses:
            curr_miner_outputs = []
            for completion in sorted(
                response.responses, key=lambda r: model_ids_sorted.index(r.model)
            ):
                curr_miner_outputs.append(
                    _get_miner_response_by_criteria(criteria, completion)
                )
            miner_outputs.append(curr_miner_outputs)
        if miner_outputs == [] or None in miner_outputs:
            raise ValueError("Miner outputs cannot be empty or contain None values")

        miner_outputs = np.array(miner_outputs)

        # this may be scores or ranks
        ground_truth = _get_ground_truth_by_criteria(criteria, model_with_score_sorted)

        logger.info(f"Miner outputs: {miner_outputs}")
        logger.info(f"Ground truth: {ground_truth}")

        diff_gt = torch.tensor(
            -1 * np.linalg.norm(miner_outputs - ground_truth, ord=2, axis=1)
        )
        logger.debug(f"{diff_gt=}")
        gt_reward = F.softmax(diff_gt, dim=0)
        logger.debug(f"{gt_reward=}")

        return GroundTruthScore(
            score=torch.tensor(gt_reward),
            raw_scores_by_miner=diff_gt,
        )

    @staticmethod
    def calculate_score(
        criteria_types: List[CriteriaType],
        request: FeedbackRequest,
        miner_responses: List[FeedbackRequest],
    ) -> Dict[CriteriaType, Score]:
        """Combines both consensus score and difference with ground truths scoring to output a final score per miner"""
        criteria_to_miner_scores = defaultdict(Score)
        hotkey_to_final_score = defaultdict(float)
        for criteria in criteria_types:
            # valid responses
            valid_miner_responses = []
            for response in miner_responses:
                values = [
                    _get_miner_response_by_criteria(criteria, completion)
                    for completion in response.responses
                ]
                if any(v is None for v in values):
                    logger.error(
                        f"Detected None values in response for request id: {request.request_id} from miner: {response.axon.hotkey}"
                    )
                    continue
                if isinstance(criteria, MultiScoreCriteria):
                    default_value = (5 / 10) * (
                        criteria.max - criteria.min
                    ) + criteria.min
                    if all(v == default_value for v in values):
                        logger.error(
                            f"Detected all default values in response for request id: {request.request_id} from miner: {response.axon.hotkey}"
                        )
                        continue

                valid_miner_responses.append(response)

            if len(valid_miner_responses) < 2:
                logger.warning(
                    f"Skipping scoring for request id: {request.request_id} as not enough valid responses"
                )
                for r in valid_miner_responses:
                    hotkey_to_final_score[r.axon.hotkey] = 0.0

                continue

            gt_score = Scoring.cmp_ground_truth(
                criteria, request, valid_miner_responses
            )
            consensus_score = Scoring.consensus_score(
                criteria, request, valid_miner_responses
            )

            for i, r in enumerate(valid_miner_responses):
                consensus = 0.6 * consensus_score.score[i]
                ground_truth = 0.4 * gt_score.score[i]

                hotkey_to_final_score[r.axon.hotkey] = (consensus + ground_truth) / len(
                    criteria_types
                )

            criteria_to_miner_scores[criteria.type] = Score(
                ground_truth=gt_score, consensus=consensus_score
            )
        return criteria_to_miner_scores, hotkey_to_final_score


def _calculate_average_rank_by_model(
    responses: List[FeedbackRequest],
) -> Dict[str, float]:
    model_id_to_average_rank = defaultdict(list)
    for request in responses:
        for completion in request.responses:
            # if completion.model_id not in model_id_to_average_rank:
            #     model_id_to_average_rank[completion.model_id] = []
            model_id_to_average_rank[completion.model].append(completion.rank_id)

    for model_id, ranks in model_id_to_average_rank.items():
        model_id_to_average_rank[model_id] = sum(ranks) / len(ranks)

    sorted_dict = dict(
        sorted(model_id_to_average_rank.items(), key=lambda item: item[1])
    )
    return sorted_dict
