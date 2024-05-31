import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import bittensor as bt
import numpy as np
import pandas as pd
import pingouin as pg
import torch
from attr import define, field
from loguru import logger

# Configure Loguru logger
logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="DEBUG")  # Add a new logger with the desired level
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from torch.nn import functional as F

from commons.dataset.leaderboard import get_gt_ranks, get_leaderboard_scores
from template.protocol import (
    CodeAnswer,
    CriteriaType,
    FeedbackRequest,
    MultiScoreCriteria,
    RankingCriteria,
    Response,
    ScoringMethod,
    ScoringResult,
    TaskType,
)


@define(kw_only=True, frozen=True, slots=True)
class Result:
    # Each request id has multiple completions, where each miner scores each of these completions.
    request_id: str
    cid_to_hotkey_to_score: Dict[str, Dict[str, float]] = field(factory=dict)


class GroundTruthScore(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    weighted_scores_by_miner: torch.Tensor
    raw_scores_by_miner: torch.Tensor


class ConsensusScore(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    weighted_score: torch.Tensor
    spearman_by_miner: torch.Tensor
    # cohen_kappa_by_miner: torch.Tensor
    icc_by_miner: torch.Tensor
    dist_penalty_by_miner: torch.Tensor


class Score(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    ground_truth: GroundTruthScore = Field(description="Raw score from ground truth")
    consensus: ConsensusScore = Field(description="Raw score from ground truth")
    weighted_consensus: Optional[torch.Tensor] = Field(
        description="Weighted score from consensus"
    )
    weighted_ground_truth: Optional[torch.Tensor] = Field(
        description="Weighted score from ground truth"
    )


class Scoring:
    @staticmethod
    def _spearman_scoring(responses: List[FeedbackRequest]):
        # map results...
        request_id = responses[0].request_id
        nested_dict: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for response in responses:
            hotkey = response.axon.hotkey
            for completion in response.responses:
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
            np.nan_to_num(spearmanr(miner_score, averages)[0])
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
                    response.responses, key=lambda x: model_id_to_avg_rank[x.model]
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

        # depending on the criteria, this may be average ranks or average scores
        if not len(responses):
            raise ValueError("Responses cannot be empty")

        # shape (num completions)
        avg = None
        # shape (num miners, num completions)
        miner_outputs = None
        icc_arr = []

        logger.info("Average scores: ", avg)
        logger.info("Miner outptus", miner_outputs)
        if isinstance(criteria, RankingCriteria):
            logger.debug("ranking criteria")
            avg, miner_outputs = Scoring._process_for_ranking(responses)
        elif isinstance(criteria, MultiScoreCriteria):
            logger.debug("got multi score criteria")
            # calculate average score per model
            model_id_to_scores = defaultdict(list)
            for response in responses:
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
                    for response in responses
                ]
            )

            avg: np.ndarray = np.array([v for k, v in model_id_to_avg_score.items()])
            logger.info(f"Average scores: {avg}")
            logger.info(f"Miner outptus {miner_outputs}")
            logger.info(f"Model id to avg {model_id_to_avg_score}")

            df = pd.DataFrame(
                {
                    "subject": [i for i in range(len(responses[0].responses))],
                }
            )
            # prepare dataframe for calculating ICC
            for response in responses:
                rater_id = response.axon.hotkey
                ordered_scores = [
                    x.score
                    for x in sorted(
                        response.responses, key=lambda x: model_id_to_avg_score[x.model]
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
                logger.debug("ICC raw")
                logger.debug(icc)

                # take ICC(2,1)
                icc2_value = icc[icc["Type"] == "ICC2"]["ICC"].iloc[0]
                icc_arr.append(icc2_value)
            icc_arr: np.ndarray = np.array(icc_arr)
            logger.info(f"ICC: {icc_arr}")

        else:
            raise NotImplementedError(
                f"Consensus score for type {criteria} not implemented yet"
            )

        if avg is None or miner_outputs is None:
            raise ValueError("avg and miner_outputs cannot be None")

        # spearman_corr = [
        #     scipy.stats.spearmanr(miner_output, avg).statistic
        #     for miner_output in miner_outputs
        # ]

        spearman = spearmanr(
            np.expand_dims(avg, axis=0) if len(avg.shape) == 1 else avg,
            miner_outputs,
            axis=1,
        )
        # spearman == 2D matrix, remove diagonal, grab first row for correlation between avg and each miner
        spearman_corr = spearman.statistic[0, 1:]
        logger.info(f"{spearman_corr=}")

        # this doesn't work for continuous data, use ICC instead
        # cohen_kappa = [
        #     cohen_kappa_score(miner_output, avg) for miner_output in miner_outputs
        # ]

        dist_penalty = -1 * torch.sum(
            torch.square(torch.tensor(avg - miner_outputs)), axis=1
        ).to(dtype=torch.float64)
        snorm = F.normalize(torch.tensor(spearman_corr), dim=0, p=2)
        # cknorm = F.normalize(torch.tensor(cohen_kappa), dim=0, p=2)
        icc_norm = F.normalize(torch.tensor(icc_arr), dim=0, p=2)
        dnorm = F.normalize(dist_penalty, dim=0, p=2)
        logger.debug(snorm)
        # bt.logging.debug(cknorm)
        logger.debug(icc_norm)
        logger.debug(dnorm)
        combined = F.softmax(snorm + icc_norm + 1.5 * dnorm, dim=0)
        logger.debug(combined)
        return ConsensusScore(
            weighted_score=combined,
            spearman_by_miner=snorm,
            # cohen_kappa_by_miner=cknorm,
            icc_by_miner=icc_norm,
            dist_penalty_by_miner=dnorm,
        )

    @staticmethod
    def cmp_ground_truth(
        criteria: CriteriaType,
        request: FeedbackRequest,
        responses: List[FeedbackRequest],
    ):
        if isinstance(criteria, RankingCriteria):
            # already sorting according to score
            model_score_tuples = get_leaderboard_scores(
                [completion.model for completion in request.responses]
            )
            # ensure we have the same ordering by model id
            model_ids_sorted = [model[0] for model in model_score_tuples]
            all_miner_ranks = [
                [
                    completion.rank_id
                    for completion in sorted(
                        response.responses,
                        key=lambda x: model_ids_sorted.index(x.model),
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
            return GroundTruthScore(
                weighted_scores_by_miner=diff_gt_sm,
                raw_scores_by_miner=torch.tensor(diff_gt),
            )

    @staticmethod
    def calculate_score(
        criteria_types: List[CriteriaType],
        request: FeedbackRequest,
        responses: List[FeedbackRequest],
    ) -> Dict[CriteriaType, Score]:
        """Combines both consensus score and difference with ground truths scoring to output a final score per miner"""
        criteria_to_miner_scores = defaultdict(Score)
        for criteria in criteria_types:
            gt_score = Scoring.cmp_ground_truth(criteria, request, responses)
            consensus_score = Scoring.consensus_score(criteria, responses)
            criteria_to_miner_scores[criteria] = Score(
                ground_truth=gt_score, consensus=consensus_score
            )
        return criteria_to_miner_scores


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


def prepare_responses():
    miner_a = FeedbackRequest(
        axon=bt.TerminalInfo(hotkey="hotkeyA"),
        prompt="Write a hello world program in python",
        task_type=TaskType.CODE_GENERATION,
        criteria_types=[
            MultiScoreCriteria(type="multi-score", options=[], min=0.0, max=100.0)
        ],
        responses=[
            Response(
                model="anthropic/claude-3-haiku-20240307",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=75,
            ),
            Response(
                model="anthropic/claude-3-opus-20240229",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=100,
            ),
            Response(
                model="anthropic/claude-3-sonnet-20240229",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=50,
            ),
            Response(
                model="meta-llama/llama-3-8b-instruct",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=69,
            ),
        ],
    )

    miner_b = FeedbackRequest(
        axon=bt.TerminalInfo(hotkey="hotkeyB"),
        prompt="Write a hello world program in python",
        task_type=TaskType.CODE_GENERATION,
        criteria_types=[
            MultiScoreCriteria(type="multi-score", options=[], min=0.0, max=100.0)
        ],
        responses=[
            Response(
                model="anthropic/claude-3-haiku-20240307",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=51,
            ),
            Response(
                model="anthropic/claude-3-opus-20240229",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=49,
            ),
            Response(
                model="anthropic/claude-3-sonnet-20240229",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=52,
            ),
            Response(
                model="meta-llama/llama-3-8b-instruct",
                completion=CodeAnswer(
                    code="print('hello, world!')",
                    language="python",
                    files=[],
                    additional_notes=None,
                    installation_commands="",
                ),
                score=53,
            ),
        ],
    )
    return [miner_a, miner_b]


if __name__ == "__main__":
    responses = prepare_responses()
    Scoring.consensus_score(responses[0].criteria_types[0], responses)
