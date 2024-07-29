from unittest.mock import patch

import bittensor as bt
import numpy as np
import pytest


@pytest.fixture
def scoring_module(mock_env_var):
    # ensure we import them depending on mock_env_var so the ValueError doesn't
    # get raised
    from commons.scoring import Scoring
    from template.protocol import (
        CodeAnswer,
        FeedbackRequest,
        MultiScoreCriteria,
        Response,
        TaskType,
    )

    return (
        Scoring,
        FeedbackRequest,
        CodeAnswer,
        MultiScoreCriteria,
        Response,
        TaskType,
    )


def mock_response(
    model: str,
    score: float | None,
    filename: str,
    content: str,
    language: str,
):
    from template.protocol import CodeAnswer, FileObject, Response

    return Response(
        model=model,
        completion=CodeAnswer(
            files=[FileObject(filename=filename, content=content, language=language)],
            additional_notes=None,
            installation_commands="",
        ),
        score=score,
    )


def mock_request(hotkey: str | None = None, scores: list[float] | None = None):
    from template.protocol import FeedbackRequest, MultiScoreCriteria, TaskType

    axon = bt.TerminalInfo(hotkey=hotkey)
    prompt = "Write a hello world program in python"
    task_type = TaskType.CODE_GENERATION
    models = [
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "meta-llama/llama-3-8b-instruct",
    ]

    responses = [
        mock_response(
            model=model,
            score=score,
            filename="hello_world.py",
            content="print('hello, world!')",
            language="python",
        )
        for model, score in zip(
            models, scores if scores is not None else [None] * len(models)
        )
    ]

    return FeedbackRequest(
        axon=axon,
        prompt=prompt,
        task_type=task_type,
        criteria_types=[
            MultiScoreCriteria(type="multi-score", options=[], min=0.0, max=100.0)
        ],
        responses=responses,
    )


def mock_scoring_data_normal() -> tuple:
    request = mock_request()
    miner_a = mock_request(hotkey="hotkeyA", scores=[75, 100, 50, 69])
    miner_b = mock_request(hotkey="hotkeyB", scores=[51, 49, 52, 53])
    return request, [miner_a, miner_b]


def mock_scoring_data_all_same_scores() -> tuple:
    request = mock_request()
    miner_a = mock_request(hotkey="hotkeyA", scores=[50, 50, 50, 50])
    miner_b = mock_request(hotkey="hotkeyB", scores=[50, 50, 50, 50])
    return request, [miner_a, miner_b]


def test_consensus_normal_data():
    from commons.scoring import ConsensusScore, Scoring

    test_data = mock_scoring_data_normal()
    request, miner_responses = test_data
    for criteria in request.criteria_types:
        score: ConsensusScore = Scoring.consensus_score(
            criteria, request, miner_responses
        )

        assert score is not None, "score should not be None"
        assert not np.isnan(
            score.score
        ).any(), "overall score does not contain NaN values"
        assert not np.isinf(
            score.score
        ).any(), "overall score does not contain inf values"
        assert np.count_nonzero(score.mse_by_miner) != 0, "MSE is not all zeros"
        assert not np.isnan(
            score.icc_by_miner
        ).any(), "ICC does not contain any NaN values"
        assert not np.isinf(
            score.icc_by_miner
        ).any(), "ICC does not contain any inf values"


def test_consensus_same_scores():
    """Used to test that both miners have provided the same scores"""
    from commons.scoring import ConsensusScore, Scoring

    test_data = mock_scoring_data_all_same_scores()
    request, miner_responses = test_data
    score: ConsensusScore = Scoring.consensus_score(
        request.criteria_types[0], request, miner_responses
    )

    assert score is not None, "score should not be None"
    assert not np.isnan(score.score).any(), "overall score does not contain NaN values"
    assert not np.isinf(score.score).any(), "overall score does not contain inf values"
    assert (
        np.count_nonzero(score.mse_by_miner) == 0
    ), "MSE is all zeros since miners provide the same score"
    assert np.isnan(
        score.icc_by_miner
    ).any(), "ICC should contain NaN values for when there is zero variance between miners ratings"


@patch("commons.scoring.get_leaderboard_scores")
def test_ground_truth_leaderboard_data_normal(mock_get_leaderboard_scores):
    from commons.scoring import Scoring

    mock_scores = [
        ("anthropic/claude-3-haiku-20240307", 68.9),
        ("anthropic/claude-3-opus-20240229", 77.4),
        ("anthropic/claude-3-sonnet-20240229", 64.0),
        ("meta-llama/llama-3-8b-instruct", 56.7),
    ]
    mock_get_leaderboard_scores.return_value = mock_scores

    test_data = mock_scoring_data_normal()
    request, miner_responses = test_data

    for criteria in request.criteria_types:
        gt_score = Scoring.cmp_ground_truth(criteria, request, miner_responses)
        assert gt_score is not None

        mock_get_leaderboard_scores.assert_called_once_with(
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-opus-20240229",
                "anthropic/claude-3-sonnet-20240229",
                "meta-llama/llama-3-8b-instruct",
            ]
        )

        assert not np.isnan(
            gt_score.score
        ).any(), "overall score does not contain NaN values"
        assert not np.isinf(
            gt_score.score
        ).any(), "overall score does not contain inf values"
        assert not np.isnan(
            gt_score.raw_scores_by_miner
        ).any(), "overall score does not contain NaN values"
        assert not np.isinf(
            gt_score.raw_scores_by_miner
        ).any(), "overall score does not contain inf values"


@pytest.mark.skip(reason="Placeholder test, not implemented yet")
def test_ground_truth_leaderboard_data_missing():
    pass


@pytest.mark.skip(reason="Placeholder test, not implemented yet")
def test_ground_truth_state_missing():
    pass
