from unittest.mock import patch

import bittensor as bt
import numpy as np
import pytest
import torch

from template.protocol import FeedbackRequest, RankingCriteria, TaskType

# # Remove the default loguru handler
# logger.remove()

# # Add a new handler to log to stdout
# logger.add(sys.stdout, level="DEBUG")

default_ground_truth = {
    "cid_1": 0,  # 1st place
    "cid_2": 1,  # 2nd place
    "cid_3": 2,  # 3rd place
    "cid_4": 3,  # 4th place
}


@pytest.fixture
def scoring_module():
    # ensure we import them depending on mock_env_var so the ValueError doesn't
    # get raised
    from commons.scoring import Scoring
    from template.protocol import CodeAnswer, MultiScoreCriteria, Response

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
    filename: str,
    content: str,
    language: str,
    score: float | None = 0.0,
    cid: str = "",
    rank_id: int = 0,
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
        rank_id=rank_id,
        cid=cid,
    )


def mock_request(hotkey: str | None = None, scores: list[float] | None = None):
    from template.protocol import MultiScoreCriteria

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

    # Include the ground truth in the request object if provided
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
def test_ground_truth_state_missing():
    pass


@patch("commons.dataset.leaderboard.get_leaderboard_data")
def test_cmp_ground_truth_missing_data(mock_get_leaderboard_data_func):
    from commons.scoring import Scoring

    # mock leaderboard data, purposely omit llama 8b which is inside `mock_request`
    mock_leaderboard_data = {
        "claude-2 (Mar 2024)": {
            "link": "https://www.anthropic.com/news/claude-2",
            "open-data": "NONE",
            "pass@1": {
                "humaneval": 69.5,
                "humaneval+": 61.6,
                "mbpp": None,
                "mbpp+": None,
            },
            "prompted": True,
            "size": None,
        },
        "claude-3-haiku (Mar 2024)": {
            "link": "https://www.anthropic.com/news/claude-3-family",
            "open-data": "NONE",
            "pass@1": {
                "humaneval": 76.8,
                "humaneval+": 68.9,
                "mbpp": 80.2,
                "mbpp+": 68.8,
            },
            "prompted": True,
            "size": None,
        },
        "claude-3-opus (Mar 2024)": {
            "link": "https://www.anthropic.com/news/claude-3-family",
            "open-data": "NONE",
            "pass@1": {
                "humaneval": 82.9,
                "humaneval+": 77.4,
                "mbpp": 89.4,
                "mbpp+": 73.3,
            },
            "prompted": True,
            "size": None,
        },
        "claude-3-sonnet (Mar 2024)": {
            "link": "https://www.anthropic.com/news/claude-3-family",
            "open-data": "NONE",
            "pass@1": {
                "humaneval": 70.7,
                "humaneval+": 64,
                "mbpp": 83.6,
                "mbpp+": 69.3,
            },
            "prompted": True,
            "size": None,
        },
    }
    mock_get_leaderboard_data_func.return_value = mock_leaderboard_data

    request, miner_responses = mock_scoring_data_normal()

    for criteria in request.criteria_types:
        # test that it raises a ValueError when data is missing
        with pytest.raises(ValueError, match=".*cannot contain None values.*"):
            Scoring.cmp_ground_truth(criteria, request, miner_responses)

    mock_get_leaderboard_data_func.assert_called_once()


"""
TESTS FOR SPEARMAN CORRELATION
"""


def mock_request_spm(
    hotkey: str | None = None,
    rank_ids: list[int] = [],
    cids: list[str] = [],
    ground_truth: dict[str, int] = default_ground_truth,
):
    """
    Dynamically generates miner responses using separate rank_ids and cids.
    """
    from template.protocol import FeedbackRequest, TaskType

    axon = bt.TerminalInfo(hotkey=hotkey)
    prompt = "Write a hello world program in python"
    task_type = TaskType.CODE_GENERATION

    # List of models for testing purposes (you can adjust this as necessary)
    models = [
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "meta-llama/llama-3-8b-instruct",
    ]

    # Ensure both rank_ids and cids have the same length
    assert len(rank_ids) == len(cids), "rank_ids and cids must have the same length."

    # Create responses dynamically using rank_ids and cids
    responses = [
        mock_response(
            model=models[i],
            score=None,
            cid=cid,
            filename=f"{models[i]}_output.py",
            content="print('hello')",
            language="python",
            rank_id=rank_id,
        )
        for i, (cid, rank_id) in enumerate(zip(cids, rank_ids))
    ]

    # Create and return FeedbackRequest with the dynamically generated responses
    return FeedbackRequest(
        axon=axon,
        prompt=prompt,
        task_type=task_type,
        criteria_types=[RankingCriteria(type="rank", options=[], min=0, max=3)],
        responses=responses,
        ground_truth=ground_truth,
    )


def mock_scoring_data_for_spm() -> tuple:
    ground_truth = {
        "cid_1": 0,  # 1st place
        "cid_2": 1,  # 2nd place
        "cid_3": 2,  # 3rd place
        "cid_4": 3,  # 4th place
    }

    # Pass separate rank_ids and cids for miner A and miner B
    request = mock_request_spm(
        ground_truth=ground_truth,
        rank_ids=[2, 1, 0, 3],
        cids=["cid_1", "cid_2", "cid_3", "cid_4"],
    )
    miner_a = mock_request_spm(
        hotkey="hotkeyA",
        rank_ids=[2, 1, 0, 3],
        cids=["cid_1", "cid_2", "cid_3", "cid_4"],
    )
    miner_b = mock_request_spm(
        hotkey="hotkeyB",
        rank_ids=[0, 1, 2, 3],
        cids=["cid_1", "cid_2", "cid_3", "cid_4"],
    )

    return request, [miner_a, miner_b]


def mock_scoring_data_with_known_values() -> tuple:
    """
    This mock data has specific values where we can predict the Spearman correlation.
    """

    ground_truth = {
        "cid1": 0,  # Best-ranked item
        "cid2": 1,
        "cid3": 2,
        "cid4": 3,  # Worst-ranked item
    }

    # Miner A ranks the items perfectly in reverse order of ground truth
    miner_a = mock_request_spm(
        hotkey="hotkeyA",
        rank_ids=[3, 2, 1, 0],  # Reverse of ground truth
        cids=["cid1", "cid2", "cid3", "cid4"],
        ground_truth=ground_truth,
    )

    # Miner B ranks the items in the exact order of the ground truth
    miner_b = mock_request_spm(
        hotkey="hotkeyB",
        rank_ids=[0, 1, 2, 3],  # Perfect match with ground truth
        cids=["cid1", "cid2", "cid3", "cid4"],
        ground_truth=ground_truth,
    )

    request = mock_request_spm(ground_truth=ground_truth)
    return request, [miner_a, miner_b]


def test_spearman_correlation(scoring_module):
    from commons.scoring import Scoring

    request, miner_responses = mock_scoring_data_for_spm()

    for criteria in request.criteria_types:
        spearman_score = Scoring.spm_ground_truth(criteria, request, miner_responses)

        # Ensure no NaN values
        assert not np.isnan(
            spearman_score.score
        ).any(), "Spearman score should not contain NaN values"

        # Ensure no inf values
        assert not np.isinf(
            spearman_score.score
        ).any(), "Spearman score should not contain inf values"

        # Check if Spearman scores are valid between -1 and 1
        assert torch.all(
            (spearman_score.raw_scores_by_miner >= -1)
            & (spearman_score.raw_scores_by_miner <= 1)
        )


def test_spearman_correlation_known_values(scoring_module):
    from scipy.stats import spearmanr

    from commons.scoring import Scoring

    request, miner_responses = mock_scoring_data_with_known_values()

    for criteria in request.criteria_types:
        spearman_score = Scoring.spm_ground_truth(criteria, request, miner_responses)

        # Expected values:
        # Miner A has the worst possible Spearman correlation (-1.0)
        # Miner B has the best possible Spearman correlation (1.0)

        # Test Miner A and Miner B's raw Spearman scores
        miner_a_spearman = spearmanr([3, 2, 1, 0], [0, 1, 2, 3]).correlation  # -1.0
        miner_b_spearman = spearmanr([0, 1, 2, 3], [0, 1, 2, 3]).correlation  # 1.0

        # Ensure that the raw scores are cast to float32 before comparison
        assert torch.allclose(
            spearman_score.raw_scores_by_miner[0].float(),  # Cast to float32
            torch.tensor(miner_a_spearman, dtype=torch.float32),
            atol=1e-6,
        ), f"Expected Spearman score for Miner A: {miner_a_spearman}, got: {spearman_score.raw_scores_by_miner[0]}"

        assert torch.allclose(
            spearman_score.raw_scores_by_miner[1].float(),  # Cast to float32
            torch.tensor(miner_b_spearman, dtype=torch.float32),
            atol=1e-6,
        ), f"Expected Spearman score for Miner B: {miner_b_spearman}, got: {spearman_score.raw_scores_by_miner[1]}"
