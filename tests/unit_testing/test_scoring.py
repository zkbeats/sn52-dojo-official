import bittensor as bt
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


def prepare_multi_score(scoring_module):
    from template.protocol import (
        CodeAnswer,
        FeedbackRequest,
        MultiScoreCriteria,
        Response,
        TaskType,
    )

    request = FeedbackRequest(
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
                score=None,
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
                score=None,
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
                score=None,
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
                score=None,
            ),
        ],
    )

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

    return request, [miner_a, miner_b]


def test_consensus_no_exceptions(scoring_module):
    try:
        from commons.scoring import Scoring

        request, miner_responses = prepare_multi_score(scoring_module)
        Scoring.consensus_score(request.criteria_types[0], request, miner_responses)
    except Exception as e:
        pytest.fail(f"consensus_score raised an exception: {e}")
