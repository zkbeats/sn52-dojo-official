import logging
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor as bt
import numpy as np
import pytest
from loguru import logger

from commons.utils import get_epoch_time
from neurons.miner import Miner
from template import VALIDATOR_MIN_STAKE
from template.protocol import (
    FeedbackRequest,
    MultiScoreCriteria,
    Response,
    ScoringMethod,
    ScoringResult,
    TaskType,
)

valid_feedback_request = FeedbackRequest(
    request_id="test_request_id",
    dendrite=bt.TerminalInfo(hotkey="mock_hotkey"),
    prompt="test_prompt",
    task_type=TaskType.CODE_GENERATION,
    criteria_types=[
        MultiScoreCriteria(type="multi-score", options=[], min=0.0, max=100.0)
    ],
    responses=[Response(model="test_model", completion="test_completion")],
)

invalid_feedback_request = FeedbackRequest(
    request_id="test_request_id",
    dendrite=bt.TerminalInfo(hotkey="mock_hotkey"),
    prompt="test_prompt",
    task_type=TaskType.CODE_GENERATION,
    criteria_types=[
        MultiScoreCriteria(type="multi-score", options=[], min=0.0, max=100.0)
    ],
    responses=[],  # Invalid because responses list is empty
)

MOCK_HOTKEYS: list[str] = [
    "o7PE28T0L1iU5s4pecJvYLGlfGZ5sljMVv4OkL5cRXHz",
    "OHBpuTVsYcKlNfP8oF2UgyfAPdm5mNEgjeV4ezAfXhXh",
]


# Fixture to propagate loguru logs to standard logging
@pytest.fixture(autouse=True)
def configure_loguru(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    logger.add(PropagateHandler(), format="{message}")


@pytest.fixture
def mock_miner(mock_initialise):
    """Fixture to setup miner with mock components for testing."""
    mock_initialise_func, mock_wallet, mock_subtensor, mock_metagraph, mock_dendrite = (
        mock_initialise
    )

    with patch.object(Miner, "__init__", lambda x: None):
        miner = Miner()

        # Inject the mocked components
        miner.wallet = mock_wallet
        miner.subtensor = mock_subtensor
        miner.metagraph = mock_metagraph
        miner.metagraph.hotkeys = MOCK_HOTKEYS
        miner.dendrite = mock_dendrite
        miner.config = mock_wallet.config
        miner.config.scoring_method = ScoringMethod.DOJO

        yield miner


@pytest.mark.asyncio
async def test_forward_result_valid(mock_miner: Miner):
    """Test that a valid miner's ScoringResult has expected return"""
    # Mock a valid scoring result
    mock_hotkey = str(mock_miner.wallet.hotkey.ss58_address)  # Convert hotkey to string
    synapse = ScoringResult(
        request_id="test_request_id", hotkey_to_scores={mock_hotkey: 0.5}
    )

    response = await mock_miner.forward_result(synapse)

    assert response.hotkey_to_scores[mock_hotkey] == 0.5


@pytest.mark.asyncio
async def test_forward_result_missing_hotkey_to_scores(mock_miner: Miner, caplog):
    """Test that a miner's ScoringResult is None or hotkey_to_scores attribute is missing or empty object"""
    synapse = None
    response = await mock_miner.forward_result(synapse)
    assert response is None

    synapse = ScoringResult(request_id="test_request_id", hotkey_to_scores={})
    with caplog.at_level(logging.ERROR):
        response = await mock_miner.forward_result(synapse)

    assert response.hotkey_to_scores == {}
    assert (
        "Invalid synapse object or missing hotkey_to_scores attribute." in caplog.text
    )


@pytest.mark.asyncio
async def test_forward_result_hotkey_not_found(mock_miner: Miner, caplog):
    """Test that a miner's hotkey not found in hotkey_to_scores triggers an error."""
    synapse = ScoringResult(
        request_id="test_request_id",
        hotkey_to_scores={"another_hotkey": 0.5},  # different hotkey
    )
    expected_log = f"Miner hotkey {mock_miner.wallet.hotkey.ss58_address} not found in scoring result but yet was sent the result"

    # Call the forward_result function
    with caplog.at_level(logging.ERROR):
        response = await mock_miner.forward_result(synapse)

    # Assert that the response is handled correctly
    assert response == synapse
    assert expected_log in caplog.text


@pytest.mark.asyncio
async def test_forward_feedback_request_invalid_synapse(mock_miner: Miner, caplog):
    """Test the case where the synapse or its properties are None."""
    synapse = None
    with caplog.at_level(logging.ERROR):
        response = await mock_miner.forward_feedback_request(synapse)
    assert response is None
    assert "dendrite or dendrite.hotkey is None" in caplog.text


@pytest.mark.asyncio
async def test_forward_feedback_request_none_responses(mock_miner: Miner, caplog):
    """Test the case where the responses field is None."""
    with caplog.at_level(logging.ERROR):
        response = await mock_miner.forward_feedback_request(invalid_feedback_request)
    assert response == invalid_feedback_request
    assert "Invalid synapse: response field is None." in caplog.text


@pytest.mark.asyncio
@patch("neurons.miner.DojoAPI.create_task", new_callable=AsyncMock)
async def test_forward_feedback_request_dojo_method(
    mock_create_task, mock_miner: Miner
):
    """Test the case where the scoring method is DOJO and the task creation is successful."""
    synapse = valid_feedback_request
    mock_miner.hotkey_to_request = {"mock_hotkey": synapse}
    mock_create_task.return_value = ["task_id"]

    response = await mock_miner.forward_feedback_request(synapse)

    assert response == synapse
    assert response.dojo_task_id == "task_id"
    mock_create_task.assert_called_once_with(synapse)


@pytest.mark.asyncio
async def test_forward_feedback_request_unrecognized_scoring_method(
    mock_miner: Miner, caplog
):
    """Test the case where the scoring method is unrecognized."""
    synapse = valid_feedback_request
    mock_miner.hotkey_to_request = {"mock_hotkey": synapse}
    mock_miner.config.scoring_method = "UNKNOWN_METHOD"

    with caplog.at_level(logging.ERROR):
        response = await mock_miner.forward_feedback_request(synapse)
    assert response == synapse
    assert "Unrecognized scoring method!" in caplog.text


@pytest.mark.asyncio
async def test_miner_blacklisting_invalid_hotkey(mock_miner: Miner, caplog):
    """Test the case where miner blacklisting the unrecognized hotkey"""
    synapse = valid_feedback_request
    synapse.dendrite.hotkey = "unrecognized_hotkey"

    response = await mock_miner.blacklist_feedback_request(synapse)
    is_blacklist, msg = response

    assert is_blacklist is True
    assert msg == "Unrecognized hotkey"


@pytest.mark.asyncio
async def test_miner_blacklisting_miner_hotkey(mock_miner: Miner, caplog):
    """Test the case where the hotkey corresponds to a miner"""
    miner_hotkey = mock_miner.wallet.hotkey.ss58_address
    mock_miner.metagraph.hotkeys = [miner_hotkey]
    mock_miner.metagraph.neurons = [MagicMock(stake=MagicMock(tao=10.0))]

    synapse = valid_feedback_request
    synapse.dendrite.hotkey = miner_hotkey

    with patch("template.utils.uids.is_miner", return_value=True):
        response = await mock_miner.blacklist_feedback_request(synapse)
    is_blacklist, msg = response

    assert is_blacklist is True
    assert msg == "Not a validator"


@pytest.mark.asyncio
async def test_miner_blacklisting_is_not_a_validator(mock_miner: Miner, caplog):
    """Test the case where the validator has insufficient stake"""
    validator_hotkey = "validator_hotkey"
    mock_miner.metagraph.hotkeys = [validator_hotkey]
    mock_miner.metagraph.total_stake = np.array(
        [
            float(VALIDATOR_MIN_STAKE - 5.0)  # this means not validator
        ]
    )

    synapse = valid_feedback_request
    synapse.dendrite.hotkey = validator_hotkey

    with caplog.at_level(logging.WARNING):
        response = await mock_miner.blacklist_feedback_request(synapse)
    is_miner, msg = response

    assert is_miner is True
    assert msg == "Not a validator"


@pytest.mark.asyncio
async def test_miner_blacklisting_insufficient_stake(mock_miner: Miner, caplog):
    """Test the case where the validator has insufficient stake"""
    validator_hotkey = "validator_hotkey"
    mock_miner.metagraph.hotkeys = [validator_hotkey]
    mock_miner.metagraph.total_stake = np.array(
        [
            float(VALIDATOR_MIN_STAKE + 5.0)  # this means validator
        ]
    )

    # Mock the neurons with insufficient stake
    mock_miner.metagraph.neurons = [
        MagicMock(stake=MagicMock(tao=float(VALIDATOR_MIN_STAKE - 5.0)))
    ]

    synapse = valid_feedback_request
    synapse.dendrite.hotkey = validator_hotkey

    with caplog.at_level(logging.WARNING):
        response = await mock_miner.blacklist_feedback_request(synapse)
    is_blacklisted, msg = response

    assert is_blacklisted is True
    assert msg == "Insufficient validator stake"
    assert (
        f"Blacklisting hotkey: {validator_hotkey} with insufficient stake"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_miner_blacklisting_sufficient_stake(mock_miner: Miner, caplog):
    """Test the case where the validator has insufficient stake"""
    validator_hotkey = "validator_hotkey"
    mock_miner.metagraph.hotkeys = [validator_hotkey]
    mock_miner.metagraph.total_stake = np.array(
        [
            float(VALIDATOR_MIN_STAKE + 5.0)  # this means validator
        ]
    )

    # Mock the neurons with insufficient stake
    mock_miner.metagraph.neurons = [
        MagicMock(stake=MagicMock(tao=float(VALIDATOR_MIN_STAKE + 5.0)))
    ]

    synapse = valid_feedback_request
    synapse.dendrite.hotkey = validator_hotkey

    response = await mock_miner.blacklist_feedback_request(synapse)
    is_blacklisted, msg = response

    assert is_blacklisted is False
    assert msg == "Valid request received from validator"


@pytest.mark.asyncio
async def test_priority_ranking_basic(mock_miner: Miner):
    """Test the basic functionality of the priority_ranking function."""
    synapse = valid_feedback_request
    current_time = datetime.fromtimestamp(get_epoch_time())
    synapse.epoch_timestamp = (
        current_time - timedelta(seconds=10)
    ).timestamp()  # 10 seconds ago

    priority = await mock_miner.priority_ranking(synapse)

    assert math.floor(priority) == 10.0  # 10 seconds difference


@pytest.mark.asyncio
async def test_priority_ranking_different_timestamps(mock_miner: Miner):
    """Test the priority_ranking function with different epoch timestamps."""
    synapse = valid_feedback_request
    current_time = datetime.fromtimestamp(get_epoch_time())

    # Case 1: 20 seconds ago
    synapse.epoch_timestamp = (current_time - timedelta(seconds=20)).timestamp()
    priority_1 = await mock_miner.priority_ranking(synapse)

    # Case 2: 5 seconds ago
    synapse.epoch_timestamp = (current_time - timedelta(seconds=5)).timestamp()
    priority_2 = await mock_miner.priority_ranking(synapse)

    # Case 3: 0 seconds ago
    synapse.epoch_timestamp = current_time.timestamp()
    priority_3 = await mock_miner.priority_ranking(synapse)

    assert priority_1 > priority_2 > priority_3
