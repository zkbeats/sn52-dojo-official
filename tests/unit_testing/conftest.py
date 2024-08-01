import json
import os
from unittest.mock import patch

import pytest

# @pytest.fixture(autouse=True)
# def mock_env_var(monkeypatch):
#     monkeypatch.setenv("DOJO_API_BASE_URL", "http://example.com")

#     with patch("subprocess.check_output") as mock_check_output:
#         mock_check_output.return_value = b"v1.0.0\n"
#         yield monkeypatch

#     # Test that the monkey patch for get_latest_git_tag works
#     from template import __version__

#     print(f"__version__ .................. {__version__}")

#     assert __version__ == "1.0.0"


@pytest.fixture(autouse=True)
def mock_env_var(monkeypatch):
    # Mock environment variable
    monkeypatch.setenv("DOJO_API_BASE_URL", "http://example.com")

    # Mock subprocess.check_output to return a specific git tag
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"v1.0.0\n"

        # Ensure template module is reloaded with mocks applied
        import sys

        if "template" in sys.modules:
            del sys.modules["template"]
        import template

        # Ensure the mock is applied

        # Yield control back to the test
        yield monkeypatch

        assert template.get_latest_git_tag() == "1.0.0"
        assert template.__version__ == "1.0.0"


@pytest.fixture
def mock_evalplus_leaderboard_results():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "evalplus_leaderboard_results.json"
    )
    with open(fixture_path) as f:
        mock_data = json.load(f)

    expected_keys = [
        "claude-2 (Mar 2024)",
        "claude-3-haiku (Mar 2024)",
        "claude-3-opus (Mar 2024)",
        "claude-3-sonnet (Mar 2024)",
    ]
    assert all(
        key in mock_data for key in expected_keys
    ), f"Missing one or more expected keys in the fixture data. Expected keys: {expected_keys}"

    return mock_data
