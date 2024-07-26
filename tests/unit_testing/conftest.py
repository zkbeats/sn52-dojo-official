from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_var(monkeypatch):
    monkeypatch.setenv("DOJO_API_BASE_URL", "http://example.com")

    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"v1.0.0\n"
        yield monkeypatch

    # Test that the monkey patch for get_latest_git_tag works
    from template import __version__

    assert __version__ == "1.0.0"
