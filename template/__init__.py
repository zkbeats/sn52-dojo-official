import os
import subprocess

from dotenv import load_dotenv

load_dotenv()


def get_latest_git_tag():
    try:
        # Get the latest git tag
        latest_tag = (
            subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .strip()
            .decode("utf-8")
        )
        return latest_tag.lstrip("v")
    except subprocess.CalledProcessError as e:
        print(f"Error getting the latest Git tag: {e}")
        raise RuntimeError("Failed to get latest Git tag")


# Define the version of the template module.
__version__ = get_latest_git_tag()
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# Import all submodules.

# TODO @dev change before live
VALIDATOR_MIN_STAKE = 20000
TASK_DEADLINE = 8 * 60 * 60

# Define the time intervals for various tasks.
VALIDATOR_RUN = 300
VALIDATOR_HEARTBEAT = 60
VALIDATOR_UPDATE_SCORE = 60
VALIDATOR_STATUS = 60
MINER_STATUS = 60
DOJO_TASK_MONITORING = 60

DOJO_API_BASE_URL = os.getenv("DOJO_API_BASE_URL")
if DOJO_API_BASE_URL is None:
    raise ValueError("DOJO_API_BASE_URL is not set in the environment")
