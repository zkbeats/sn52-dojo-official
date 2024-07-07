import os

from dotenv import load_dotenv

load_dotenv()

# Define the version of the template module.
__version__ = "0.0.1"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# Import all submodules.

# TODO @dev change before live
VALIDATOR_MIN_STAKE = 99
TASK_DEADLINE = 5 * 60
# TASK_DEADLINE = 8 * 60 * 60

DOJO_API_BASE_URL = os.getenv("DOJO_API_BASE_URL")
if DOJO_API_BASE_URL is None:
    raise ValueError("DOJO_API_BASE_URL is not set in the environment")
