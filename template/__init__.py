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

GENERATOR_MODELS = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-3.5-turbo-0125",
    "openai/gpt-3.5-turbo-1106",
    "openai/gpt-3.5-turbo-0613",
    "openai/gpt-3.5-turbo-0301",
    "openai/gpt-3.5-turbo-16k",
    "openai/gpt-4-turbo",
    "openai/gpt-4-turbo-preview",
    "openai/gpt-4-1106-preview",
    "openai/gpt-4",
    "openai/gpt-4-0314",
    "openai/gpt-4-32k",
    "openai/gpt-4-32k-0314",
    "openai/gpt-4-vision-preview",
    "openai/gpt-3.5-turbo-instruct",
]

ANSWER_MODELS = [
    # "phind/phind-codellama-34b",
    # "codellama/codellama-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "openai/gpt-4-turbo-2024-04-09",
    "openai/gpt-4-1106-preview",
    "openai/gpt-3.5-turbo-1106",
    "meta-llama/llama-3-70b-instruct",
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-haiku-20240307",
    "mistralai/mistral-large",
    "google/gemini-pro-1.5",
    "cognitivecomputations/dolphin-mixtral-8x7b",
    "cohere/command-r-plus",
    "google/gemini-pro-1.0",
    "meta-llama/llama-3-8b-instruct",
]

TASK_DEADLINE = 8 * 60 * 60

DOJO_API_BASE_URL = os.getenv("DOJO_API_BASE_URL")
if DOJO_API_BASE_URL is None:
    raise ValueError("DOJO_API_BASE_URL is not set in the environment")
