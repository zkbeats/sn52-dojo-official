from typing import Any, Dict, List, Tuple
import boto3

import textwrap
from dotenv import load_dotenv
import os

from commons.llm.prompts import ScoreRange
from template.protocol import Completion
import bittensor as bt
import botocore.exceptions

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
MTURK_SANDBOX = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
# MTURK_SANDBOX = "https://mturk-requester.us-east-1.amazonaws.com"
US_EAST_REGION = "us-east-1"
# ensure regions in 'endpoint' key matches
environment_dict = {
    "production": {
        "endpoint_url": "https://mturk-requester.us-east-1.amazonaws.com",
        "preview_url": "https://www.mturk.com/mturk/preview",
    },
    "sandbox": {
        "endpoint_url": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
        "preview_url": "https://workersandbox.mturk.com/mturk/preview",
    },
}


def get_environment_config(environment: str) -> Dict[str, Any]:
    if environment not in environment_dict:
        raise ValueError(f"Invalid environment: {environment}")

    bt.logging.info("Using AWS environment: " + environment)
    return environment_dict[environment]


env_config = get_environment_config("sandbox")

mturk_client = boto3.client(
    "mturk",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=US_EAST_REGION,
    endpoint_url=env_config["endpoint_url"],
)


def get_balance():
    print(
        "I have $"
        + mturk_client.get_account_balance()["AvailableBalance"]
        + " in my Sandbox account"
    )
    return mturk_client.get_account_balance()["AvailableBalance"]


def _build_description(num_completions):
    description = f"""
    This is a task where you will need to read up to {num_completions} sentences.
    You will need to score each prompt and sentence and score them on a scale of 1 to 10 in terms of quality.
    """
    return textwrap.dedent(description)


def _build_instruction(num_completions, score_range: ScoreRange):
    instruction = f"""
    This is a task where you will need to read up to {num_completions} sentences.<br>
    You will need to score each prompt and sentence and score them on a scale of 1 to 10 in terms of quality.<br>
    Rate the following texts on a scale of {score_range.lower} to {score_range.upper}, where {score_range.lower} is the lowest quality and {score_range.upper} is the highest quality.
    """
    return textwrap.dedent(instruction)


def _build_completions_html(completions: List[str]) -> str:
    html_repr = "\n".join(
        [
            f"<p><strong>Response #{index + 1}: </strong>{completion}</p>"
            for index, completion in enumerate(completions)
        ]
    )
    return html_repr


def _build_sliders_html(completion_ids: List[str], score_range: ScoreRange):
    """Build sliders for each completion, so that we can tie the score back to the request"""
    list_items = []
    for idx, cid in enumerate(completion_ids):
        list_item = f"""
          <li>
            <p>Response #{idx+1}</p>
            <p><crowd-slider name="cid_{cid}" step="0.01" min="{score_range.lower}" max="{score_range.upper}" required pin/></p>
          </li>
        """
        list_items.append(textwrap.dedent(list_item))

    return "<ul>" + "\n".join(list_items) + "</ul>"


def _build_task_xhtml_content(
    prompt: str, completions: List[Completion], score_range: ScoreRange
):
    xhtml_content = f"""
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
        <!DOCTYPE html>
        <body>
            <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
            <crowd-form>
                <p><strong>
                {_build_instruction(num_completions=len(completions), score_range=score_range)}
                </strong></p>

                <p><strong>Prompt: </strong>{prompt}</p>

                {_build_completions_html([completion.text for completion in completions])}

                <p><strong>Score each response here!</strong>
                    {_build_sliders_html([completion.cid for completion in completions], score_range)}
                </p>

            <crowd-toast duration="10000" opened>
            Thank you for your interest in our task. By completing this task, you will be contributing to the future of Artificial Intelligence!
            </crowd-toast>

            </crowd-form>
        </body>
        </html>
    ]]></HTMLContent>
    <FrameHeight>0</FrameHeight>
</HTMLQuestion>
    """
    return xhtml_content


# TODO take these as parameters
def create_mturk_task():
    """Create a human intellgence task to send to AWS MTurk workers."""
    score_range = ScoreRange(lower=1, upper=10)

    # TODO take in real data after testing
    prompt = "What is your name?"
    completions = [
        Completion(text="My name is Alice, nice to meet you!"),
        Completion(text="My name is Bob and I like apples."),
    ]

    max_num_mturk_workers = 3
    payout_auto_approval_seconds = 3600 * 48
    search_keywords = ", ".join(
        ["text", "quick", "labeling", "scoring", "easy", "bonus"]
    )
    try:
        new_hit = mturk_client.create_hit(
            Title="Prompt & Completion Evaluation Task",
            Description=_build_description(len(completions)),
            Keywords=search_keywords,
            Reward=str(0.01),
            MaxAssignments=max_num_mturk_workers,
            LifetimeInSeconds=3600,
            AssignmentDurationInSeconds=600,
            AutoApprovalDelayInSeconds=payout_auto_approval_seconds,
            Question=_build_task_xhtml_content(prompt, completions, score_range),
        )
        print("A new HIT has been created. You can preview it here:")
        print(
            "https://workersandbox.mturk.com/mturk/preview?groupId="
            + new_hit["HIT"]["HITGroupId"]
        )
        print("HITID = " + new_hit["HIT"]["HITId"] + " (Use to Get Results)")
    except botocore.exceptions.ClientError as e:
        print(f"Error occurred while trying to create hit... exception: {e}")


if __name__ == "__main__":
    # send_test_event_notification()
    create_mturk_task()
    print("Done...")
    pass
