import json
from typing import Any, Dict, List
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
US_EAST_REGION = "us-east-1"
# ensure regions in 'endpoint' key matches

mturk_env_dict = {
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
    if environment not in mturk_env_dict:
        raise ValueError(f"Invalid environment: {environment}")

    bt.logging.info("Using AWS environment: " + environment)
    return mturk_env_dict[environment]


env_config = get_environment_config("sandbox")

mturk_client = boto3.client(
    "mturk",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=US_EAST_REGION,
    endpoint_url=env_config["endpoint_url"],
)


class MTurkUtils:
    @staticmethod
    def encode_task_key(completion_id: str):
        """Simple method to take Completion.cid and encode it in a way that we can receive it from AWS Lambda"""
        return f"cid_{completion_id}"

    @staticmethod
    def decode_task_key(encoded_key: str) -> str:
        """Simple method to decode the key from AWS Lambda, to get the corresponding completion UUID as string"""
        return encoded_key.split("_")[1]

    @staticmethod
    def get_balance():
        return mturk_client.get_account_balance()["AvailableBalance"]

    @staticmethod
    def create_mturk_task(
        prompt: str,
        completions: List[Completion],
        score_range: ScoreRange = ScoreRange(lower=1, upper=10),
        title: str = "Prompt & Completion Evaluation Task",
        max_num_workers: int = 3,
        search_keywords: List[str] = [
            "text",
            "quick",
            "labeling",
            "scoring",
            "easy",
            "bonus",
        ],
        reward_in_dollars: float = 0.01,
    ):
        """Create a human intellgence task to send to AWS MTurk workers."""
        payout_auto_approval_seconds = 3600 * 24
        try:
            new_hit = mturk_client.create_hit(
                Title=title,
                Description=MTurkUtils.build_description(len(completions)),
                Keywords=", ".join(search_keywords),
                Reward=str(reward_in_dollars),
                MaxAssignments=max_num_workers,
                LifetimeInSeconds=3600,
                AssignmentDurationInSeconds=600,
                AutoApprovalDelayInSeconds=payout_auto_approval_seconds,
                Question=MTurkUtils.build_task_xhtml_content(
                    prompt, completions, score_range
                ),
            )
            hit_url = (
                f"{env_config['preview_url']}?groupId={new_hit['HIT']['HITGroupId']}"
            )
            bt.logging.info(
                f"A new HIT has been created. You can preview it here:\n{hit_url}"
            )
            bt.logging.info(
                "HITID = " + new_hit["HIT"]["HITId"] + " (Use to Get Results)"
            )
        except botocore.exceptions.ClientError as e:
            bt.logging.error(
                f"Error occurred while trying to create hit... exception: {e}"
            )

    @staticmethod
    def handle_mturk_event(payload: Any):
        # json_payload = None
        # if isinstance(payload, str):
        #     json_payload = json.loads(payload)
        # elif isinstance(payload, dict):
        #     json_payload = payload
        # elif isinstance(payload, list):
        #     json_payload = payload

        # assert json_payload is not None, f"Unexpected payload type {type(payload)}"

        # TODO handle this event data
        # [
        #     {
        #         "WorkerId": "A3IEQBE35F5IHB",
        #         "Answer": [
        #             {
        #                 "taskAnswers": [
        #                     {
        #                         "cid_a6f41ad1-d8a8-4bf3-a698-1b431bf2edac": 5.68,
        #                         "cid_f95cae4d-38ed-4911-b97a-f92a0c3bad9a": 7.49,
        #                     }
        #                 ]
        #             }
        #         ],
        #     }
        # ]
        print(f"Payload: {payload}")

        pass

    @staticmethod
    def build_description(num_completions):
        description = f"""
        This is a task where you will need to read up to {num_completions} sentences.
        You will need to score each prompt and sentence and score them on a scale of 1 to 10 in terms of quality.
        """
        return textwrap.dedent(description)

    @staticmethod
    def _build_instruction(num_completions, score_range: ScoreRange):
        instruction = f"""
        This is a task where you will need to read up to {num_completions} sentences.<br>
        You will need to score each prompt and sentence and score them on a scale of 1 to 10 in terms of quality.<br>
        Rate the following texts on a scale of {score_range.lower} to {score_range.upper}, where {score_range.lower} is the lowest quality and {score_range.upper} is the highest quality.
        """
        return textwrap.dedent(instruction)

    @staticmethod
    def _build_completions_html(completions: List[str]) -> str:
        html_repr = "\n".join(
            [
                f"<p><strong>Response #{index + 1}: </strong>{completion}</p>"
                for index, completion in enumerate(completions)
            ]
        )
        return html_repr

    @staticmethod
    def _build_sliders_html(completion_ids: List[str], score_range: ScoreRange):
        """Build sliders for each completion, so that we can tie the score back to the request"""
        list_items = []
        for idx, cid in enumerate(completion_ids):
            list_item = f"""
              <li>
                <p>Response #{idx+1}</p>
                <p><crowd-slider name="{MTurkUtils.encode_task_key(cid)}" step="0.01" min="{score_range.lower}" max="{score_range.upper}" required pin/></p>
              </li>
            """
            list_items.append(textwrap.dedent(list_item))

        return "<ul>" + "\n".join(list_items) + "</ul>"

    @staticmethod
    def build_task_xhtml_content(
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
                    {MTurkUtils._build_instruction(num_completions=len(completions), score_range=score_range)}
                    </strong></p>

                    <p><strong>Prompt: </strong>{prompt}</p>

                    {MTurkUtils._build_completions_html([completion.text for completion in completions])}

                    <p><strong>Score each response here!</strong>
                        {MTurkUtils._build_sliders_html([completion.cid for completion in completions], score_range)}
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
