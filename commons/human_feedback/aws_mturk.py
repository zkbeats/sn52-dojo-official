import os
import textwrap
from collections import defaultdict
from typing import Any, Dict, List
import xml.etree.ElementTree as ET
import json

import bittensor as bt
import boto3
import botocore.exceptions
import markdown
from dotenv import load_dotenv
from commons.objects import ObjectManager

from commons.llm.prompts import ScoreRange
from template.protocol import AWSCredentials, Response

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
US_EAST_REGION = "us-east-1"
# should look like the form: arn:aws:sns:us-east-1:1234567890:sns_topic_name
AWS_SNS_ARN_ID = os.getenv("AWS_SNS_ARN_ID")
AWS_ASSUME_ROLE_ARN = os.getenv("AWS_ASSUME_ROLE_ARN")


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
        raise ValueError(
            f"Invalid environment: {environment}, should be one of {mturk_env_dict.keys()}"
        )

    current_env = mturk_env_dict[environment]
    if US_EAST_REGION not in current_env["endpoint_url"]:
        raise ValueError(
            f"Invalid region in endpoint url: {current_env['endpoint_url']}"
        )

    bt.logging.info("Using AWS environment: " + environment)
    return current_env


def parse_assignment(assignment):
    result = {
        # "WorkerId": assignment["WorkerId"],
        "Answer": [],
        "HITId": assignment["HITId"],
    }

    ns = {
        "mt": "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"
    }
    root = ET.fromstring(assignment["Answer"])

    for a in root.findall("mt:Answer", ns):
        name = a.find("mt:QuestionIdentifier", ns).text
        value = a.find("mt:FreeText", ns).text
        result["Answer"].append({name: json.loads(value)})
    return result


class STSUtils:
    _sts_client = None

    @classmethod
    def get_client(
        cls,
        access_key_id: str = AWS_ACCESS_KEY_ID,
        secret_access_key: str = AWS_SECRET_KEY,
    ):
        if cls._sts_client is None:
            kwargs = {
                "aws_access_key_id": access_key_id,
                "aws_secret_access_key": secret_access_key,
                "region_name": US_EAST_REGION,
            }
            sts_client = boto3.client("sts", **kwargs)
            cls._sts_client = sts_client
        return cls._sts_client

    @classmethod
    def assume_role(cls, role_arn: str = AWS_ASSUME_ROLE_ARN):
        assert isinstance(role_arn, str)
        client = cls.get_client()
        res = client.assume_role(RoleArn=role_arn, RoleSessionName="subnet_validator")

        return AWSCredentials(
            access_key_id=res["Credentials"]["AccessKeyId"],
            secret_access_key=res["Credentials"]["SecretAccessKey"],
            session_token=res["Credentials"]["SessionToken"],
            access_expiration=res["Credentials"]["Expiration"],
        )


class MTurkUtils:
    _mturk_client = None

    @classmethod
    def get_client(
        cls,
        access_key_id: str = AWS_ACCESS_KEY_ID,
        secret_access_key: str = AWS_SECRET_KEY,
        session_token: str = None,
        environment: str = ObjectManager.get_config().aws_mturk_environment,
    ):
        if cls._mturk_client is None:
            env_config = get_environment_config(environment)
            kwargs = {
                "aws_access_key_id": access_key_id,
                "aws_secret_access_key": secret_access_key,
                "region_name": US_EAST_REGION,
                "endpoint_url": env_config["endpoint_url"],
            }
            if session_token:
                kwargs["aws_session_token"] = session_token
            mturk_client = boto3.client("mturk", **kwargs)
            cls._mturk_client = mturk_client

        return cls._mturk_client

    @staticmethod
    def encode_task_key(completion_id: str):
        """Simple method to take Completion.cid and encode it in a way that we can receive it from AWS Lambda"""
        # return f"cid_{completion_id}"
        return completion_id

    @staticmethod
    def decode_task_key(encoded_key: str) -> str:
        """Simple method to decode the key from AWS Lambda, to get the corresponding completion UUID as string"""
        # return encoded_key.split("_")[1]
        return encoded_key

    @staticmethod
    def get_balance():
        return MTurkUtils.get_client().get_account_balance()["AvailableBalance"]

    @staticmethod
    def create_mturk_task(
        prompt: str,
        completions: List[Response],
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
        success = False
        hit_id = None
        try:
            new_hit = MTurkUtils.get_client().create_hit(
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
            success = True
            env_config = get_environment_config(
                ObjectManager.get_config().aws_mturk_environment
            )
            hit_url = (
                f"{env_config['preview_url']}?groupId={new_hit['HIT']['HITGroupId']}"
            )
            bt.logging.success(
                f"A new HIT has been created. You can preview it here: {hit_url}"
            )
            bt.logging.success(
                "HITID = " + new_hit["HIT"]["HITId"] + " (Use to Get Results)"
            )

            hit_id = new_hit["HIT"]["HITId"]
            try:
                hit_type_id = new_hit["HIT"]["HITTypeId"]
                MTurkUtils.get_client().update_notification_settings(
                    HITTypeId=hit_type_id,
                    Notification={
                        "Destination": AWS_SNS_ARN_ID,
                        "Transport": "SNS",
                        "Version": "2006-05-05",
                        "EventTypes": [
                            "AssignmentSubmitted",
                        ],
                    },
                    Active=True,
                )
            except Exception as e:
                success = False
                bt.logging.error("Failed to update notification settings: " + str(e))
                pass

            return success, hit_id
        except botocore.exceptions.ClientError as e:
            bt.logging.error(
                f"Error occurred while trying to create hit... exception: {e}"
            )
            return False, None

    @staticmethod
    async def handle_mturk_event(event_payload: Dict):
        if event_payload is None or not event_payload:
            bt.logging.warning(f"MTurk event is None or empty, {event_payload=}")
            return

        completion_id_to_scores = defaultdict(list)
        for item in event_payload:
            answers = item.get("Answer")
            for answer in answers:
                task_answers = answer.get("taskAnswers")
                if task_answers is None:
                    bt.logging.warning("MTurk event has no task answers")
                    continue

                for task_answer in task_answers:
                    for task_key, score in task_answer.items():
                        completion_id = MTurkUtils.decode_task_key(task_key)
                        completion_id_to_scores[completion_id].append(score)

        completion_id_to_scores = dict(completion_id_to_scores)
        bt.logging.info(
            f"Processed MTurk event, completion ID to scores: {completion_id_to_scores}"
        )
        for k, v in completion_id_to_scores.items():
            completion_id_to_scores[k] = float(sum(v) / len(v))

        bt.logging.info(
            f"Taking the average of set of scores: {completion_id_to_scores}"
        )
        return event_payload["HITId"], completion_id_to_scores

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
                f"<p><strong>Response #{index + 1}: </strong>{markdown.markdown(completion, output_format='xhtml')}</p>"
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
        prompt: str, completions: List[Response], score_range: ScoreRange
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

                    {MTurkUtils._build_completions_html([completion.json() for completion in completions])}

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
