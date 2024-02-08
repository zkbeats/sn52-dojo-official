from typing import List, Tuple
import boto3

import textwrap
from dotenv import load_dotenv
import os

from commons.llm.prompts import ScoreRange
from template.protocol import Completion

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
MTURK_SANDBOX = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

mturk_client = boto3.client(
    "mturk",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name="us-east-1",
    endpoint_url=MTURK_SANDBOX,
)
print(
    "I have $"
    + mturk_client.get_account_balance()["AvailableBalance"]
    + " in my Sandbox account"
)


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


# TODO based on prompt and completions build xhtml content


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
                <p><strong>{_build_instruction(num_completions=len(completions), score_range=score_range)}</strong></p>
                <p><strong>Prompt: </strong>{prompt}</p>
                {_build_completions_html([completion.text for completion in completions])}

                <p><strong>Score each response here.</strong>
                    {_build_sliders_html([completion.cid for completion in completions], score_range)}
                </p>
            </crowd-form>
        </body>
        </html>
    ]]></HTMLContent>
    <FrameHeight>0</FrameHeight>
</HTMLQuestion>
    """
    return xhtml_content


score_range = ScoreRange(lower=1, upper=10)
prompt = "What is your name?"
completions = [
    Completion(text="My name is Alice, nice to meet you!"),
    Completion(text="My name is Bob and I like apples."),
]

new_hit = mturk_client.create_hit(
    Title="Is this Tweet happy, angry, excited, scared, annoyed or upset?",
    Description=_build_description(5),
    Keywords="text, quick, labeling, scoring, easy, bonus",
    Reward=str(0.15),
    MaxAssignments=1,
    LifetimeInSeconds=3600,
    AssignmentDurationInSeconds=600,
    AutoApprovalDelayInSeconds=3600 * 48,
    Question=build_task_xhtml_content(prompt, completions, score_range),
)
print("A new HIT has been created. You can preview it here:")
print(
    "https://workersandbox.mturk.com/mturk/preview?groupId="
    + new_hit["HIT"]["HITGroupId"]
)
print("HITID = " + new_hit["HIT"]["HITId"] + " (Use to Get Results)")
# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=


################################ STORING AWS XHTML CONTENT SAMPLES


def get_sentiment_classification_example():
    return """
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
      <HTMLContent><![CDATA[
        <!DOCTYPE html>
          <body>
            <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
            <crowd-form>
              <crowd-classifier
                name="sentiment"
                categories="['Positive', 'Negative', 'Neutral', 'N/A']"
                header="What sentiment does this text convey?"
              >
                <classification-target>
                Everything is wonderful.
                </classification-target>

                <full-instructions header="Sentiment Analysis Instructions">
                <p><strong>Positive</strong>
                  sentiment include: joy, excitement, delight</p>
                <p><strong>Negative</strong> sentiment include:
                  anger, sarcasm, anxiety</p>
                <p><strong>Neutral</strong>: neither positive or
                  negative, such as stating a fact</p>
                <p><strong>N/A</strong>: when the text cannot be
                  understood</p>
                <p>When the sentiment is mixed, such as both joy and sadness,
                  use your judgment to choose the stronger emotion.</p>
                </full-instructions>

                <short-instructions>
                 Choose the primary sentiment that is expressed by the text.
                </short-instructions>
              </crowd-classifier>
            </crowd-form>
          </body>
        </html>
      ]]></HTMLContent>
      <FrameHeight>0</FrameHeight>
    </HTMLQuestion>
    """
