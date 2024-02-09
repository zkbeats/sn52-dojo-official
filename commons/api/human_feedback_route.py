import bittensor as bt
from dotenv import load_dotenv
from fastapi import APIRouter, Request

from commons.human_feedback.aws_mturk import MTurkUtils

load_dotenv()

human_feedback_router = APIRouter(prefix="/api/human_feedback")


# process our completed tasks sent to AWS MTurk
@human_feedback_router.post("/callback")
async def task_completion_callback(request: Request):
    response_json = await request.json()
    bt.logging.info(f"Received task completion callback with body: {response_json}")
    # TODO completions and scores
    await MTurkUtils.handle_mturk_event(response_json)
    return
