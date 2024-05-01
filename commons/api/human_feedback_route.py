import bittensor as bt
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from commons.objects import ObjectManager

from commons.human_feedback.aws_mturk import MTurkUtils
from template.protocol import MTurkResponse

load_dotenv()

human_feedback_router = APIRouter(prefix="/api/human_feedback")


# process our completed tasks sent to AWS MTurk
@human_feedback_router.post("/callback")
async def task_completion_callback(request: Request):
    response_json = await request.json()
    bt.logging.info(f"Received task completion callback with body: {response_json}")
    hit_id, completion_id_to_scores = await MTurkUtils.handle_mturk_event(response_json)
    if not completion_id_to_scores:
        return

    try:
        miner = ObjectManager.get_miner()
        await miner.send_mturk_response(
            MTurkResponse(
                mturk_hit_id=hit_id, completion_id_to_score=completion_id_to_scores
            )
        )
    except Exception as e:
        bt.logging.error(f"Failed to send MTurk response: {e}")
    return
