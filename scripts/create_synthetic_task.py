import asyncio

from commons.dataset.synthetic import SyntheticAPI
from commons.human_feedback.dojo import DojoAPI
from template.protocol import FeedbackRequest, MultiScoreCriteria, TaskType


async def main():
    data = await SyntheticAPI.get_qa()
    synapse = FeedbackRequest(
        task_type=str(TaskType.CODE_GENERATION),
        criteria_types=[
            MultiScoreCriteria(
                options=[completion.model for completion in data.responses],
                min=1.0,
                max=100.0,
            ),
        ],
        prompt=data.prompt,
        responses=data.responses,
    )

    task_response = await DojoAPI.create_task(synapse)
    print(task_response)
    await DojoAPI._http_client.aclose()


asyncio.run(main())
