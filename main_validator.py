import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from commons.api.reward_route import reward_router
from commons.api.middleware import LimitContentLengthMiddleware
from commons.factory import Factory
from neurons.validator import log_validator_status
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger

load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LimitContentLengthMiddleware)
app.include_router(reward_router)


async def main():
    validator = Factory.get_validator()
    config = Factory.get_config()
    scheduler = AsyncIOScheduler(
        job_defaults={"max_instances": 1, "misfire_grace_time": 3}
    )
    trigger = OrTrigger([CronTrigger(second=0), CronTrigger(second=30)])
    scheduler.add_job(validator.update_score_and_send_feedback, trigger=trigger)
    scheduler.start()

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=config.api.port,
        workers=1,
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)
    with validator as v:
        log_task = asyncio.create_task(log_validator_status())

        await server.serve()

        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
