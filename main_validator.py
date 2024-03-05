import commons.patch_logging  # noqa: F401
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import bittensor as bt

from commons.api.reward_route import reward_router
from commons.api.middleware import LimitContentLengthMiddleware
from commons.factory import Factory
from neurons.validator import log_validator_status
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from contextlib import asynccontextmanager

load_dotenv()


validator = Factory.get_validator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BEFORE YIELD == ON STARTUP
    bt.logging.info("Performing startup tasks...")
    yield
    # AFTER YIELD == ON SHUTDOWN
    bt.logging.info("Performing shutdown tasks...")
    validator.should_exit = True
    validator.save_state()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LimitContentLengthMiddleware)
app.include_router(reward_router)


async def main():
    scheduler = AsyncIOScheduler(
        job_defaults={"max_instances": 3, "misfire_grace_time": 3}
    )

    every_30_min_trigger = IntervalTrigger(minutes=30)
    hourly_trigger = IntervalTrigger(minutes=0, hours=1)
    daily_trigger = IntervalTrigger(hours=24)

    scheduler.add_job(validator.update_score_and_send_feedback, trigger=hourly_trigger)
    scheduler.add_job(
        validator.calculate_miner_classification_accuracy, trigger=every_30_min_trigger
    )
    scheduler.add_job(validator.reset_accuracy, trigger=daily_trigger)
    scheduler.start()

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=Factory.get_config().api.port,
        workers=1,
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)
    # with validator as v:
    log_task = asyncio.create_task(log_validator_status())
    run_task = asyncio.create_task(validator.run())

    await server.serve()

    log_task.cancel()
    run_task.cancel()
    try:
        await log_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
