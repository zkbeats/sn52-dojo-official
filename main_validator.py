import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from commons.api.eval_route import evals_router
from commons.api.middleware import LimitContentLengthMiddleware
from neurons.validator import Validator, log_validator_status
from apscheduler.schedulers.asyncio import AsyncIOScheduler

load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LimitContentLengthMiddleware)
app.include_router(evals_router)


async def main():
    validator = Validator()
    scheduler = AsyncIOScheduler(
        job_defaults={"max_instances": 1, "misfire_grace_time": 3}
    )
    scheduler.add_job(validator.update_score_and_send_feedback, "interval", seconds=10)
    scheduler.start()

    with validator as v:
        log_task = asyncio.create_task(log_validator_status())

        config = uvicorn.run(
            app=app,
            host="0.0.0.0",
            port=5004,
            workers=1,
            log_level="info",
            # NOTE should only be used in development.
            reload=False,
        )

        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
