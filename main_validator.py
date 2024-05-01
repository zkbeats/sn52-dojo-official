import asyncio
from contextlib import asynccontextmanager

import bittensor as bt
import uvicorn
import wandb
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from commons.api.middleware import LimitContentLengthMiddleware
from commons.api.reward_route import reward_router
from commons.objects import ObjectManager
from commons.logging.patch_logging import apply_patch
from neurons.validator import DojoTaskTracker, log_validator_status

load_dotenv()
apply_patch()


validator = ObjectManager.get_validator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BEFORE YIELD == ON STARTUP
    bt.logging.info("Performing startup tasks...")
    yield
    # AFTER YIELD == ON SHUTDOWN
    bt.logging.info("Performing shutdown tasks...")
    validator.should_exit = True
    validator.save_state()
    wandb.finish()


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
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=ObjectManager.get_config().api.port,
        workers=1,
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)
    await asyncio.gather(
        *[
            log_validator_status(),
            validator.run(),
            server.serve(),
            validator.update_score_and_send_feedback(),
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
