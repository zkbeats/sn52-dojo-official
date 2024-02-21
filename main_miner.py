import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from commons.api.human_feedback_route import human_feedback_router
from commons.api.middleware import LimitContentLengthMiddleware
from commons.factory import Factory
from neurons.miner import log_miner_status

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LimitContentLengthMiddleware)
app.include_router(human_feedback_router)


async def main():
    miner = Factory.get_miner()
    config = Factory.get_config()
    with miner as m:
        log_task = asyncio.create_task(log_miner_status())

        config = uvicorn.run(
            app=app,
            host="0.0.0.0",
            port=config.api.port,
            workers=1,
            log_level="info",
            # NOTE should only be used in development.
            reload=False,
        )

        # once the server is closed, cancel the logging task
        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
