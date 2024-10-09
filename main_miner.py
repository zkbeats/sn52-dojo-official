import asyncio

from bittensor.btlogging import logging as logger
from dotenv import load_dotenv

from commons.objects import ObjectManager

load_dotenv()


async def main():
    miner = ObjectManager.get_miner()
    log_task = asyncio.create_task(miner.log_miner_status())
    run_task = asyncio.create_task(miner.run())

    await asyncio.gather(log_task, run_task)
    logger.info("Exiting main function.")


if __name__ == "__main__":
    asyncio.run(main())
