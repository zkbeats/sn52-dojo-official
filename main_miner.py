import asyncio
import signal

import bittensor as bt
from dotenv import load_dotenv

import wandb
from commons.objects import ObjectManager
from commons.logging.patch_logging import apply_patch
from neurons.miner import log_miner_status

load_dotenv()
apply_patch()

miner = ObjectManager.get_miner()


async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    bt.logging.info(f"Received exit signal {signal.name}...")
    bt.logging.info("Performing shutdown tasks...")
    miner.should_exit = True
    wandb.finish()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    bt.logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
    bt.logging.info("Shutdown complete.")


async def main():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )

    await asyncio.gather(
        *[
            log_miner_status(),
            miner.run(),
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
