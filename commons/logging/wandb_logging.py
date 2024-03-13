import asyncio
from typing import Any, Dict

import wandb


async def wandb_log(wandb_data: Dict[Any, Any]):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, wandb.log, wandb_data)
