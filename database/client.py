import asyncio
from contextlib import asynccontextmanager

from loguru import logger

from database.prisma import Prisma

db = None

prisma = Prisma(auto_register=True)


async def connect_db(retries: int = 5, delay: int = 2) -> None:
    global db
    attempt = 1

    while attempt <= retries:
        try:
            if not prisma.is_connected():
                await prisma.connect()
                db = prisma
                logger.success("Successfully connected to the database.")
            else:
                db = prisma
                logger.info("Already connected to the database.")
            return  # Exit the function if connection is successful
        except Exception as e:
            logger.error(
                f"Failed to connect to the database (Attempt {attempt}/{retries}): {e}"
            )
            await asyncio.sleep(delay**attempt)
            attempt += 1

    logger.critical("Exceeded maximum retry attempts to connect to the database.")
    raise ConnectionError("Failed to connect to the database after multiple attempts.")


async def disconnect_db():
    if prisma.is_connected():
        logger.info("Releasing connection......")
        await prisma.disconnect()
        logger.info("Connection released......")


@asynccontextmanager
async def transaction():
    async with prisma.tx() as tx:
        yield tx
