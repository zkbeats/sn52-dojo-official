import json
import os

from redis import asyncio as aioredis


def build_redis_url() -> str:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    username = os.getenv("REDIS_USERNAME")
    password = os.getenv("REDIS_PASSWORD")

    if username and password:
        return f"redis://{username}:{password}@{host}:{port}"
    elif password:
        return f"redis://:{password}@{host}:{port}"
    else:
        return f"redis://{host}:{port}"


class RedisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.redis = None
            # loop = asyncio.get_running_loop()
            # loop.run_until_complete(cls._instance.connect())
        return cls._instance

    async def connect(self):
        if self.redis is None:
            redis_url = build_redis_url()
            self.redis = await aioredis.from_url(redis_url)

    async def put(self, key: str, value: dict):
        if self.redis is None:
            await self.connect()
        await self.redis.set(key, json.dumps(value))

    async def get(self, key: str) -> dict | None:
        if self.redis is None:
            await self.connect()
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def close(self):
        if self.redis:
            self.redis.close()
            await self.redis.aclose()

    async def check_num_keys(self, key):
        if self.redis is None:
            await self.connect()
        keys = await self.redis.keys(f"{key}*")
        return len(keys)
