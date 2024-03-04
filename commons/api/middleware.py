import asyncio
from http import HTTPStatus
import time
from fastapi import Request
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from ipaddress import ip_network, ip_address
from commons.human_feedback.aws_mturk import US_EAST_REGION

MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class LimitContentLengthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length <= MAX_CONTENT_LENGTH:
            return await call_next(request)
        return Response(status_code=413)


class IPFilterMiddleware(BaseHTTPMiddleware):
    _aws_ips_url = "https://ip-ranges.amazonaws.com/ip-ranges.json"
    _allowed_ip_ranges = []
    _last_checked: float = 0
    _allowed_networks = []

    @classmethod
    async def get_allowed_networks(cls):
        return [ip_network(ip_range) for ip_range in allowed_ip_ranges]

    @classmethod
    async def get_allowed_ip_ranges(cls):
        if (time.time() - cls._last_checked) < 300:
            return cls._allowed_ip_ranges

        async with httpx.AsyncClient() as client:
            response = await client.get(cls._aws_ips_url)
            data = response.json()
            cls._allowed_ip_ranges = [
                ip_range["ip_prefix"]
                for ip_range in data["prefixes"]
                if ip_range["region"] == US_EAST_REGION
            ]
        return cls._allowed_ip_ranges

    def __init__(self, app):
        super().__init__(app)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.get_allowed_ip_ranges())
        self.allowed_networks = [ip_network(ip_range) for ip_range in allowed_ip_ranges]

    async def dispatch(self, request: Request, call_next):
        client_ip = ip_address(request.client.host)
        for network in [ip_network(ip_range) for ip_range in self._allowed_ip_ranges]:
            if client_ip in network:
                response = await call_next(request)
                return response
        return Response("Forbidden", status_code=403)


# Get the list of allowed IP ranges
allowed_ip_ranges = get_allowed_ip_ranges()
