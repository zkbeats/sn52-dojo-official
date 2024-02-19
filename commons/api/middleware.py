from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class LimitContentLengthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length <= MAX_CONTENT_LENGTH:
            return await call_next(request)
        return Response(status_code=413)
