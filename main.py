from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from logging.config import dictConfig

from commons.routes.evals import evals_router

load_dotenv()

MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class LimitContentLengthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length <= MAX_CONTENT_LENGTH:
            return await call_next(request)
        return Response(status_code=413)


# dictConfig(uvicorn_logging_config)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(RemoveNginxHeadersMiddleware)
app.add_middleware(LimitContentLengthMiddleware)
app.include_router(evals_router)
