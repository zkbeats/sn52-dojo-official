from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from commons.api.eval_route import evals_router
from commons.api.middleware import LimitContentLengthMiddleware

load_dotenv()


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
