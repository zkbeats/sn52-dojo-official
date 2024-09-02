import logging
import socket

from fastapi import FastAPI, HTTPException
from loguru import logger

from commons.human_feedback.dojo import DojoAPI

app = FastAPI()


@app.get("/task-result/{task_id}")
async def get_task_result(task_id: str):
    try:
        task_result = await DojoAPI._get_task_by_id(task_id)
        return task_result
    except Exception as e:
        logging.error(f"Error fetching task result for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hello")
async def hello():
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    return {"message": "Hello World", "miner_ip": host_ip, "miner_name": host_name}


if __name__ == "__main__":
    import uvicorn

    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    logger.info(f"Miner API starting at http://{host_ip}:8000")

    logging.info("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
