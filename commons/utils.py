import time
import uuid


def get_new_uuid():
    return uuid.uuid4()


def get_epoch_time():
    return int(time.time() * 1000)
