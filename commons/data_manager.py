import asyncio
import pickle
from pathlib import Path
from typing import Any, List, Optional

import bittensor as bt
import torch

from commons.objects import ObjectManager
from template.protocol import DendriteQueryResponse, FeedbackRequest


class DataManager:
    _lock = asyncio.Lock()
    _validator_lock = asyncio.Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._ensure_paths_exist()
        return cls._instance

    @classmethod
    def _ensure_paths_exist(cls):
        cls.get_validator_state_filepath().parent.mkdir(parents=True, exist_ok=True)
        cls.get_requests_data_path().parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_requests_data_path() -> Path:
        config = ObjectManager.get_config()
        base_path = config.data_manager.base_path
        return base_path / "data" / "requests.pkl"

    @staticmethod
    def get_validator_state_filepath() -> Path:
        config = ObjectManager.get_config()
        base_path = config.data_manager.base_path
        return base_path / "data" / "validator_state.pt"

    @classmethod
    async def _load_without_lock(cls, path) -> Optional[List[DendriteQueryResponse]]:
        try:
            with open(str(path), "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            bt.logging.error(f"File not found at {path}... , exception:{e}")
            return None
        except pickle.PickleError as e:
            bt.logging.error(f"Pickle error: {e}")
            return None
        except Exception:
            bt.logging.error("Failed to load existing ranking data from file.")
            return None

    @classmethod
    async def load(cls, path):
        # Load the list of Pydantic objects from the pickle file
        async with cls._lock:
            return await cls._load_without_lock(path)

    @classmethod
    async def _save_without_lock(cls, path, data: Any):
        try:
            with open(str(path), "wb") as file:
                pickle.dump(data, file)
                bt.logging.success(f"Saved data to {path}")
                return True
        except Exception as e:
            bt.logging.error(f"Failed to save data to file: {e}")
            return False

    @classmethod
    async def save(cls, path, data: Any):
        try:
            async with cls._lock:
                with open(str(path), "wb") as file:
                    pickle.dump(data, file)
                    bt.logging.success(f"Saved data to {path}")
                    return True
        except Exception as e:
            bt.logging.error(f"Failed to save data to file: {e}")
            return False

    @classmethod
    async def save_response(cls, response: DendriteQueryResponse):
        path = DataManager.get_requests_data_path()
        async with cls._lock:
            # ensure parent path exists
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            data = await DataManager._load_without_lock(path=path)
            if not data:
                # store initial data
                await DataManager._save_without_lock(path, [response])
                return

            # append our data, if the existing data exists
            assert isinstance(data, list)
            data.append(response)
            await DataManager._save_without_lock(path, data)
        return

    @classmethod
    async def append_responses(cls, request_id: str, responses: List[FeedbackRequest]):
        async with cls._lock:
            _path = DataManager.get_requests_data_path()
            data = await cls._load_without_lock(path=_path)
            found_response_index = next(
                (i for i, x in enumerate(data) if x.request.request_id == request_id),
                None,
            )
            if not found_response_index:
                return

            data[found_response_index].responses.extend(responses)
            # overwrite the data
            await cls._save_without_lock(_path, data)
        return

    @classmethod
    async def get_by_request_id(cls, request_id):
        async with cls._lock:
            _path = DataManager.get_requests_data_path()
            data = await cls._load_without_lock(path=_path)
            found_response = next(
                (x for x in data if x.request.request_id == request_id),
                None,
            )
            return found_response

    @classmethod
    async def remove_responses(
        cls, responses: List[DendriteQueryResponse]
    ) -> Optional[DendriteQueryResponse]:
        path = DataManager.get_requests_data_path()
        async with cls._lock:
            data = await DataManager._load_without_lock(path=path)
            assert isinstance(data, list)

            if data is None:
                return

            new_data = []

            for d in data:
                if d in responses:
                    continue
                new_data.append(d)

            await DataManager._save_without_lock(path, new_data)

    @classmethod
    async def validator_save(cls, scores):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")
        async with cls._validator_lock:
            cls._ensure_paths_exist()
            # nonzero_hotkey_to_accuracy = {
            #     k: v for k, v in hotkey_to_accuracy.items() if v != 0
            # }
            torch.save(
                {
                    "scores": scores,
                },
                cls.get_validator_state_filepath(),
            )

    @classmethod
    async def validator_load(cls):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        async with cls._validator_lock:
            cls._ensure_paths_exist()
            try:
                # Load the state of the validator from file.
                state = torch.load(cls.get_validator_state_filepath())
                return True, state["scores"]
            except FileNotFoundError:
                bt.logging.error("Validator state file not found.")
                return False, None
