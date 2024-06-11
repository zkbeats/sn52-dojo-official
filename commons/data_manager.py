import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, List, Optional

import torch
from commons.objects import ObjectManager
from loguru import logger
from strenum import StrEnum
from template.protocol import DendriteQueryResponse, FeedbackRequest


class ValidatorStateKeys(StrEnum):
    SCORES = "scores"
    DOJO_TASKS_TO_TRACK = "dojo_tasks_to_track"


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
            logger.error(f"File not found at {path}... , exception:{e}")
            return None
        except pickle.PickleError as e:
            logger.error(f"Pickle error: {e}")
            return None
        except Exception:
            logger.error("Failed to load existing ranking data from file.")
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
                logger.success(f"Saved data to {path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            return False

    @classmethod
    async def save(cls, path, data: Any):
        try:
            async with cls._lock:
                with open(str(path), "wb") as file:
                    pickle.dump(data, file)
                    logger.success(f"Saved data to {path}")
                    return True
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            return False

    @classmethod
    async def save_dendrite_response(cls, response: DendriteQueryResponse):
        path = DataManager.get_requests_data_path()
        async with cls._lock:
            # ensure parent path exists
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            data = await DataManager._load_without_lock(path=path)
            if not data:
                # store initial data
                success = await DataManager._save_without_lock(path, [response])
                logger.debug(
                    f"Storing initial data for responses from dendrite query, is successful ? {success}"
                )
                return success

            # append our data, if the existing data exists
            assert isinstance(data, list)
            data.append(response)
            success = await DataManager._save_without_lock(path, data)
            logger.debug(
                f"Storing appended data for responses from dendrite query, is successful ? {success}"
            )
            return success

    @classmethod
    async def overwrite_miner_responses_by_request_id(
        cls, request_id: str, miner_responses: List[FeedbackRequest]
    ):
        async with cls._lock:
            _path = DataManager.get_requests_data_path()
            data = await cls._load_without_lock(path=_path)
            found_idx = next(
                (i for i, x in enumerate(data) if x.request.request_id == request_id),
                None,
            )
            data[found_idx].miner_responses = miner_responses
            # overwrite the data
            is_saved = await cls._save_without_lock(_path, data)
            return is_saved

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
                logger.info("No data found to remove responses...")
                return

            new_data = []

            for d in data:
                if d in responses:
                    logger.debug(
                        f"Found response to remove with request id: {d.request.request_id}"
                    )
                    continue
                new_data.append(d)

            await DataManager._save_without_lock(path, new_data)

    @classmethod
    async def validator_save(cls, scores, requestid_to_mhotkey_to_task_id):
        """Saves the state of the validator to a file."""
        logger.debug("Attempting to save validator state.")
        async with cls._validator_lock:
            cls._ensure_paths_exist()
            dojo_task_data = json.loads(json.dumps(requestid_to_mhotkey_to_task_id))
            if not dojo_task_data and torch.count_nonzero(scores).item() == 0:
                raise ValueError("Dojo task data and scores are empty. Skipping save.")

            torch.save(
                {
                    ValidatorStateKeys.SCORES: scores,
                    ValidatorStateKeys.DOJO_TASKS_TO_TRACK: json.loads(
                        json.dumps(requestid_to_mhotkey_to_task_id)
                    ),
                },
                cls.get_validator_state_filepath(),
            )

            logger.success(
                f"Saving validator state with scores: {scores}, and for {len(dojo_task_data)} request"
            )

    @classmethod
    async def validator_load(cls):
        """Loads the state of the validator from a file."""
        logger.info("Loading validator state.")
        async with cls._validator_lock:
            cls._ensure_paths_exist()
            try:
                state = torch.load(cls.get_validator_state_filepath())
                return state
            except FileNotFoundError:
                logger.error("Validator state file not found.")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
