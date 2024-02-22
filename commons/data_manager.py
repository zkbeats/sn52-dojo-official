import asyncio
from pathlib import Path
import pickle
from typing import Any, List, Optional
from commons.factory import Factory
from commons.objects import DendriteQueryResponse
import bittensor as bt
# import aiofiles

__LOG_PREFIX__ = "[DataManager]"


class DataManager:
    _lock = asyncio.Lock()

    @staticmethod
    def get_ranking_data_filepath() -> Path:
        config = Factory.get_config()
        base_path = config.data_manager.base_path
        return base_path / "data" / "ranking" / "data.pkl"

    @classmethod
    async def _load_without_lock(cls, path) -> Optional[List[DendriteQueryResponse]]:
        try:
            with open(str(path), "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            bt.logging.warning(
                __LOG_PREFIX__, f"File not found at {path}... , exception:{e}"
            )
            return None
        except pickle.PickleError as e:
            bt.logging.error(__LOG_PREFIX__, f"Pickle error: {e}.")
            return None
        except Exception:
            bt.logging.error(
                __LOG_PREFIX__, "Failed to load existing ranking data from file."
            )
            return None

    @classmethod
    async def load(cls, path) -> Optional[List[DendriteQueryResponse]]:
        # Load the list of Pydantic objects from the pickle file
        async with cls._lock:
            return cls._load_without_lock(path)

    @classmethod
    async def _save_without_lock(cls, path, data: Any):
        try:
            with open(str(path), "wb") as file:
                pickle.dump(data, file)
                bt.logging.debug(__LOG_PREFIX__, f"Saved data to {path}")
                return True
        except Exception as e:
            bt.logging.error(__LOG_PREFIX__, f"Failed to save data to file: {e}")
            return False

    @classmethod
    async def save(cls, path, data: Any):
        try:
            async with cls._lock:
                with open(str(path), "wb") as file:
                    pickle.dump(data, file)
                    bt.logging.debug(__LOG_PREFIX__, f"Saved data to {path}")
                    return True
        except Exception as e:
            bt.logging.error(__LOG_PREFIX__, f"Failed to save data to file: {e}")
            return False

    @staticmethod
    async def append_responses(response: DendriteQueryResponse):
        path = DataManager.get_ranking_data_filepath()
        # ensure parent path exists
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        data = await DataManager.load(path=path)
        if not data:
            # store initial data
            await DataManager.save(path, [response])
            return

        # append our data
        assert isinstance(data, list)
        data.append(response)
        await DataManager.save(path, data)

        return

    @classmethod
    async def remove_responses(
        cls, responses: List[DendriteQueryResponse]
    ) -> Optional[DendriteQueryResponse]:
        path = DataManager.get_ranking_data_filepath()
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
