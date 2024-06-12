from template.utils.config import get_config


class ObjectManager:
    _miner = None
    _validator = None
    _seed_dataset_iter = None
    _config = None

    @classmethod
    def get_miner(cls):
        from neurons.miner import Miner

        if cls._miner is None:
            cls._miner = Miner()
        return cls._miner

    @classmethod
    def get_validator(cls):
        from neurons.validator import Validator

        if cls._validator is None:
            cls._validator = Validator()
        return cls._validator

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = get_config()
        return cls._config
