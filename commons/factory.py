from commons.dataset import get_seed_dataset
from commons.human_feedback.aws_mturk import get_aws_client
from template.utils.config import get_config


class Factory:
    _miner = None
    _validator = None
    _aws_client = None
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
    def get_aws_client(cls):
        if cls._aws_client is None:
            # TODO: Change the environment to production when launch
            cls._aws_client = get_aws_client("sandbox")
        return cls._aws_client

    @classmethod
    def get_seed_dataset_iter(cls):
        if cls._seed_dataset_iter is None:
            cls._seed_dataset_iter = iter(get_seed_dataset())
        return cls._seed_dataset_iter

    @classmethod
    def new_seed_dataset_iter(cls):
        cls._seed_dataset_iter = iter(get_seed_dataset())
        return cls._seed_dataset_iter

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = get_config()
        return cls._config
