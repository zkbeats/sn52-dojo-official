from commons.dataset import get_seed_dataset
from commons.human_feedback.aws_mturk import get_aws_client
from neurons.miner import Miner
from neurons.validator import Validator


class Factory:
    _miner_instance = None
    _validator_instance = None
    _aws_client = None
    _seed_dataset_iter = None

    @classmethod
    def get_miner(cls):
        if cls._miner_instance is None:
            cls._miner_instance = Miner()
        return cls._miner_instance

    @classmethod
    def get_validator(cls):
        if cls._validator_instance is None:
            cls._validator_instance = Validator()
        return cls._validator_instance

    @classmethod
    def get_aws_client(cls):
        if cls._aws_client is None:
            # TODO: Change the environment to production when launch
            cls._aws_client = get_aws_client("sandbox")
        return cls._aws_client

    @classmethod
    def get_seed_dataset_iterator(cls):
        if cls._seed_dataset_iter is None:
            cls._seed_dataset_iter = iter(get_seed_dataset())
        return cls._seed_dataset_iter

    @classmethod
    def new_seed_dataset_iterator(cls):
        cls._seed_dataset_iter = iter(get_seed_dataset())
        return cls._seed_dataset_iter
