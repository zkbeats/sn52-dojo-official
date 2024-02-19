from neurons.miner import Miner


class MinerFactory:
    _miner_instance = None

    @classmethod
    def get_miner(cls):
        if cls._miner_instance is None:
            cls._miner_instance = Miner()
        return cls._miner_instance
