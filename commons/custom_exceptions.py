class InvalidNumberOfCompletions(Exception):
    def __init__(self, message):
        super().__init__(self, message)


class UnspecifiedNeuronType(Exception):
    def __init__(self, message):
        super().__init__(self, message)


class InvalidNeuronType(Exception):
    def __init__(self, message):
        super().__init__(self, message)


class InvalidScoreLength(Exception):
    def __init__(self, message):
        super().__init__(self, message)
