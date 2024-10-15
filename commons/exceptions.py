class NoNewUnexpiredTasksYet(Exception):
    """Exception raised when no unexpired tasks are found for processing."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class UnexpiredTasksAlreadyProcessed(Exception):
    """Exception raised when all unexpired tasks have already been processed."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidValidatorRequest(Exception):
    """Exception raised when a miner response is invalid."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidMinerResponse(Exception):
    """Exception raised when a miner response is invalid."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidCompletion(Exception):
    """Exception raised when a completion response is invalid."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidTask(Exception):
    """Exception raised when a task is invalid."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class EmptyScores(Exception):
    """Exception raised when scores are invalid."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CreateTaskFailed(Exception):
    """Exception raised when creating a task fails."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
