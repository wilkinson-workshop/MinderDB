"""Collection of shared exceptions."""

class MinerException(Exception):
    """Raised for general MinderDB errors."""

    status_code: int | None = None
    message:     str | None = None

    def __init__(self, msg: str | None = None, status: int | None = None):
        self.message     = msg or self.message
        self.status_code = status or self.status_code

    def __str__(self):
        msg = self.message or ""
        if self.status_code is not None:
            msg += f" [{self.status_code}]"
        return msg


class RecordNotFound(MinerException):
    """
    Raised when a record or records could not be
    found.
    """

    def __init__(self, *args, **kwds):
        self.status_code = 404
        super().__init__(*args, **kwds)


class RecordExists(MinerException):
    """
    Raised when a record was already created.
    """

    def __init__(self, *args, **kwds):
        self.status_code = 409
        super().__init__(*args, **kwds)
