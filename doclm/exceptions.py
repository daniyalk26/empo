class RetryableException(Exception):
    """Retry file if this exception occurs"""


class ScannedDocException(Exception):
    code="No text Found"
    """File may be scanned if this exception occurs"""


class NotSupportedException(Exception):
    def __init__(self, detected_langauge:str, code: str):
        super().__init__(detected_langauge)
        self.code = code


def get_error_message(e: Exception):
    code = None
    if hasattr(e, "code"):
        code_lo = getattr(e, 'code')
        code = f'{code_lo}'.upper()

    return {'code': code,
            'description': f'{e}'.upper()}
