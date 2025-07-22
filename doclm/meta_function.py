from datetime import datetime
from langdetect import detect

# TODO: Use fast-text for language detection, remove langdetect from here
def detect_language(text, **kwargs):
    lang = detect(text)
    return lang


def get_timestamp(*arg, **kwargs):
    current_timestamp = datetime.now().astimezone()
    date_time = current_timestamp.isoformat()
    return date_time

def get_keyword(*arg, **kwargs):

    return []
