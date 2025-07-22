"""
    utility functions
"""
import logging
from functools import wraps
import time



def make_generator(pages):
    for p in pages:
        if hasattr(p, 'render'):
            yield p.render()
            continue
        yield p

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        log.info("Function %s Took %s seconds", func.__name__, total_time)
        return result

    return timeit_wrapper


from ..splitters import (
    custom_uniform_splitter,
    custom_uniform_structured_splitter,
    additional_processing,
)

#
# CHUNK_SIZE = 1200
# CHUNK_OVERLAP = 250

log = logging.getLogger("doclogger")
log.disabled = False


# pylint: disable=C0103,C0116


def get_file_info_from_reader(reader):
    fields = ["author", "creator", "producer", "subject", "title"]
    info = {}
    if not reader or not hasattr(reader, "metadata") or not reader.metadata:
        for f in fields:
            info[f] = None
        info["pages"] = None
        return info
    try:
        info["pages"] = len(reader.pages) if hasattr(reader, "pages") else None
    except Exception as e:
        log.error(e, exc_info=True)

    return get_file_info_from_meta_data(reader.metadata, fields, info)


def get_file_info_from_meta_data(metadata, fields, info):
    meta = metadata
    for f in fields:
        try:
            info[f] = getattr(meta, f) if hasattr(meta, f) else None
        except Exception as e:
            log.error(e, exc_info=True)
    return info


# TODO: Future use better text splitter
def text_simple_splitter(data, chunk_size, chunk_overlap):
    docs = custom_uniform_splitter(data, chunk_size, chunk_overlap)
    additional_processing(docs)
    return docs


def text_structured_splitter(data, chunk_size, chunk_overlap):
    docs = custom_uniform_structured_splitter(data, chunk_size, chunk_overlap)
    additional_processing(docs)
    return docs


