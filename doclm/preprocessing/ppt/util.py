"""
    utility functions to get metadata, to be displayed at front-end user
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..splitters import (
    custom_uniform_splitter,
    additional_processing,
)
# from langchain.text_splitter import TokenTextSplitter

# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 125


# pylint: disable=C0103,C0116
def get_file_info_from_reader(reader):
    fields = ["author", "creator", "producer", "subject", "title"]
    info = {}
    if (
        not reader
        or not hasattr(reader, "core_properties")
        or not reader.core_properties
    ):
        for f in fields:
            info[f] = None
        return info

    return get_file_info_from_meta_data(reader.core_properties, fields, info)


def get_file_info_from_meta_data(metadata, fields, info):
    meta = metadata
    for f in fields:
        info[f] = getattr(meta, f) if hasattr(meta, f) else None
    return info

def text_simple_splitter(data, chunk_size, chunk_overlap):
    docs = custom_uniform_splitter(data, chunk_size, chunk_overlap)
    additional_processing(docs)
    return docs

# def text_splitter(data, chunk_size, chunk_overlap):
#     # text_split = TokenTextSplitter(
#     text_split = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=[". ", ""],
#     )
#     docs = text_split.split_documents(data)
#     for idx, doc in enumerate(docs):
#         doc.metadata["chunk_num"] = idx + 1
#         doc.metadata["chunk_len_in_chars"] = len(doc.page_content)
#         if idx > 0:
#             overlap_length = len(
#                 _overlap_between_strings(docs[idx - 1].page_content, doc.page_content)
#             )
#             doc.metadata["previous_chunk_text"] = docs[idx - 1].page_content[
#                 : len(docs[idx - 1].page_content) - overlap_length
#             ]
#         if idx + 1 < len(docs):
#             overlap_length = len(
#                 _overlap_between_strings(doc.page_content, docs[idx + 1].page_content)
#             )
#             doc.metadata["next_chunk_text"] = docs[idx + 1].page_content[
#                 overlap_length:
#             ]
#     return docs


def _overlap_between_strings(current_str, next_str):
    for ele in range(1, len(current_str)):
        if next_str.startswith(current_str[ele:]):
            return current_str[ele:]
    return ""
