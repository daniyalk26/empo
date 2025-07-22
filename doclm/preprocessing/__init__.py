"""
  model implement utility functions
"""
import os
import logging
from typing import Any, Dict, List

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document

from .pdf import text_simple_splitter as pdf_simple_text_splitter
from .pdf import text_structured_splitter as pdf_structured_text_splitter
from .pdf import structured_doc_processor as pdf_structured_processor
from .pdf import simple_pdf_processor as pdf_simple_processor
from .pdf import simple_ocr_pdf_processor as pdf_simple_ocr_processor

from .ppt import simple_ppt_processor as ppt_simple_processor
from .ppt import text_splitter as ppt_text_splitter
from .excel.xlsx import xlsx_processor
from .excel.xls import xls_processor
from .csv import simple_processor as csv_simple_processor
from .html import html_processor
from .splitters import markdown_splitter_wrapper, table_text_splitter as xlsx_csv_simple_text_splitter
from ..schema import Schema


log = logging.getLogger("doclogger")
log.disabled = False

pdf_parser_type = os.getenv("PARSER_TYPE")
chunk_size = int(os.getenv("CHUNK_SIZE", "1600"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))

pdf_processor_dict = {"simple": pdf_simple_processor,
                      "simple_ocr": pdf_simple_ocr_processor,
                      "structure": pdf_structured_processor
                      }

pdf_splitter_dict = {
    "simple": pdf_simple_text_splitter,
    "structure": pdf_structured_text_splitter
}


def get_pdf_processor(_type=None, **kwargs):
    _parser_type = _type or pdf_parser_type
    if _parser_type == 'smart':
        from .pdf.deep_doctection_processing import deep_doctection_pdf_processor
        return deep_doctection_pdf_processor
    if _parser_type == 'simple_ocr':
        if kwargs.get('parser_type') == 0:
            _parser_type = 'simple'
    if _parser_type not in pdf_processor_dict:
        _parser_type = 'simple'
    return pdf_processor_dict[_parser_type]


def get_ppt_processor(**kwargs):
    return ppt_simple_processor


def get_pdf_text_splitter(_type=None):
    _splitter_type = _type or pdf_parser_type
    if _splitter_type not in pdf_splitter_dict:
        _splitter_type = 'simple'
    return pdf_splitter_dict.get(_splitter_type)


def get_ppt_text_splitter():
    return ppt_text_splitter


extension_specific_processor = {
    "PDF": get_pdf_processor,
    "PPTX": get_ppt_processor,
    "XLSX": lambda **kwargs: xlsx_processor,
    "XLS": lambda **kwargs: xls_processor,
    "CSV": lambda **kwargs: csv_simple_processor,
    "HTML": lambda **kwargs:html_processor,
    }

extension_specific_splitter = {
    "PDF": get_pdf_text_splitter(),
    "PPTX": get_ppt_text_splitter(),
    "XLSX": xlsx_csv_simple_text_splitter,
    "XLS": xlsx_csv_simple_text_splitter,
    "CSV": xlsx_csv_simple_text_splitter,
    "HTML": markdown_splitter_wrapper,
}


# pylint: disable=C0103,C0116
def add_metadata(meta_datas, keys):
    def add(document):
        for key in keys:
            if key in meta_datas:
                assert isinstance(meta_datas[key], (str, int, float, list)
                                  ), f'{key} type {type(meta_datas[key])} is not supported'
                document.metadata[key] = meta_datas[key]
        return document

    return add


def enrich_split_document(
        data: List[Document], file_type: str, extracted_meta: Dict[str, Any]
):
    log.debug("Extracted metadata %s", extracted_meta)
    # data = list(map(add_metadata(extracted_meta, extracted_meta.keys()), data))
    data = list(map(add_metadata(extracted_meta, ['lang', Schema.source_tag]), data))

    log.debug("Splitting data")
    text_splitter = extension_specific_splitter[file_type]
    documents = None
    try:
        documents = text_splitter(data, chunk_size, chunk_overlap)
    except Exception as e_x:
        log.error(e_x, exc_info=True)
        raise Exception(e_x) from e_x

    texts = [doc.page_content for doc in documents]
    meta_data = [doc.metadata for doc in documents]

    return texts, meta_data


def document_processor(stream, file_path, file_type, initial_text_size, source_tag, **kwargs):
    """

    """
    initial_text = []
    docs = []
    log.debug("extracting text form file %s", file_path)
    extractor = extension_specific_processor[file_type](**kwargs)
    meta_info, the_iterator = extractor(stream)
    if file_type in ["XLSX", "XLS"]:
        initial_text_size = int(initial_text_size / meta_info.get('pages', 1))

    for page_number, page_text in enumerate(the_iterator, start=1):
        name = None
        if isinstance(page_text, tuple):
            name, page_text = page_text
        page_text = page_text.replace('\x00','')              # remove null characters if present

        doc = Document(
            page_content=page_text,
            metadata={
                # source_tag: file_path,
                "page": str(page_number),
                "page_name": name,
                source_tag: file_path,
            },

        )
        docs.append(doc)

        if len(initial_text) < initial_text_size:
            text_form_page =  page_text.split(",") if file_type == "XLSX" else page_text.split(" ")
            if name:
                initial_text += [name]
            initial_text += text_form_page[:initial_text_size]

    initial_text = ",".join(initial_text) if file_type == "XLSX" else " ".join(initial_text)
    return initial_text, docs, meta_info


# def document_processor(stream, file_path, file_type, initial_num_pages, source_tag, **kwargs):
#     assert file_type in extension_specific_processor, KeyError(
#         f"{file_type} file type is not supported yet"
#     )
#
#     _doc_processor = extension_specific_processor[file_type](**kwargs)
#     return _doc_processor(stream, file_path, initial_num_pages, source_tag, **kwargs)


def get_supported_extension(format_name: str):
    if format_name.lower() == "pdf":
        return "PDF"
    if (format_name.lower() == "ppt") or (format_name.lower() == "pptx"):
        return "PPTX"
    if format_name.lower() == "xlsx" :# or (format_name.lower() == "xls"):
        return "XLSX"
    if format_name.lower() == "csv":
        return "CSV"
    if format_name.lower() == "xls":
        return "XLS"
    if format_name.lower() == "htm" or format_name.lower() == "html":
        return "HTML"

    raise ValueError(f"{format_name} not supported")
