"""
module contains implementation to extract data from pdf documents
"""
import logging
from PyPDF2 import PdfReader

from .util import get_file_info_from_reader, make_generator
from ...exceptions import ScannedDocException

log = logging.getLogger("doclogger")


def simple_pdf_processor(stream, **kwargs):
    """
    extracts data from pdf documents  in the simplest form
    :param stream: BytesIO object
    :param file_path: file path/ remote_id
    :param initial_num_pages: pages for summary/title extraction
    :param source_tag: tage in db
    :return:
    """
    pdf_reader = PdfReader(stream)
    meta_info = get_file_info_from_reader(pdf_reader)
    log.info("Meta data Extracted from")

    # log.debug("extracting text form PDF_reader file %s", file_path)
    pages = [page.extract_text() for page in pdf_reader.pages]

    if sum([len(page) for page in pages]) == 0:
        raise ScannedDocException('No text could be extracted using simple reader. Probably a scanned doc.')

    return meta_info, make_generator(pages)
