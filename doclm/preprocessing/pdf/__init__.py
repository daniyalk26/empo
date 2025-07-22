"""
this module select between simple, simple_ocr and structure aware parser base on environment variable
"""
# from . import deep_doctection_pdf_processor
from .simple_ocr_processing import simple_ocr_processor as simple_ocr_pdf_processor
from .structure_aware_processing import structured_doc_processor
from .simple_processing import simple_pdf_processor
from .util import text_simple_splitter, text_structured_splitter
