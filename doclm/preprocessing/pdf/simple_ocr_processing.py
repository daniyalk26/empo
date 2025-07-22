import logging

from PyPDF2 import PdfReader

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from .util import get_file_info_from_reader, make_generator

log = logging.getLogger("doclogger")
log.disabled = False

deep_doc_logger = logging.getLogger('deepdoctection.utils.logger')

# this is to add same logging handel  to deep doctection
for h in log.handlers:
    deep_doc_logger.addHandler(h)


def apply_doctr_to_image(img):
    model = ocr_predictor(pretrained=True)
    #     single_img_doc = DocumentFile.from_images(img)
    result = model([img])
    return result.render()


def simple_ocr_processor(stream, **kwargs):
    pdf_reader = PdfReader(stream)
    meta_info = get_file_info_from_reader(pdf_reader)
    log.info("Meta data Extracted",)

    log.debug("performing ocr extracting text form PDF_reader file")
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(stream)

    result = model(doc)

    return meta_info, make_generator(result.pages)
