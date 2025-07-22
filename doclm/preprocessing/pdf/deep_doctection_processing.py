import os
import logging
import tempfile
from contextlib import contextmanager
from typing import List, Union


from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from PyPDF2 import PdfReader

# from deepdoctection.dataflow.custom import CustomDataFromIterable
from deepdoctection.pipe.transform import SimpleTransformService
from deepdoctection import Jdeskewer
from deepdoctection.dataflow.serialize import DataFromList
from deepdoctection.utils.detection_types import ImageType
import deepdoctection as dd

from .util import get_file_info_from_reader, timeit

log = logging.getLogger("doclogger")
log.disabled = False

deep_doc_logger = logging.getLogger('deepdoctection.utils.logger')

# this is to add same logging handel  to deep doctection
for h in log.handlers:
    deep_doc_logger.addHandler(h)


class CustomJdeskewer(Jdeskewer):
    """
    Deskew an image following <https://phamquiluan.github.io/files/paper2.pdf>. It allows to determine that deskew angle
    up to 45 degrees and provides the corresponding rotation so that text lines range horizontally.
    """

    def transform(self, np_img: ImageType) -> ImageType:
        angle = get_angle(np_img)

        if abs(angle) > self.min_angle_rotation:
            return rotate(np_img, angle)
        return np_img


class DDPdfReader:
    def __init__(self):
        self._DD_ONE = os.path.join('/'.join(os.path.realpath(__file__).split('/')[:-1]), "conf_dd_one.yaml")
        self.TESS_CONF = os.path.join('/'.join(os.path.realpath(__file__).split('/')[:-1]), "conf_tesseract.yaml")

        self.cfg = dd.set_config_by_yaml(self._DD_ONE)
        self.cfg.freeze(freezed=False)
        self.cfg.DEVICE = "cpu"
        self.cfg.freeze()

        # layout detector
        # TODO: add a better layout detector
        self.layout_config_path = dd.ModelCatalog.get_full_path_configs(self.cfg.CONFIG.D2LAYOUT)
        self.layout_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(self.cfg.WEIGHTS.D2LAYOUT)
        self.categories_layout = dd.ModelCatalog.get_profile(self.cfg.WEIGHTS.D2LAYOUT).categories
        # assert self.categories_layout is not None
        # assert self.layout_weights_path is not None
        self.d_layout = dd.D2FrcnnDetector(self.layout_config_path, self.layout_weights_path, self.categories_layout,
                                           device=self.cfg.DEVICE)

        # cell detector
        # TODO: Try microsoft table detection instead of this
        self.cell_config_path = dd.ModelCatalog.get_full_path_configs(self.cfg.CONFIG.D2CELL)
        self.cell_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(self.cfg.WEIGHTS.D2CELL)
        self.categories_cell = dd.ModelCatalog.get_profile(self.cfg.WEIGHTS.D2CELL).categories
        # assert categories_cell is not None
        self.d_cell = dd.D2FrcnnDetector(self.cell_config_path, self.cell_weights_path, self.categories_cell,
                                         device=self.cfg.DEVICE)

        # row/column detector
        self.item_config_path = dd.ModelCatalog.get_full_path_configs(self.cfg.CONFIG.D2ITEM)
        self.item_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(self.cfg.WEIGHTS.D2ITEM)
        self.categories_item = dd.ModelCatalog.get_profile(self.cfg.WEIGHTS.D2ITEM).categories
        # assert categories_item is not None
        self.d_item = dd.D2FrcnnDetector(self.item_config_path, self.item_weights_path, self.categories_item,
                                         device=self.cfg.DEVICE)

        # language detection model
        self.categories_language = dd.ModelCatalog.get_profile("fasttext/lid.176.bin").categories
        self.language_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs("fasttext/lid.176.bin")
        self.fast_lang = dd.FasttextLangDetector(self.language_weights_path, self.categories_language)

        # skew straightening
        # TODO: add a custom class here for better skew adjustment
        self.im_skew = CustomJdeskewer()

        # pdf miner
        self.pdf_text = dd.PdfPlumberTextDetector()

        # ocr
        # self.tex_text = dd.TesseractOcrDetector(
        #     self.TESS_CONF, config_overwrite=[f"LANGUAGES=eng"]
        # )
        if self.cfg.USE_DOCTR:
            doctr_det_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(
                self.cfg.WEIGHTS.DOCTR_WORD)
            doctr_det_profile = dd.ModelCatalog.get_profile(self.cfg.WEIGHTS.DOCTR_WORD)

            self.doctr_word = dd.DoctrTextlineDetector(doctr_det_profile.architecture, doctr_det_weights_path,
                                                       doctr_det_profile.categories, self.cfg.DEVICE, lib='PT')
            # _build_doctr_word(cfg)

            doctr_rec_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(
                self.cfg.WEIGHTS.DOCTR_RECOGNITION)
            doctr_rec_profile = dd.ModelCatalog.get_profile(self.cfg.WEIGHTS.DOCTR_RECOGNITION)
            self.doctr_ocr = dd.DoctrTextRecognizer(doctr_rec_profile.architecture, doctr_rec_weights_path,
                                                    self.cfg.DEVICE, lib='PT')
            # _build_ocr(cfg)
            # skip_if_text_extracted = cfg.USE_PDF_MINER

        else:
            self.tex_text = dd.TesseractOcrDetector(
                self.TESS_CONF, config_overwrite=[f"LANGUAGES=eng"]
            )

        self.analyzer_checker = self._build_checker()
        self.analyzer_pdfminer = self._build_pdf_reader_analyzer()
        self.analyzer_ocr = self._build_ocr_analyzer()

    def _build_checker(self):
        self.cfg.freeze(freezed=False)
        self.cfg.TAB = True
        self.cfg.TAB_REF = True
        self.cfg.OCR = True
        self.cfg.freeze()

        pipe_component_list = []

        d_text = dd.TextExtractionService(self.pdf_text)
        pipe_component_list.append(d_text)

        page_parsing = dd.PageParsingService(text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER,
                                             floating_text_block_categories=self.cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK,
                                             include_residual_text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK)
        pipe = dd.DoctectionPipe(pipeline_component_list=pipe_component_list,
                                 page_parsing_service=page_parsing)
        return pipe

    def _build_pdf_reader_analyzer(self):
        self.cfg.freeze(freezed=False)
        self.cfg.TAB = True
        self.cfg.TAB_REF = True
        self.cfg.OCR = True
        self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK = True
        self.cfg.freeze()

        pipe_component_list = []

        layout = dd.ImageLayoutService(self.d_layout, to_image=True, crop_image=True)
        pipe_component_list.append(layout)

        nms_service = dd.AnnotationNmsService(nms_pairs=self.cfg.LAYOUT_NMS_PAIRS.COMBINATIONS,
                                              thresholds=self.cfg.LAYOUT_NMS_PAIRS.THRESHOLDS)
        pipe_component_list.append(nms_service)

        if self.cfg.TAB:

            detect_result_generator = dd.DetectResultGenerator(self.categories_cell)
            cell = dd.SubImageLayoutService(self.d_cell, dd.LayoutType.table, {1: 6}, detect_result_generator)
            pipe_component_list.append(cell)

            detect_result_generator = dd.DetectResultGenerator(self.categories_item)
            item = dd.SubImageLayoutService(self.d_item, dd.LayoutType.table, {1: 7, 2: 8}, detect_result_generator)
            pipe_component_list.append(item)

            table_segmentation = dd.TableSegmentationService(
                self.cfg.SEGMENTATION.ASSIGNMENT_RULE,
                self.cfg.SEGMENTATION.THRESHOLD_ROWS,
                self.cfg.SEGMENTATION.THRESHOLD_COLS,
                self.cfg.SEGMENTATION.FULL_TABLE_TILING,
                self.cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                self.cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                self.cfg.SEGMENTATION.STRETCH_RULE
            )
            pipe_component_list.append(table_segmentation)

            if self.cfg.TAB_REF:
                table_segmentation_refinement = dd.TableSegmentationRefinementService()
                pipe_component_list.append(table_segmentation_refinement)

        if self.cfg.OCR:
            match_words = dd.MatchingService(
                parent_categories=self.cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
                child_categories=self.cfg.WORD_MATCHING.CHILD_CATEGORIES,
                matching_rule=self.cfg.WORD_MATCHING.RULE,
                threshold=self.cfg.WORD_MATCHING.THRESHOLD,
                max_parent_only=self.cfg.WORD_MATCHING.MAX_PARENT_ONLY
            )
            pipe_component_list.append(match_words)

            order = dd.TextOrderService(
                text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER,
                floating_text_block_categories=self.cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK,
                text_block_categories=self.cfg.TEXT_ORDERING.TEXT_BLOCK,
                include_residual_text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK)
            pipe_component_list.append(order)

        page_parsing = dd.PageParsingService(text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER,
                                             floating_text_block_categories=self.cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK,
                                             include_residual_text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK)
        pipe = dd.DoctectionPipe(pipeline_component_list=pipe_component_list,
                                 page_parsing_service=page_parsing)
        return pipe

    def _build_ocr_analyzer(self):
        self.cfg.freeze(freezed=False)
        self.cfg.TAB = True
        self.cfg.TAB_REF = True
        self.cfg.OCR = True
        self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK = False
        self.cfg.freeze()

        pipe_component_list = []

        skew = SimpleTransformService(self.im_skew)
        pipe_component_list.append(skew)

        layout = dd.ImageLayoutService(self.d_layout, to_image=True, crop_image=True)
        pipe_component_list.append(layout)

        nms_service = dd.AnnotationNmsService(nms_pairs=self.cfg.LAYOUT_NMS_PAIRS.COMBINATIONS,
                                              thresholds=self.cfg.LAYOUT_NMS_PAIRS.THRESHOLDS)
        pipe_component_list.append(nms_service)

        if self.cfg.TAB:

            detect_result_generator = dd.DetectResultGenerator(self.categories_cell)
            cell = dd.SubImageLayoutService(self.d_cell, dd.LayoutType.table, {1: 6}, detect_result_generator)
            pipe_component_list.append(cell)

            detect_result_generator = dd.DetectResultGenerator(self.categories_item)
            item = dd.SubImageLayoutService(self.d_item, dd.LayoutType.table, {1: 7, 2: 8}, detect_result_generator)
            pipe_component_list.append(item)

            table_segmentation = dd.TableSegmentationService(
                self.cfg.SEGMENTATION.ASSIGNMENT_RULE,
                self.cfg.SEGMENTATION.THRESHOLD_ROWS,
                self.cfg.SEGMENTATION.THRESHOLD_COLS,
                self.cfg.SEGMENTATION.FULL_TABLE_TILING,
                self.cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                self.cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                self.cfg.SEGMENTATION.STRETCH_RULE
            )
            pipe_component_list.append(table_segmentation)

            if self.cfg.TAB_REF:
                table_segmentation_refinement = dd.TableSegmentationRefinementService()
                pipe_component_list.append(table_segmentation_refinement)

        if self.cfg.OCR:
            if self.cfg.USE_DOCTR:
                word = dd.ImageLayoutService(self.doctr_word, to_image=True, crop_image=True,
                                             skip_if_layout_extracted=True)
                pipe_component_list.append(word)

                extract_from_roi = dd.LayoutType.word
                text = dd.TextExtractionService(
                    self.doctr_ocr, skip_if_text_extracted=False, extract_from_roi=extract_from_roi
                )
                pipe_component_list.append(text)
            else:
                f_lang = dd.LanguageDetectionService(self.fast_lang, text_detector=self.tex_text)
                pipe_component_list.append(f_lang)

                t_text = dd.TextExtractionService(self.tex_text, skip_if_text_extracted=True,
                                                  run_time_ocr_language_selection=True)
                pipe_component_list.append(t_text)

            match_words = dd.MatchingService(
                parent_categories=self.cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
                child_categories=self.cfg.WORD_MATCHING.CHILD_CATEGORIES,
                matching_rule=self.cfg.WORD_MATCHING.RULE,
                threshold=self.cfg.WORD_MATCHING.THRESHOLD,
                max_parent_only=self.cfg.WORD_MATCHING.MAX_PARENT_ONLY
            )
            pipe_component_list.append(match_words)

            order = dd.TextOrderService(
                text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER,
                floating_text_block_categories=self.cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK,
                text_block_categories=self.cfg.TEXT_ORDERING.TEXT_BLOCK,
                include_residual_text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK)
            pipe_component_list.append(order)

        page_parsing = dd.PageParsingService(text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER,
                                             floating_text_block_categories=self.cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK,
                                             include_residual_text_container=self.cfg.TEXT_ORDERING.TEXT_CONTAINER_TO_TEXT_BLOCK)
        pipe = dd.DoctectionPipe(pipeline_component_list=pipe_component_list,
                                 page_parsing_service=page_parsing)
        return pipe

    def analyze_image(self, img, pdf):

        df_checker = self.analyzer_checker.analyze(path=pdf)
        df_checker.reset_state()
        text_categories = self.analyzer_checker.pipe_component_list[0].predictor.possible_categories()
        dps = []
        # documents = []
        # dpts = []
        # html_dict = {}
        initial_text = ''
        idx = 0
        for dp_checker in df_checker:
            text_anns = dp_checker.get_annotation(category_names=text_categories)
            if text_anns:
                df = DataFromList(lst=[dp_checker])
                df = self.analyzer_pdfminer.analyze(dataset_dataflow=df)
            else:
                df = DataFromList(lst=[dp_checker])
                df = self.analyzer_ocr.analyze(dataset_dataflow=df)
            df.reset_state()
            # dps.append(next(iter(df)))
            try:
                dp = next(iter(df))
            except Exception as e:
                log.error(e, exc_info=True)

            # for idx, dp in enumerate(dps):
            # html_dict[idx + 1] = {}
            # dpts.append(dp)

            page_content = ''
            # page_number = dp.page_number + 1

            layout_items = [layout for layout in dp.layouts if layout.reading_order is not None]
            layout_items.sort(key=lambda x: x.reading_order)
            for item in layout_items:
                page_content += f"{item.text if item.category_name != 'table' else item.csv}\n"
            # document = Document(
            #     page_content=page_content,
            #     metadata={
            #         # source_tag: file_path,
            #         "page": int(page_number)},
            # )
            # documents.append(document)
            # for table in dp.tables:
            #     html_dict[idx + 1][table.annotation_id] = table.html
            # if page_number <= initial_num_pages:
            #     initial_text += page_content
            # idx += 1
            yield page_content
        # return [dp.viz(show_cells=False) for dp in dpts], html_dict, documents, initial_text
        # return documents, initial_text


@contextmanager
def temporary_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, 'temp_file.pdf')
        try:
            yield temp_file_path
        except Exception as e:
            log.error(e, exc_info=True)
            raise e
        # finally:
        #     try:
        #         temp_file_path
        #     except Exception as e:
        #         log.error(e, exc_info=True)
        #         raise e


dd_reader: Union[DDPdfReader, None] = None


def deep_doctection_pdf_processor(stream, **kwargs):
    global dd_reader

    pdf_reader = PdfReader(stream)
    meta_info = get_file_info_from_reader(pdf_reader)

    if dd_reader is None:
        dd_reader = DDPdfReader()
    with temporary_file() as temp_path:
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(stream.read())
        return meta_info, dd_reader.analyze_image(None, temp_path)
    # stream.seek(0)

    # log.info("Meta data Extracted from %s", file_path)
    # return initial_text, documents, meta_info
