import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv

load_dotenv()

sys.path.append("./")
from doclm.processing import get_processor

class Test_parser_Profile(unittest.TestCase):
    def setUp(self) -> None:
        self.doc_obj=get_processor()
    
    def test_pptx_add_document(self):
        files = [
            {
                "remote_id": "taimoor/taimoor/architecture.pptx",
                "id": 1,
                "name": "architecture.pptx",
                "format": "pptx",
                "original_format":"pptx",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
             },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]
        self.doc_obj.add_document(files)

    def test_html_add_document(self):

        files = [
            {
                "remote_id": "taimoor/taimoor/index.html",
                "id": 2,
                "name": "index.html",
                "format": "html",
                "original_format":"html",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
             },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]
        self.doc_obj.add_document(files)

    def test_csv(self):
        files = [
            {
                "remote_id": "taimoor/taimoor/complete_results.csv",
                'id': 3,
                'name': 'complete_results.csv',
                'format': 'csv',
                'original_format': 'csv',
                "processing_step": 0,
                'tenant_id':2
             },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]
        self.doc_obj.add_document(files)

    def test_xlsx(self):
        # file_path = 'taimoor/taimoor/Expert Advisor Competitors Pricing.xlsx'
        self.files = [
            {
             "remote_id": 'taimoor/taimoor/Expert Advisor Competitors Pricing.xlsx', 
             'id': 4,
             'name': 'Expert Advisor Competitors Pricing.xlsx',
             'format': 'xlsx',
             'original_format': 'xslx',
             "processing_step": 0,
             'tenant_id':2
            },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]

        self.doc_obj.add_document(self.files)

    def test_pdf(self):
        files = [
            {
                "remote_id": "taimoor/taimoor/Amarex KRT; Motor F; DN40-300, 50 Hz, CE-DE 3.pdf",
                "id": 1,
                "name": "PHL Centrifugal pump",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
             },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]

        self.doc_obj.add_document(files)

    
    # def tearDown(self):
    #     del self.doc_obj
    #     return super().tearDown()

    # def test_structured_parser(self):
    #     from doclm.preprocessing.simple_processing import simple_doc_processor
    #     from doclm.file_loader import load_reader
    #     initial_num_pages = 3
    #     # structured_doc_processor(stream, file_path, initial_num_pages, source_tag):
    #     file_path = '../test_doc/2.1.6 Kunstharze, Novalok S.30-31 (6).pdf'
    #     # file_path = 'flowserve/71569212_EN_AQ.pdf'
    #     # reader = load_reader("azure_blob")
    #     reader = load_reader("local")
    #     stream = reader(file_path, None)
    #     # stream = open(file_path, "rb")
    #     initial_text, docs, meta_info = simple_doc_processor(
    #         stream, file_path, initial_num_pages, MetaExtractor.source_tag
    #     )
    #     print(docs)
    #     with open('.result') as f:
    #         f.write(docs)
    #     stream.close()
    #     # val = structured_doc_processor()
    #
    # def test_simple_ppt_processor(self):
    #     from doclm.preprocessing.ppt import document_processor
    #     initial_text, docs, meta_info = document_processor(
    #         "/home/in01-nbk-741/Downloads/architecture.pptx",
    #         "/home/in01-nbk-741/Downloads/architecture.pptx",
    #         3,
    #         "source",
    #     )
if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()