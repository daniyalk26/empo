import json
import time
import unittest
import asyncio

# import aiounittest
import os, sys
# from pyinstrument import Profiler
from dotenv import load_dotenv
# from memray import Tracker
# from memray._memray import size_fmt
# from rich import print as rprint
# from memray.reporters.tree import TreeReporter
# from memray import FileReader
from pathlib import Path
import logging

load_dotenv()
sys.path.append("./")
from doclm.doclm import get_interactive
from doclm.schema import Schema
from doclm.processing import get_processor

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

obj=get_interactive()
processor_obj=get_processor()
# from doclm.preprocessing.structure_aware_processing import structured_doc_processor

# from .structure_aware_processing import (
#     structured_doc_processor as document_processor,
# )

# docker run --name postgres --network bridge -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector
# docker run  -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 --link postgres dpage/pgadmin4
# docker run  -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 --link hungry_mcclintock dpage/pgadmin4


def store_message(q, r):
    return {"q": {"id": 309, "message": q}, "r": {"id": 310, "message": r}}


Files = [
    {"remote_id": "flowserve/71569212_EN_AQ.pdf", "id": 1, "name": "file1"},
    # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", "id": 2, "name": "file2"},
    # {"remote_id": "flowserve/71569195_DE_A4.pdf", "id": 3, "name": "file3"},
    # {"remote_id": "flowserve/PS-20-1_DE_A4.pdf", "id": 4, "name": "file4"},
]


# class Test_parser_Profile(unittest.TestCase):
#     def setUp(self) -> None:
#         pass

#     def test_structured_parser(self):
#         from doclm.file_loader import load_reader

#         profiler = Profiler(interval=0.001)
#         with profiler:
#             initial_num_pages = 3
#             # structured_doc_processor(stream, file_path, initial_num_pages, source_tag):
#             file_path = "./test_doc/00079593_EN_A4.pdf"
#             # file_path = 'flowserve/71569212_EN_AQ.pdf'
#             # reader = load_reader("azure_blob")
#             reader = load_reader("local")
#             stream = reader(file_path, None)
#             # stream = open(file_path, "rb")
#             initial_text, docs, meta_info = structured_doc_processor(
#                 stream, file_path, initial_num_pages, MetaExtractor.source_tag
#             )
#             stream.close()
#             # val = structured_doc_processor()

#         profiler.print()


class TestSyncDocumentProfiler(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [
            # {
            #     "remote_id": "taimoor/taimoor/2.1.6 Kunstharze, Novalok S.30-31.pdf",
            #     "id": 1,
            #     "name": "file1",
            # },
            # {"remote_id": "taimoor/taimoor/BA 1121+23_D.pdf", "id": 2, "name": "file1"},
            # {
            #     "remote_id": "taimoor/taimoor/BA 1121+24,01_D.pdf",
            #     "id": 3,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1127+15,01_D.pdf",
            #     "id": 4,
            #     "name": "file1",
            # },
            # {"remote_id": "taimoor/taimoor/BA 1216+49_D.pdf", "id": 5, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/BA 1241+03_D.pdf", "id": 6, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/BA 1246+05_D.pdf", "id": 7, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/BA 1247+05_D.pdf", "id": 8, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/BA 1248+04_D.pdf", "id": 9, "name": "file1"},
            # {
            #     "remote_id": "taimoor/taimoor/BA 1248+06_D.pdf",
            #     "id": 10,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1250+02_D.pdf",
            #     "id": 11,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1251+02_D.pdf",
            #     "id": 12,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1254+04_D.pdf",
            #     "id": 13,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1259+02_D.pdf",
            #     "id": 14,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1261+01_D.pdf",
            #     "id": 15,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1267+05_D.pdf",
            #     "id": 16,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1267+09_D.pdf",
            #     "id": 17,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1268+07_D.pdf",
            #     "id": 18,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1278+02_D.pdf",
            #     "id": 19,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1286+02_D.pdf",
            #     "id": 20,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1288+10_D.pdf",
            #     "id": 21,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1288+24_D.pdf",
            #     "id": 22,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1288+31_D.pdf",
            #     "id": 23,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1289+05_D.pdf",
            #     "id": 24,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1301+03_D.pdf",
            #     "id": 25,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1302+06_D.pdf",
            #     "id": 26,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1303+02_D.pdf",
            #     "id": 27,
            #     "name": "file1",
            # },
            # # {
            # #     "remote_id": "taimoor/taimoor/BA 1312+10,01_D.pdf",
            # #     "id": 28,
            # #     "name": "file1",
            # # },
            # # {
            # #     "remote_id": "taimoor/taimoor/BA 1313+07_D.pdf",
            # #     "id": 29,
            # #     "name": "file1",
            # # },
            # # {
            # #     "remote_id": "taimoor/taimoor/BA 1314+01_D.pdf",
            # #     "id": 30,
            # #     "name": "file1",
            # # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1314+02_D.pdf",
            #     "id": 31,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1567+10_D.pdf",
            #     "id": 32,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1567+11_D.pdf",
            #     "id": 33,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1577+04_D.pdf",
            #     "id": 34,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+46_D.pdf",
            #     "id": 35,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+52_D.pdf",
            #     "id": 36,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+54_D.pdf",
            #     "id": 37,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+68_D.pdf",
            #     "id": 38,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+75_D.pdf",
            #     "id": 39,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+77_D.pdf",
            #     "id": 40,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+78_D.pdf",
            #     "id": 41,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1660+79_D.pdf",
            #     "id": 42,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1661+12_D.pdf",
            #     "id": 43,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1661+14_D.pdf",
            #     "id": 44,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/dow-pump-portfolio-en-data.pdf",
            #     "id": 45,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/Maschinendatenblatt CW_TEST CHAT GPT.pdf",
            #     "id": 46,
            #     "name": "file1",
            # },
            # {
            #     "remote_id": "taimoor/taimoor/VRP_59654-Split Kichererbsen ZPS 200 ATP 200 Food.pdf",
            #     "id": 47,
            #     "name": "file1",
            # },
            # {"remote_id": "taimoor/taimoor/amarex_krt.pdf", "id": 48, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/cpk.pdf", "id": 49, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/etabloc.pdf", "id": 50, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/etaline.pdf", "id": 51, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/etanorm.pdf", "id": 52, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/etanorm_syt.pdf", "id": 53, "name": "file1"},
            # {
            #     "remote_id": "taimoor/taimoor/giw_lcc_series.pdf",
            #     "id": 54,
            #     "name": "file1",
            # },
            # {"remote_id": "taimoor/taimoor/hpkl.pdf", "id": 55, "name": "file1"},
            # {
            #     "remote_id": "flowserve/0003056334C_EN_AQ.pdf",
            #     "id": 56,
            #     "name": "file1",
            #     "format": "pdf",
            # },
            # {"remote_id": "taimoor/taimoor/megacpk.pdf", "id": 56, "name": "file1"},
            # {"remote_id": "taimoor/taimoor/meganorm.pdf", "id": 57, "name": "file1"},
            # {
            #     "remote_id": "taimoor/taimoor/movitec_series.pdf",
            #     "id": 58,
            #     "name": "movitec_series.pdf",
            #     "format": "pdf",
            #     "original_format":"pdf",
            #     "processing_step": 0,
            #     "tenant_id": 2,
            #     "application_id":3
            # },
            # {
            #     "remote_id": "taimoor/taimoor/movitec_series.pdf",
            #     "id": 58,
            #     "name": "movitec_series.pdf",
            #     "format": "pdf",
            #     "original_format":"pdf",
            #     "processing_step": 0
            # },
            # {
            #     "remote_id": "taimoor/taimoor/amarex_krt.pdf",
            #     "id": 62,
            #     "name": "amarex_krt.pdf",
            #     "format": "pdf",
            #     "original_format":"pdf",
            #     "processing_step": 0,
            #     "tenant_id": 2,
            #     "application_id":3
            # },
            # {
            #     "remote_id": "taimoor/taimoor/BA 1121+23_D.pdf",
            #     "id": 65,
            #     "name": "BA 1121+23_D.pdf",
            #     "format": "pdf",
            #     "original_format":"pdf",
            #     "processing_step": 0
            # },
            # {
            #     "remote_id": "taimoor/taimoor/omega.pdf", 
            #     "id": 59, 
            #     "name": "omega.pdf",
            #     "format": "pdf",
            #     "original_format":"pdf",
            #     "processing_step": 0,
            #     "tenant_id": 2,
            #     "application_id":3
            # },
            {
                "remote_id": "taimoor/taimoor/Etanorm (EU, ME, NA) 1.PDF",
                "id": 75,
                "name": "Etanorm (EU, ME, NA) 1.PDF",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
            },
            {
                "remote_id": "taimoor/taimoor/Etanorm (EU, ME, NA) 1.PDF",
                "id": 55,
                "name": "Etanorm (EU, ME, NA) 1.PDF",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 3,
                "application_id":4
            },
            # {
            #     "remote_id": "taimoor/taimoor/C-001_Triconex ESD System (1) (1).ppt",
            #     "id": 60,
            #     "name": "file1",
            #     "format": "ppt",
            # },
        ]
        # self.que = Schema.encrypt("installation steps for seal-less pumps")
        self.que = Schema.encrypt(
            "Which pump is better, Amarex or Etaline"
        )
        self.que_de = Schema.encrypt("Installationsschritte für dichtungslose Pumpen")
        self.file_query = [
            {"remote_id": "temp1.pdf"},
            {"remote_id": "temp2.pdf"},
        ]

    def test_sync_time_profiler(self):
    # profiler = Profiler(interval=0.1)
    # with profiler:
        chat_id = 41
        params = [
            dict(name="Temperature", value="0.0", data_type="float"),
            dict(name="Max length (tokens)", value="3000", data_type="int"),
        ]
        # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
        # processor_obj.add_document(self.files)
        additional_args = {
            "chat_id": chat_id,
            "lang": "en",
            "params": params,
            "chat_subject": "New Chat",
            "model_name":'gpt-4o',
            "extras": {
                "namespace_id": 4,
                "app_id": 3,
                "tenant_id": 2,
                "username": "adminqa",
                "lang": "en",
                "lang_basis": "i",
                "chat_type": "E",
                "is_user_exempted_from_tokens_count": False,
                "web_search_enabled": False,
                "user_profile":{
                    "name":"Osama Khan",
                    "designation": "Chief Data Scientist",
                    "company_name":"Microsoft",
                    "country":"Argentina",
                    "response_customization":"Respond in Poetic language"
                }
            }
        }
        print("here right now")
        ency_answer, chat_subject = obj.ask_question(
            self.que, 
            # files=[{"id":70},{"id":63}], 
            files=[
                {"id":70,"summary":"This document is an installation and operating manual for the Amarex KRT submersible motor pump manufactured by KSB. It provides detailed instructions and guidelines for the installation, commissioning, servicing, and maintenance of the pump. The manual also includes information on safety regulations, transportation, electrical systems, and troubleshooting. The document contains a glossary of terms and an index for easy reference. Named entities identified in the text include KSB (manufacturer), Amarex KRT (product), and various sizes and types of pumps."},
                {"id":63,"summary":"Die vorliegende Dokumentation beschreibt die Wassernormpumpe Etanorm und ihre verschiedenen Ausführungen. Es werden Hauptanwendungen, Fördermedien, Betriebsdaten und der konstruktive Aufbau der Pumpe erläutert. Das Dokument enthält auch Informationen zu den Lieferumfang, den Werkstoffen und den Produktvorteilen der Pumpe. Es werden auch Inhaltsverzeichnisse und Zeichnungen bereitgestellt. Der Hersteller der Pumpe ist KSB SE & Co. KGaA und die Pumpe wird in Europa, dem Mittleren Osten und Nordafrika verwendet."}
                ],
            chat_type='E',
            chat_history={}, 
            **additional_args
        )
        obj.executor.shutdown(wait=True)
        print(1)
    #     # # dummy = asyncio.gather(obj.async_task_list)
    #     # answer = Schema.decrypt(ency_answer)
    #     # print("Should be printed early chat subject", chat_subject)
    #     # print("Should be printed early answer", answer)
    #     # future = obj.loop.run_until_complete(
    #     #     asyncio.gather(*asyncio.all_tasks(obj.loop))
    #     # )
    #     # print(obj.loop.is_closed())
    #     # print(future)
    # # profiler.print()
    
    # def test_sync_time_profiler_adv_search(self):
    # # profiler = Profiler(interval=0.1)
    # # with profiler:
    #     chat_id = 41
    #     params = [
    #         dict(name="Temperature", value="0.0", data_type="float"),
    #         dict(name="Max length (tokens)", value="3000", data_type="int"),
    #     ]
    #     # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
    #     # processor_obj.add_document(self.files)
    #     additional_args = {
    #         "chat_id": chat_id,
    #         # "lang": "en",
    #         "params": params,
    #         "chat_subject": "New Chat",
    #         "model_name":'gpt-3.5-turbo-16k',
    #         "extras": {
    #             "namespace_id": 4,
    #             "app_id": 3,
    #             "tenant_id": 2,
    #             "username": "adminqa",
    #             "lang": "en",
    #             "lang_basis": "i",
    #             "chat_type": "C",
    #             "is_user_exempted_from_tokens_count": False,
    #             "web_search_enabled": False,
    #             "user_profile":{
    #                 "name":"Osama Khan",
    #                 "designation": "Chief Data Scientist",
    #                 "company_name":"Microsoft",
    #                 "country":"Argentina"
    #             }
    #         }
    #     }
    #     print("here right now")
    #     # (self, user_input, files=None, lang=None, **kwargs)
    #     ency_answer = obj.advance_search(self.que, files=[{"id":59},{"id":61},{"id":63},{"id":64},{"id":65}], **additional_args)
    #     print(ency_answer)
        
    #     # # dummy = asyncio.gather(obj.async_task_list)
    #     # answer = Schema.decrypt(ency_answer)
    #     # print("Should be printed early chat subject", chat_subject)
    #     # print("Should be printed early answer", answer)
    #     # future = obj.loop.run_until_complete(
    #     #     asyncio.gather(*asyncio.all_tasks(obj.loop))
    #     # )
    #     # print(obj.loop.is_closed())
    #     # print(future)
    
    # def test_sync_memory_profiler(self):
    #     biggest_allocations = 20
    #     temporary_allocation_threshold = 0
    #     ALL_ALLOCATIONS = 0
    #     AGGREGATE_ALLOCATIONS = 1
    #     allocation_type = AGGREGATE_ALLOCATIONS
    #     arg_dict = dict(
    #         file_name="memory_prof.bin",
    #         trace_python_allocators=True,
    #         native_traces=False,
    #         file_format=allocation_type,  # ALL_ALLOCATIONS
    #     )

    #     Path(arg_dict["file_name"]).unlink(missing_ok=True)
    #     with Tracker(**arg_dict):
    #         chat_id = 42
    #         obj.add_document(self.files)

    #         params = dict(name="Temperature", value="0.5", data_type="float")
    #         additional_args = {
    #             "chat_id": chat_id,
    #             "lang": "de",
    #             "params": [params],
    #             "chat_subject": "New Chat",
    #         }
    #         ency_answer, chat_subject = obj.ask_question(
    #             self.que_de, files=None, chat_history={}, **additional_args
    #         )
    #         answer = Schema.decrypt(ency_answer)

    #         print(chat_subject)
    #         print(answer)
    #         obj.delete_document(Files)
    #     reader = FileReader("memory_prof.bin", report_progress=True)
    #     if temporary_allocation_threshold >= 0 and allocation_type == ALL_ALLOCATIONS:
    #         snapshot = iter(
    #             reader.get_temporary_allocation_records(
    #                 threshold=temporary_allocation_threshold,
    #                 merge_threads=False,
    #             )
    #         )
    #     else:
    #         snapshot = iter(
    #             reader.get_high_watermark_allocation_records(merge_threads=False)
    #         )
    #     reporter = TreeReporter.from_snapshot(
    #         snapshot,
    #         biggest_allocs=biggest_allocations,
    #         native_traces=reader.metadata.has_native_traces,
    #     )
    #     print()
    #     header = "Allocation metadata"
    #     rprint(f"{header}\n{'-'*len(header)}")
    #     rprint(f"Command line arguments: '{reader.metadata.command_line}'")
    #     rprint(f"Peak memory size: {size_fmt(reader.metadata.peak_memory)}")
    #     rprint(f"Number of allocations: {reader.metadata.total_allocations}")
    #     print()
    #     header = f"Biggest {biggest_allocations} allocations:"
    #     rprint(header)
    #     rprint("-" * len(header))
    #     reporter.render()

    # def test_sync_web_time_profiler(self):
    #     profiler = Profiler(interval=0.1)
    #     with profiler:
    #         chat_id = 51
    #         params = [
    #             dict(name="Temperature", value="0.0", data_type="float"),
    #             dict(name="Max length (tokens)", value="3000", data_type="int"),
    #         ]
    #         # obj.add_document(self.files)

    #         additional_args = {
    #             "chat_id": chat_id,
    #             "lang": "en",
    #             "params": params,
    #             "chat_subject": "New Chat",
    #             "model_name":'gpt-3.5-turbo-16k',
    #             "app_id": 3,
    #             "tenant_id": 2,
    #             "username": "adminqa",
    #             "lang": "en",
    #             "lang_basis": "i",
    #             "chat_type": "C",
    #             "is_user_exempted_from_tokens_count": False,
    #             "web_search_enabled": False,
    #             "lang": 'en',
    #             "subject":'a specific field',
    #             "extras": {
    #                         "namespace_id": 4,
    #                         "app_id": 3,
    #                         "tenant_id": 2,
    #                         "username": "adminqa",
    #                         "lang": "en",
    #                         "lang_basis": "i",
    #                         "chat_type": "C",
    #                         "is_user_exempted_from_tokens_count": False,
    #                         "web_search_enabled": True,
    #                     }
    #         }
    #         print("here right now")
    #         ency_answer, chat_subject = obj.ask_question(
    #             self.que, files=[{"id":65},{"id":58}], chat_history={}, **additional_args
    #         )
    #         # dummy = asyncio.gather(obj.async_task_list)
    #         answer = Schema.decrypt(ency_answer)
    #         print("Should be printed early chat subject", chat_subject)
    #         print("Should be printed early answer", answer)
    #         # future = obj.loop.run_until_complete(
    #         #     asyncio.gather(*asyncio.all_tasks(obj.loop))
    #         # )
    #         # print(obj.loop.is_closed())
    #         # print(future)
    #         obj.executor.shutdown(wait=True)
    #     profiler.print()
    #     print("alpha go")

    # def test_sync_memory_profiler(self):
    #     biggest_allocations = 20
    #     temporary_allocation_threshold = 0
    #     ALL_ALLOCATIONS = 0
    #     AGGREGATE_ALLOCATIONS = 1
    #     allocation_type = AGGREGATE_ALLOCATIONS
    #     arg_dict = dict(
    #         file_name="memory_prof.bin",
    #         trace_python_allocators=True,
    #         native_traces=False,
    #         file_format=allocation_type,  # ALL_ALLOCATIONS
    #     )

    #     Path(arg_dict["file_name"]).unlink(missing_ok=True)
    #     with Tracker(**arg_dict):
    #         chat_id = 42
    #         obj.add_document(self.files)

    #         params = dict(name="Temperature", value="0.5", data_type="float")
    #         additional_args = {
    #             "chat_id": chat_id,
    #             "lang": "de",
    #             "params": [params],
    #             "chat_subject": "New Chat",
    #         }
    #         ency_answer, chat_subject = obj.ask_question(
    #             self.que_de, files=None, chat_history={}, **additional_args
    #         )
    #         answer = Schema.decrypt(ency_answer)

    #         print(chat_subject)
    #         print(answer)
    #         obj.delete_document(Files)
    #     reader = FileReader("memory_prof.bin", report_progress=True)
    #     if temporary_allocation_threshold >= 0 and allocation_type == ALL_ALLOCATIONS:
    #         snapshot = iter(
    #             reader.get_temporary_allocation_records(
    #                 threshold=temporary_allocation_threshold,
    #                 merge_threads=False,
    #             )
    #         )
    #     else:
    #         snapshot = iter(
    #             reader.get_high_watermark_allocation_records(merge_threads=False)
    #         )
    #     reporter = TreeReporter.from_snapshot(
    #         snapshot,
    #         biggest_allocs=biggest_allocations,
    #         native_traces=reader.metadata.has_native_traces,
    #     )
    #     print()
    #     header = "Allocation metadata"
    #     rprint(f"{header}\n{'-'*len(header)}")
    #     rprint(f"Command line arguments: '{reader.metadata.command_line}'")
    #     rprint(f"Peak memory size: {size_fmt(reader.metadata.peak_memory)}")
    #     rprint(f"Number of allocations: {reader.metadata.total_allocations}")
    #     print()
    #     header = f"Biggest {biggest_allocations} allocations:"
    #     rprint(header)
    #     rprint("-" * len(header))
    #     reporter.render()


# class TestAsyncDocumentProfiler(unittest.IsolatedAsyncioTestCase):
#     def setUp(self):
#         self.files = [
#             {"remote_id": "flowserve/71569212_EN_AQ.pdf", "id": 4, "name": "file1"},
#         ]
#         self.que = Schema.encrypt("installation steps for seal-less pumps")
#         self.que_de = Schema.encrypt("Installationsschritte für dichtungslose Pumpen")

#     async def test_async_time_profiler(self):
#         profiler = Profiler(interval=0.01)
#         with profiler:
#             chat_id = 4
#             # obj.add_document(self.files)
#             await obj.add_document(self.files)
#             params = [
#                 dict(name="Temperature", value="0.5", data_type="float"),
#                 dict(name="Max length (tokens)", value="1300", data_type="int"),
#             ]
#             additional_args = {
#                 "chat_id": chat_id,
#                 "lang": "en",
#                 "params": params,
#                 "chat_subject": "New Chat",
#                 # "el": asyncio.get_event_loop(),
#             }
#             # tasks = [
#     obj.aask_question(
#         self.que, files=None, chat_history={}, **additional_args
#     )
#     for _ in range(10)
# ]
# results = await asyncio.gather(*tasks, return_exceptions=True)
# print(len(results), results)
# ency_answer = await obj.aask_question(
#     self.que, files=None, chat_history={}, **additional_args
#     # )
#     ency_answer, chat_title = obj.ask_question(
#         self.que, files=None, chat_history={}, **additional_args
#     )
#     answer = Schema.decrypt(ency_answer)
#     print(type(answer), answer)
#     print(type(chat_title), chat_title)
#     # obj.delete_document(Files)
# dummy_vars = await asyncio.gather(*obj.async_task_list)
# print(dummy_vars)
# profiler.print()


#     async def test_async_memory_profiler(self):
#         biggest_allocations = 20
#         temporary_allocation_threshold = 0
#         ALL_ALLOCATIONS = 0
#         AGGREGATE_ALLOCATIONS = 1
#         allocation_type = AGGREGATE_ALLOCATIONS
#         arg_dict = dict(
#             file_name="memory_prof.bin",
#             trace_python_allocators=True,
#             native_traces=False,
#             file_format=allocation_type,  # ALL_ALLOCATIONS
#         )

#         Path(arg_dict["file_name"]).unlink(missing_ok=True)
#         with Tracker(**arg_dict):
#             chat_id = 42
#             await obj.aadd_document(Files)
#             params = dict(name="Temperature", value="0.5", data_type="float")
#             additional_args = {
#                 "chat_id": chat_id,
#                 "lang": "de",
#                 "params": [params],
#                 "chat_subject": "New Chat",
#             }
#             ency_answer, chat_subject = obj.ask_question(
#                 self.que_de, files=None, chat_history={}, **additional_args
#             )
#             answer = fileMeta.decrypt(ency_answer)

#             print(chat_subject)
#             print(answer)
#             obj.delete_document(Files)
#         reader = FileReader("memory_prof.bin", report_progress=True)
#         if temporary_allocation_threshold >= 0 and allocation_type == ALL_ALLOCATIONS:
#             snapshot = iter(
#                 reader.get_temporary_allocation_records(
#                     threshold=temporary_allocation_threshold,
#                     merge_threads=False,
#                 )
#             )
#         else:
#             snapshot = iter(
#                 reader.get_high_watermark_allocation_records(merge_threads=False)
#             )
#         reporter = TreeReporter.from_snapshot(
#             snapshot,
#             biggest_allocs=biggest_allocations,
#             native_traces=reader.metadata.has_native_traces,
#         )
#         print()
#         header = "Allocation metadata"
#         rprint(f"{header}\n{'-'*len(header)}")
#         rprint(f"Command line arguments: '{reader.metadata.command_line}'")
#         rprint(f"Peak memory size: {size_fmt(reader.metadata.peak_memory)}")
#         rprint(f"Number of allocations: {reader.metadata.total_allocations}")
#         print()
#         header = f"Biggest {biggest_allocations} allocations:"
#         rprint(header)
#         rprint("-" * len(header))
#         reporter.render()

#     def tearDown(self) -> None:
#         obj.delete_document(self.files)

# output = profiler.output_html()
# fp = open("./pyinstrument_output.html", "w")
# fp.write(output)
# fp.close()

#     def test_sync_memory_profiler(self):
#         biggest_allocations = 20
#         temporary_allocation_threshold = 0
#         ALL_ALLOCATIONS = 0
#         AGGREGATE_ALLOCATIONS = 1
#         allocation_type = AGGREGATE_ALLOCATIONS
#         arg_dict = dict(
#             file_name="memory_prof.bin",
#             trace_python_allocators=True,
#             native_traces=False,
#             file_format=allocation_type,  # ALL_ALLOCATIONS
#         )

#         Path(arg_dict["file_name"]).unlink(missing_ok=True)
#         with Tracker(**arg_dict):
#             chat_id = 42
#             obj.add_document(Files)
#             params = dict(name="Temperature", value="0.5", data_type="float")
#             additional_args = {
#                 "chat_id": chat_id,
#                 "lang": "de",
#                 "params": [params],
#                 "chat_subject": "New Chat",
#             }
#             ency_answer, chat_subject = obj.ask_question(
#                 self.que_de, files=None, chat_history={}, **additional_args
#             )
#             answer = fileMeta.decrypt(ency_answer)
#             #
#             print(chat_subject)
#             print(answer)
#             obj.delete_document(Files)
#         reader = FileReader("memory_prof.bin", report_progress=True)
#         if temporary_allocation_threshold >= 0 and allocation_type == ALL_ALLOCATIONS:
#             snapshot = iter(
#                 reader.get_temporary_allocation_records(
#                     threshold=temporary_allocation_threshold,
#                     merge_threads=False,
#                 )
#             )
#         else:
#             snapshot = iter(
#                 reader.get_high_watermark_allocation_records(merge_threads=False)
#             )
#         reporter = TreeReporter.from_snapshot(
#             snapshot,
#             biggest_allocs=biggest_allocations,
#             native_traces=reader.metadata.has_native_traces,
#         )
#         print()
#         header = "Allocation metadata"
#         rprint(f"{header}\n{'-'*len(header)}")
#         rprint(f"Command line arguments: '{reader.metadata.command_line}'")
#         rprint(f"Peak memory size: {size_fmt(reader.metadata.peak_memory)}")
#         rprint(f"Number of allocations: {reader.metadata.total_allocations}")
#         print()
#         header = f"Biggest {biggest_allocations} allocations:"
#         rprint(header)
#         rprint("-" * len(header))
#         reporter.render()

#     def tearDown(self) -> None:
#         obj.delete_document(self.files)

# chat_id = 4
# obj.add_document(Files)
# params = dict(name='Temperature', value='0.5', data_type='float')
# additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': [params]}
# ency_answer = obj.ask_question(self.que, files=None, chat_history={},
#                                **additional_args)
# answer = fileMeta.decrypt(ency_answer)
# #
# print(answer)
# # self.assertEqual(type(answer), str)
# # obj.delete_document(self.files)

#     def test_extensive_test(self):
#         questions = [
#             'hello',
#             'what is pump',
#             'tell me about seal-less pumps',
#             'what is the specification for the seal-less pump'
#         ]
#         file = [
#             {'remote_id': 'flowserve/71569212_EN_AQ.pdf'},
#             # {'remote_id': 'flowserve/71569195_DE_A4.pdf'},
#             # {'remote_id': 'flowserve/PS-20-1_DE_A4.pdf'},
#         ]
#         obj.add_document(file)
#         files = [file, None, Files, None]
#         chat_id = [None, 1, 2, 3]
#         chat_history = {}
#         params = [dict(name='Temperature', value='0.0', data_type='float')]
#         for que, fil, ch_id in zip(questions, files, chat_id):
#             encrypt_que = fileMeta.encrypt(que)
#             additional_args = {'chat_id': ch_id, 'lang': 'en', 'params': params}
#             ency_answer = obj.ask_question(encrypt_que, files=fil, chat_history=chat_history,
#                                            **additional_args)
#             answer = fileMeta.decrypt(ency_answer)
#             chat_history[time.time()] = store_message(
#                 fileMeta.encrypt(que),
#                 fileMeta.encrypt(str(answer))
#             )
#             print(que)
#             print(answer)

#     # def test_language_check(self):
#     #     obj.ask_question(file)

# class TestStringMethods(unittest.TestCase):
#     def setUp(self):
#         self.user_input = fileMeta.encrypt('installation steps for seal-less pumps')
#         self.params = dict(name='Temperature', value='0.0', data_type='float')
#         self.files = Files
#         self.file_query = [
#             {'remote_id': 'flowserve/71569212_EN_AQ.pdf'},
#         ]
#         self.empty_file = [
#             {'remote_id': 'flowserve/dumy.pdf'},
#         ]
#         obj.add_document(Files)

#     def tearDown(self) -> None:
#         obj.delete_document(self.files)

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
    # loop.close()
