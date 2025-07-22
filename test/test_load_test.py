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

class TestLoadTestAskQuestion(unittest.TestCase):
    def setUp(self) -> None:
        # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
        files_to_upload=[
            {
                "remote_id": "concurrencyTesting/Amarex KRT; Motor F; DN40-300, 50 Hz, CE-EN.PDF",
                "id": 71,
                "name": "Amarex KRT; Motor F; DN40-300, 50 Hz, CE-EN.PDF",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
            },
            {
                "remote_id": "concurrencyTesting/CreditCard.pdf",
                "id": 72,
                "name": "CreditCard.pdf",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
            },
            {
                "remote_id": "concurrencyTesting/JavaProgramming.pdf",
                "id": 73,
                "name": "JavaProgramming.pdf",
                "format": "pdf",
                "original_format":"pdf",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
            },]
        for file in files_to_upload:
            processor_obj.add_document([file])
        
    
    def test_loadtest_enterprise_knowledge(self):
        chat_id = 41
        params = [
            dict(name="Temperature", value="0.0", data_type="float"),
            dict(name="Max length (tokens)", value="3000", data_type="int"),
        ]
        additional_args = {
            "chat_id": chat_id,
            "lang": "en",
            "params": params,
            "chat_subject": "New Chat",
            "model_name":'gpt-4o',
            "extras": {
                "namespace_id": 4,
                "app_id": 3,
                "rerank_bool": True,
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
        # ency_answer, chat_subject = obj.ask_question(
        #     self.que, 
        #     files=[
        #         {"id":70,"summary":"This document is an installation and operating manual for the Amarex KRT submersible motor pump manufactured by KSB. It provides detailed instructions and guidelines for the installation, commissioning, servicing, and maintenance of the pump. The manual also includes information on safety regulations, transportation, electrical systems, and troubleshooting. The document contains a glossary of terms and an index for easy reference. Named entities identified in the text include KSB (manufacturer), Amarex KRT (product), and various sizes and types of pumps."},
        #         {"id":63,"summary":"Die vorliegende Dokumentation beschreibt die Wassernormpumpe Etanorm und ihre verschiedenen Ausführungen. Es werden Hauptanwendungen, Fördermedien, Betriebsdaten und der konstruktive Aufbau der Pumpe erläutert. Das Dokument enthält auch Informationen zu den Lieferumfang, den Werkstoffen und den Produktvorteilen der Pumpe. Es werden auch Inhaltsverzeichnisse und Zeichnungen bereitgestellt. Der Hersteller der Pumpe ist KSB SE & Co. KGaA und die Pumpe wird in Europa, dem Mittleren Osten und Nordafrika verwendet."}
        #         ],
        #     chat_type='E',
        #     chat_history={}, 
        #     **additional_args
        # )
        obj.executor.shutdown(wait=True)
    

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
    # loop.close()
