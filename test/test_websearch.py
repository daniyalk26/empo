import json
import time
import unittest
import asyncio

import os, sys
from dotenv import load_dotenv
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

class TestSyncWeb(unittest.TestCase):
    def setUp(self) -> None:
        self.user_question = Schema.encrypt(
            "What is the current temperature in lahore?"
        )
    def test_sync_web(self):
        chat_id = 43
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
                "rerank_bool": True,
                "tenant_id": 2,
                "username": "adminqa",
                "lang": "en",
                "lang_basis": "i",
                "chat_type": "C",
                "is_user_exempted_from_tokens_count": False,
                "web_search_enabled": True,
                "user_profile":{
                    "name":"Osama Khan",
                    "designation": "Chief Data Scientist",
                    "company_name":"Microsoft",
                    "country":"Argentina",
                    "response_customization": "Respond in a poetic manner"
                }
            }
        }

        ency_answer, chat_subject = obj.ask_question(
            self.user_question,
            files=[], 
            chat_type='C', 
            chat_history={}, 
            **additional_args
        )
        obj.executor.shutdown(wait=True)

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()