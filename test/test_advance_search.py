import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv
load_dotenv()
sys.path.append("./")

from doclm.doclm import get_interactive
from doclm.schema import Schema
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


class TestAdvanceSearch(unittest.TestCase):
    def setUp(self):
        self.obj = get_interactive()

    def test_advance_search(self):
        user_question = "amarex krt characteristics noise " if os.getenv('ENCRYPT')=='False' else Schema.encrypt( "amarex krt characteristics noise ")
        adv_search_results = self.obj.advance_search(user_question, [{'id':i} for i in range(1,100)], **{"extras": {
                "namespace_id": 4,
                "app_id": 3,
                "rerank_bool": True,
                "tenant_id": 2,}})
        print(adv_search_results)
        self.obj.executor.shutdown(wait=True)

    def tearDown(self): 
        del self.obj
        return super().tearDown()
    
if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()