import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv
load_dotenv()
sys.path.append("./")
from doclm.operations import get_record_obj

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())



Files = [{
                "remote_id": "taimoor/taimoor/Expert Advisor Competitors Pricing.xlsx",
                "id": 65,
                "name": "Expert Advisor Competitors Pricing.xlsx",
                "format": "xlsx",
                "original_format":"xlsx",
                "processing_step": 0,
                "tenant_id": 2,
                "application_id":3
            }]

class TestDeleteFile(unittest.TestCase):
    def setUp(self):
        self.obj = get_record_obj()
        pass

    def test_delete_file(self):
        self.obj.delete_document(Files)
        pass

    def tearDown(self): 
        del self.obj
        return super().tearDown()
    
if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()