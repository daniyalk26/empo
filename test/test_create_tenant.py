import unittest
import os, sys


# import aiounittest
# from pyinstrument import Profiler
from dotenv import load_dotenv
# from memray import Tracker
# from memray._memray import size_fmt
# from rich import print as rprint
# from memray.reporters.tree import TreeReporter
# from memray import FileReader
import logging

load_dotenv()
sys.path.append("./")
from doclm.operations import get_record_obj

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

obj=get_record_obj()

class TestCreateTenant(unittest.TestCase):
    def test_create_tenant(self):
        obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')


    
if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
