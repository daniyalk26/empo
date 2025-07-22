import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv

if True:
    load_dotenv()
sys.path.append("./")


from doclm.doclm import get_interactive
from doclm.util import parse_param
from doclm.schema import Schema

obj = get_interactive()


def list_files(dir, format):
    file = []
    count = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_name = os.path.join(root, name)
            if format == 'pdf':
                # count+=1
                file.append(
                    {'remote_id': file_name, 'id': 1, 'name': file_name, 'format': 'pdf', 'original_format': 'pdf',
                     'lang': 'en'})
            elif format == 'ppt':
                file.append({'remote_id': file_name, 'id': 'test', 'name': file_name, 'format': 'ppt'})
    return file


class TestStringMethods(unittest.TestCase):
    def test_extensive_test(self):
        # os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres2"
        format = 'pdf'
        folder_path = '../test_doc/data_add/'
        # lang = 'English'
        # files = list_files(folder_path, format)
        # print('files', files)
        files = [{"id": 1}, {"id": 54}, {"id": 56}, {"id": 58}, {"id": 60}
                 ]
        question1 = Schema.encrypt(json.dumps({'question': 'Installation Instructions of Amarex KRT pump'}))
        question2 = Schema.encrypt(json.dumps({'question': 'Installation Instructions of PHL pump'}))
        question = [question1]  #, question2]
        chat_history = {}
        params = [dict(name="Temperature", value="0.0", data_type="float"),
                  dict(name="Max length (tokens)", value="3000", data_type="int")]
        additional_args = {'chat_id': -100, 'lang': 'en', 'params': params,
                           "chat_subject": "New Chat", "model_name": 'gpt-4o',
                           "extras": {"web_search_enabled": False,
                                      "tenant_id": 2},
                           }

        result = obj.ask_agent_question(question, files, **additional_args)
        print(result)
    # ask_agent_question(question, files)
    # rag_object = RagNode()
    # result = rag_object.run_rag_node(files=files, question=question, lang=lang,)
    # x = json.loads(result)
    # print(x['response'])


if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
    # loop.close()
