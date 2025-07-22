
import aiounittest
import json
import unittest
from dotenv import load_dotenv

if True:
    load_dotenv()

from doclm.processing import DocumentProcessor
from doclm.doclm import get_interactive
from doclm.schema import Schema

doc_obj = DocumentProcessor()
obj = get_interactive()

class TestAsyncApis(aiounittest.AsyncTestCase):
    def setUp(self):
        self.files = [
            {"remote_id": "flowserve/71569212_EN_AQ.pdf", 'id': 4, 'name': 'file1'},
            {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]

        self.file_query = [
            {'remote_id': 'flowserve/71569212_EN_AQ.pdf'},
        ]

        self.empty_file = [
            {'remote_id': 'flowserve/dumy.pdf'},
        ]
        # self.que = input("Ask me:")
        self.que = Schema.encrypt("installation steps for seal-less pumps")
        # self.que = Schema.encrypt(self.que)

        self.que_de = Schema.encrypt("Installationsschritte für dichtungslose Pumpen")
        self.params = dict(name="Temperature", value="0.0", data_type="float")
        self.additional_args = {'chat_id': -100, 'lang': 'en', 'params': [self.params],
                                "chat_subject": "New Chat",
                                }
        # obj.add_document(self.files)

    async def test_general_rout_selection(self):
        question = 'how are you'
        chat_id = 0
        chat_history = {}
        # obj.add_document(self.files)

        encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': chat_id})

        ency_answer, chat_subject = await obj.aask_question(
            encrypt_que, files=None, chat_history=chat_history, **additional_args)
        answer = Schema.decrypt(ency_answer)
        self.assertIsInstance(answer, str)
        answer = json.loads(answer)
        self.assertIsNone(chat_subject)
        self.assertNotIn('sorry', answer['response'].lower())
        self.assertIsInstance(answer['files'], list)
        self.assertEqual(len(answer['files']), 0)


    async def test_pdf(self):
        files = [
            {"remote_id": '/home/administrator/Codes/documentlm/test_doc/00079593_EN_A4.pdf',
             'id': 11,
             'name': 'file1',
             'format': 'pdf',
             'original_format': 'pdf',
             'tenant_id':2
             },
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]

        await doc_obj.async_add_document(files)

    async def test_html_ask(self):
        question = 'what are theo peration, and maintenance of a PHL pump'

        encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': 'async query',
                                'lang': 'en',
                                'chat_subject': 'something ',
                                "extras": {"web_search_enabled": False,
                                           "tenant_id": 2},

                                })

        obj.ask_question(
            encrypt_que, files=[{'id': 7}], chat_history='', chat_type='E', model_name='gpt-4o',
            **additional_args)



if __name__ == "__main__":
    unittest.main()
