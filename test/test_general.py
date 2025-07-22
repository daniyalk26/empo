import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv

load_dotenv()

from doclm.doclm import get_interactive
from doclm.util import parse_param
from doclm.schema import Schema
from doclm.processing import DocumentProcessor

test_german = True # False
test_english = False # True
test_encription = False

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

# docker run --name postgres --network bridge -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector
# docker run  -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 --link postgres dpage/pgadmin4
# docker run  -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 --link hungry_mcclintock dpage/pgadmin4
# postgresql://postgres:postgres@10.1.17.209:5432/expert_advisor_ahmed
# postgresql://postgres:postgres@localhost:5432/postgres

# os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres"
# docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres  -v /home/administrator/Codes/postgres_data:/var/lib/postgresql/data  ankane/pgvector
# docker run --network bridge -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 dpage/pgadmin4
# docker run  -e PGADMIN_DEFAULT_EMAIL=abcd@efg.com -e PGADMIN_DEFAULT_PASSWORD=abc1234 -p 5000:80 --link hungry_mcclintock dpage/pgadmin4
# az acr login --name intechwesteuropecr
# docker run -e POSTGRESQL_PASSWORD=postgres -p 5432:5432 --name postgres -v pgdata:/bitnami/postgresql -d intechwesteuropecr.azurecr.io/postgresql:15-bitnami

def store_message(q, r):
    return {"q": {"id": 309, "message": q}, "r": {"id": 310, "message": r}}


obj = get_interactive()
doc_obj = DocumentProcessor()

Files = [
    {"remote_id": "flowserve/71569212_EN_AQ.pdf", "id": 1, 'name': 'file1'},
    {"remote_id": "flowserve/71569195_DE_A4.pdf", "id": 1, 'name': 'file3'},
    # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", "id": 2, 'name': 'file2'},
    # {"remote_id": "flowserve/71569195_DE_A4.pdf", "id": 3, 'name': 'file3'},
    # {"remote_id": "flowserve/PS-20-1_DE_A4.pdf", "id": 4, 'name': 'file4'},
]

class TestSyncApisSpecific(unittest.TestCase):
    def setUp(self):
        self.que_de = Schema.encrypt(
            "Initiale Ausrichtungsprozedur während der Installation einer dichtungslosen Pumpe")
        self.params = [dict(name="Temperature", value="0.0", data_type="float"),
                       dict(name="Max length (tokens)", value="3000", data_type="int")]
        self.additional_args = {'chat_id': -100, 'lang': 'en', 'params': self.params,
                                "chat_subject": "New Chat",
                                }


    def test_route_general_chat_attach_file_irrelevant_question(self):
        question = 'tell me key points about flowserve pump'
        question = 'give me general statics for Afghanistan'
        chat_id = 41
        # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
        # processor_obj.add_document(self.files)

        additional_args = self.additional_args.copy()

        additional_args.update({'chat_id': chat_id,
                                'lang': 'en',
                                'chat_subject': 'something',
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
                                    "web_search_enabled": False,
                                    "user_profile": {
                                        "name": "Osama Khan",
                                        "designation": "Chief Data Scientist",
                                        "company_name": "Microsoft",
                                        "country": "Argentina"
                                    },
                                }
                                })


        encrypt_que = Schema.encrypt(question)
        f_files = [
            {'id': 5, 'summary':'''The document details observations and recommendations for the "Go-Live Permit to Work System (WPMS)" at Adhi Field. It includes various high-priority issues that need addressing, such as mandatory training, system testing, and SOP development. Named entities include Adhi Field, PPL (Pakistan Petroleum Limited), Intech, and individuals like Mr. Abdul Samad SE (E&I).'''},
            {'id': 1,
             'summary': '''The document appears to be a detailed log of GPS-synchronized data points collected along a specific route from KM 0.000 to KM 5.000. It includes timestamps, latitude and longitude coordinates, signal quality, number of satellites, and other related metrics. Named entities identified include "KM 0 000 IJ SV1 STATION" and "FENCE," as well as geographical coordinates and dates.'''},
            {'id': 2,
             'summary': ''' The document appears to be a detailed log of GPS data points collected on June 20, 2022. It includes information such as chainage, on/off times, UTC time, latitude, longitude, signal quality, number of satellites, PDOP, altitude, and other related metrics. Named entities include GPS equipment, the date "2022-06-20," and geographical coordinates (latitude and longitude).'''},
            {'id': 3,
             'summary': '''The document provides statistical data for countries including Afghanistan, Albania, Algeria, American Samoa, and Andorra from 2000 to 2010. It covers categories such as transit (railways, passenger cars), business (mobile phone subscribers, internet users), health (mortality, health expenditure), population (total, urban, birth rate, life expectancy), and finance (GDP, GDP per capita).'''},
        ]

        ency_answer, chat_subject = obj.ask_question(
            encrypt_que, files=f_files, chat_history='', chat_type='C', model_name='gpt-4o',
            **additional_args)
        # ency_answer, chat_subject = obj.ask_question(
        #     self.que, files=[{"id":8}], chat_type='C',chat_history={}, **additional_args
        # )
        print(1)
        obj.executor.shutdown(wait=True)


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.user_input = Schema.encrypt('installation steps for seal-less pumps')
        self.params = dict(name='Temperature', value='0.0', data_type='float')
        self.files = Files
        self.file_query = [
            {'remote_id': 'flowserve/71569212_EN_AQ.pdf'},
        ]
        self.empty_file = [
            {'remote_id': 'flowserve/dumy.pdf'},
        ]

    #     obj.add_document(Files)
    #
    # def tearDown(self) -> None:
    #     obj.delete_document(self.files)

    def test_router_chain(self):
        pass

    def test_param_parsert(self):
        params = dict(name='Temperature', value='0.5', data_type='float')
        additional_args = {'chat_id': 1, 'lang': 'en', 'params': [params],
                           "chat_subject": "New Chat"}
        chat_id, param, lang, chat_subject = parse_param(additional_args)
        self.assertEqual(chat_subject, 'New Chat')
        self.assertEqual(chat_id, 1)
        self.assertEqual(lang, additional_args['lang'])
        self.assertEqual(type(param), dict)

    def test_wrong_dtype_params(self):
        chat_id = 3
        params = dict(name='Temperature', value='0.0', data_type='float')
        additional_args = {'chat_id': chat_id, 'lang': 'ss', 'params': [params],
                           "chat_subject": None}
        self.assertRaises(AssertionError, parse_param, additional_args)


#     params = [dict(name="Temperature", value="0.0", data_type="float")]
#     additional_args = {"chat_id": chat_id, "lang": "en", "params": params}
#     ency_answer = obj.ask_question(
#         self.user_input, files=file, chat_history={}, **additional_args
#     )
#     answer = fileMeta.decrypt(ency_answer)
#     print(answer)
#     self.assertIn("sorry", answer.lower())

#     obj.delete_document(file)

class TestSyncApisSpecific(unittest.TestCase):
    def setUp(self):
        self.que_de = Schema.encrypt(
            "Initiale Ausrichtungsprozedur während der Installation einer dichtungslosen Pumpe")
        self.params = [dict(name="Temperature", value="0.0", data_type="float"),
                       dict(name="Max length (tokens)", value="3000", data_type="int")]
        self.additional_args = {'chat_id': -100, 'lang': 'en', 'params': self.params,
                                "chat_subject": "New Chat",
                                }


    def test_route_general_chat_attach_file_irrelevant_question(self):
        question = 'tell me key points about flowserve pump'
        chat_id = 41
        # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
        # processor_obj.add_document(self.files)

        additional_args = self.additional_args.copy()

        additional_args.update({'chat_id': chat_id,
                                'lang': 'en',
                                'chat_subject': 'something',
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
                                    "web_search_enabled": False,
                                    "user_profile": {
                                        "name": "Osama Khan",
                                        "designation": "Chief Data Scientist",
                                        "company_name": "Microsoft",
                                        "country": "Argentina"
                                    },
                                }
                                })


        encrypt_que = Schema.encrypt(question)
        f_files = [
            {'id': 5, 'summary':'''The document details observations and recommendations for the "Go-Live Permit to Work System (WPMS)" at Adhi Field. It includes various high-priority issues that need addressing, such as mandatory training, system testing, and SOP development. Named entities include Adhi Field, PPL (Pakistan Petroleum Limited), Intech, and individuals like Mr. Abdul Samad SE (E&I).'''},
            {'id': 1,
             'summary': '''The document appears to be a detailed log of GPS-synchronized data points collected along a specific route from KM 0.000 to KM 5.000. It includes timestamps, latitude and longitude coordinates, signal quality, number of satellites, and other related metrics. Named entities identified include "KM 0 000 IJ SV1 STATION" and "FENCE," as well as geographical coordinates and dates.'''},
            {'id': 2,
             'summary': ''' The document appears to be a detailed log of GPS data points collected on June 20, 2022. It includes information such as chainage, on/off times, UTC time, latitude, longitude, signal quality, number of satellites, PDOP, altitude, and other related metrics. Named entities include GPS equipment, the date "2022-06-20," and geographical coordinates (latitude and longitude).'''},
            {'id': 3,
             'summary': '''The document provides statistical data for countries including Afghanistan, Albania, Algeria, American Samoa, and Andorra from 2000 to 2010. It covers categories such as transit (railways, passenger cars), business (mobile phone subscribers, internet users), health (mortality, health expenditure), population (total, urban, birth rate, life expectancy), and finance (GDP, GDP per capita).'''},
        ]

        ency_answer, chat_subject = obj.ask_question(
            encrypt_que, files=f_files, chat_history='', chat_type='C', model_name='gpt-4o',
            **additional_args)
        # ency_answer, chat_subject = obj.ask_question(
        #     self.que, files=[{"id":8}], chat_type='C',chat_history={}, **additional_args
        # )
        print(1)
        obj.executor.shutdown(wait=True)


class TestSyncApis(unittest.TestCase):
    def setUp(self):
        self.files = [
            {"remote_id": "flowserve/71569212_EN_AQ.pdf", 'id': 4, 'name': 'file1'},
            # {"remote_id": "flowserve/71569195_DE_A4.pdf", 'id': 4, 'name': 'file1'},
            # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", 'id': 2, 'name': 'file2'},
        ]

        self.file_query = [
            {'remote_id': 'flowserve/71569212_EN_AQ.pdf'},
        ]

        self.empty_file = [
            {'remote_id': 'flowserve/dumy.pdf'},
        ]
        # self.que = input("Ask me:")

        self.que = Schema.encrypt(
            # "Initial alignment procedure during installation of seal-less pump")
            "Initial alignment procedure seal-less pump")
        # self.que = Schema.encrypt(self.que)

        self.que_de = Schema.encrypt(
            "Initiale Ausrichtungsprozedur während der Installation einer dichtungslosen Pumpe")
        self.params = [dict(name="Temperature", value="0.0", data_type="float"),
                       dict(name="Max length (tokens)", value="3000", data_type="int")]
        self.additional_args = {'chat_id': -100, 'lang': 'en', 'params': self.params,
                                "chat_subject": "New Chat",
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
                                    "web_search_enabled": False,
                                    "user_profile": {
                                        "name": "Osama Khan",
                                        "designation": "Chief Data Scientist",
                                        "company_name": "Microsoft",
                                        "country": "Argentina"
                                    },
                                }
                                }
        # obj.add_document(self.files)

    def tearDown(self) -> None:
        pass

    def test_trasnlation(self):
        target_language = 'de'
        input_text = 'how are you'
        answser = obj.translate(input_text, target_language)
        print(answser)
        # obj.delete_document(Files)


    def test_o1_model(self):
        question = 'write 10 words on each chunk'

        encrypt_que = Schema.encrypt(question)

        params = [dict(name="Temperature", value="1.0", data_type="float"),
                   dict(name="Max length (tokens)", value="2000", data_type="int")]
        additional_args = {'chat_id': -100, 'lang': 'en', 'params': params,
                           'chat_subject': 'something ',
                           "extras": {"web_search_enabled": False,
                                      "tenant_id": 2},
                           'ck':Schema.encrypt("asdsadsa"),
                                }
        obj.ask_question(
            encrypt_que, files=[{'id': 0},
                                {'id': 10},
                                {'id': 11},
                                {'id': 18},
                                {'id': 40},
                                {'id': 43},
                                {'id': 45},
                                ], chat_history='', chat_type='E', model_name='gpt-4o',  #
            **additional_args)
        obj.executor.shutdown(wait=True)


    def test_html_ask(self):
        question = 'what are the problems faced by brookner'

        encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': 'html',
                                'lang': 'en',
                                'chat_subject': 'something ',
                                "extras": {"web_search_enabled": False,
                                           "tenant_id": 2},

                                })

        obj.ask_question(
            encrypt_que, files=[{'id': 0}], chat_history='', chat_type='C', model_name='previewo1',#
            **additional_args)

    def test_custom(self):
        question = 'tell me key points about flowserve pump'
        question = 'tell me about the document'
        # question = 'tell me key points about load aware transformers'
        # file_path = '../test_doc/2.1.6 Kunstharze, Novalok S.30-31 (6).pdf'
        # file_path = '/home/testingmachine2-0/bitbucket/alpine-expertadvisor-be/external/apps/documentlm/test_doc/71569212_EN_AQ.pdf'
        # reader = load_reader("azure_blob")

        # obj.add_document([
        #     {"remote_id": "../test_doc/redpestttttttttttttttttttttedvgevgdcvg(1).de.en.pdf",
        #      "id": 1, 'name': 'file1'},
        # ])
        encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': 'custom',
                                'lang': 'en',
                                'chat_subject': 'something '})
        ency_answer, chat_subject = obj.ask_question(
            encrypt_que, files=[{'id': 3}], chat_history='', chat_type='C', model_name='gpt-4o',
            **additional_args)
        obj.executor.shutdown(wait=True)

    # gpt-3.5-turbo-16k
    def test_add_document(self):
        # files_subset = [
        #     # {"remote_id": "flowserve/0003056334C_EN_AQ.pdfs", "id": 2, 'name': 'file2'},
        #     # {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", "id": 2, 'name': 'file2'},
        #     {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", "id": 2, 'name': 'file2'},
        #     # {"remote_id": "flowserve/71569212_EN_AQ.pdf", "id": 1, 'name': 'file1'},
        # ]
        # # obj.add_document(files_subset)

        obj.add_document([
            {"remote_id": "flowserve/0003056334C_EN_AQ.pdf", "id": 1, 'name': 'file1'},
        ])

    def test_general_rout_selection(self):
        if test_english:
            question = 'how are you'
            chat_id = 0
            chat_history = {}

            encrypt_que = Schema.encrypt(question)
            additional_args = self.additional_args.copy()
            additional_args.update({'chat_id': chat_id,
                                    'lang': 'en',
                                    'chat_subject': 'New Chat'})
            ency_answer, chat_subject = obj.ask_question(
                encrypt_que, files=None, chat_history=chat_history, **additional_args)
            print(ency_answer)
            print(chat_subject)
            # answer = MetaExtractor.decrypt(ency_answer)
            # answer = json.loads(answer)
            # self.assertIsNone(chat_subject)
            # self.assertNotEqual(len(answer['response']), 0)
            # # self.assertNotIn('sorry', answer['response'].lower())
            # self.assertTrue(('apologize' not in answer['response'].lower()) and
            #                 ('sorry' not in answer['response'].lower()))
            # self.assertIsInstance(answer['files'], list)
            # self.assertEqual(len(answer['files']), 0)

        if test_german:
            question = 'Wie sind Sie'
            chat_id = 0
            chat_history = {}

            encrypt_que = Schema.encrypt(question)
            additional_args = self.additional_args.copy()
            additional_args.update({'chat_id': chat_id,
                                    'lang': 'de',
                                    'chat_subject': 'New Chat'})

            ency_answer, chat_subject = obj.ask_question(
                encrypt_que, files=None, chat_history=chat_history, **additional_args)
            answer = Schema.decrypt(ency_answer)
            answer = json.loads(answer)
            self.assertIsNone(chat_subject)  # being convesational chat chat_subject should be None
            self.assertNotEqual(len(answer['response']), 0)
            # self.assertNotIn('sorry', answer['response'].lower())
            self.assertTrue(('apologize' not in answer['response'].lower()) and
                            ('sorry' not in answer['response'].lower()))
            self.assertIsInstance(answer['files'], list)
            self.assertEqual(len(answer['files']), 0)

    def test_subject_rout_selection(self):
        # obj.add_document(self.files)

        if test_english:
            # question = 'what is a pump'  # given an empty db
            # encrypt_que = MetaExtractor.encrypt(question)
            # question = 'did romeo and juliet get married'  # given an empty db
            # question = 'did romeo and juliet get married'  # given an empty db
            chat_id = 1
            chat_history = {}

            additional_args = self.additional_args.copy()
            additional_args.update({'chat_id': chat_id})
            # additional_args.update({'chat_subject': 'Defining Pumps'})

            ency_answer, chat_subject = obj.ask_question(
                self.que, files=None, chat_history=chat_history, **additional_args)
            # answer = MetaExtractor.decrypt(ency_answer)
            # answer = json.loads(answer)
            # print(chat_subject)
            # print(answer)
            # if additional_args.get('chat_subject') or (
            #         additional_args.get('chat_subject') == 'New Chat'):
            #     self.assertIsNotNone(chat_subject)
            # else:
            #     self.assertIsNone(chat_subject)
            # self.assertIsInstance(answer['files'], list)
            #
            # if len(answer['files']) == 0:
            #     self.assertTrue(('apologize' in answer['response'].lower()) or
            #                     ('sorry' in answer['response'].lower()))
            #
            # else:
            #     self.assertTrue(('apologize' not in answer['response'].lower()) and
            #                     ('sorry' not in answer['response'].lower()))
            #     self.assertEqual(len(answer['files']), 1)

        if test_german:
            ## German language
            question_de = 'Was ist eine Pumpe?'  # given an empty db
            chat_history = {}

            encrypt_que = Schema.encrypt(question_de)
            additional_args = self.additional_args.copy()
            additional_args.update({'lang': 'de'})
            additional_args.update({'chat_subject': 'New Chat'})
            #
            status, details = obj.ask_question(
                encrypt_que, files=None, chat_history=chat_history, **additional_args)
            print(status)
            print(details)
            # self.assertIsNone(chat_subject)
            # self.assertTrue(('apologize' not in answer['response'].lower()) and
            #                 ('sorry' not in answer['response'].lower()))
            # self.assertIsInstance(answer['files'], list)
            # self.assertEqual(len(answer['files']), 0)

    def test_single_file_filter_chat(self):
        obj.delete_document(Files)
        obj.add_document(Files)
        chat_id = 2
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': chat_id})

        ency_answer, chat_subject = obj.ask_question(self.que, files=self.file_query,
                                                     chat_history={},
                                                     **additional_args)
        answer = Schema.decrypt(ency_answer)
        answer = json.loads(answer)
        self.assertIsNotNone(chat_subject)
        self.assertTrue(('apologize' not in answer['response'].lower()) and
                        ('sorry' not in answer['response'].lower()))
        self.assertEqual(len(self.files),
                         len(dict.fromkeys([a['source'] for a in answer['files']])))

    def test_multiple_file_chat_subject(self):
        chat_id = 4
        # obj.add_document(self.files)
        # question = 'list types of pumps'

        # encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': chat_id, 'chat_subject': 'pumps description'})

        ency_answer, chat_subject = obj.ask_question(self.que, files=self.files_, chat_history={},
                                                     **additional_args)
        answer = Schema.decrypt(ency_answer)
        answer = json.loads(answer)

        # TODO: uncomment below to completely test the return types
        # self.assertIsNone(chat_subject)
        self.assertTrue(('apologize' not in answer['response'].lower()) and
                        ('sorry' not in answer['response'].lower()))
        for i in dict.fromkeys([a['source'] for a in answer['files']]):
            self.assertIn(i, [j['remote_id'] for j in self.files])

        # ency_answer = obj.ask_question(self.que, chat_history={},
        #                                **additional_args)
        # answer = Schema.decrypt(ency_answer)

        # obj.add_document(file)
        # files = [file, None, Files, None]
        # chat_id = [None, 1, 2, 3]
        # chat_history = {}
        # params = [dict(name='Temperature', value='0.0', data_type='float')]
        # additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': params}
        # ency_answer = obj.ask_question(self.user_input, files=file, chat_history={},
        #                                **additional_args)
        # answer = Schema.decrypt(ency_answer)
        # print(answer)
        # self.assertIn('sorry', answer.lower())
        #
        # obj.delete_document(file)

    def test_subject_lang_en(self):
        chat_id = 1
        chat_history = {}

        params = [dict(name='Temperature', value='0.0', data_type='float')]
        additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': params,
                           "chat_subject": None, "extras": {},
                           "model_name": "gpt-4o", }

        obj.ask_question(
            self.que, files=self.files, chat_history=chat_history, **additional_args)

        # answer = Schema.decrypt(ency_answer)
        # answer = json.loads(answer)
        # self.assertIsNotNone(chat_subject)
        # self.assertTrue(('apologize' not in answer['response'].lower()) and
        #                 ('sorry' not in answer['response'].lower()))
        # self.assertIsInstance(answer['files'], list)
        # self.assertNotEqual(len(answer['files']), 0)

    def test_subject_lang_de(self):
        chat_id = 1
        chat_history = {}
        obj.add_document([
            {"remote_id": "flowserve/71569212_DE_AQ.pdf", "id": 1, 'name': 'file3'},
        ])
        params = [dict(name='Temperature', value='0.0', data_type='float')]
        additional_args = {'chat_id': chat_id, 'lang': 'de', 'params': params,
                           "chat_subject": None}
        ency_answer, chat_subject = obj.ask_question(
            self.que_de, files=None, chat_history=chat_history, **additional_args)

        answer = Schema.decrypt(ency_answer)
        answer = json.loads(answer)
        self.assertIsNotNone(chat_subject)
        self.assertTrue(('Entschuldigung' not in answer['response'].lower()) and
                        ('Entschuldigungen' not in answer['response'].lower()))
        self.assertIsInstance(answer['files'], list)
        self.assertNotEqual(len(answer['files']), 0)

    def test_answer_if_files_deleted_params(self):
        chat_id = 9
        obj.delete_document(self.files)
        time.sleep(10)
        params = dict(name='Temperature', value='0.5', data_type='float')
        additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': [params],
                           "chat_subject": None}
        ency_answer = obj.ask_question(self.que, files=self.files, chat_history={},
                                       **additional_args)
        # assert (answer, "I'm sorry, I'm unable to answer this question as I don't have any
        # relevant documents to provide an answer." )
        answer = Schema.decrypt(ency_answer)

        print(answer)
        self.assertIn('sorry', answer.lower())
        # obj.delete_document(self.files)

    def test_check_disjoint_filter_de_file_en_lang(self):
        file = [
            {'remote_id': 'flowserve/71569195_DE_A4.pdf'},
            {'remote_id': 'flowserve/PS-20-1_DE_A4.pdf'},
        ]

        chat_id = 41
        params = [dict(name='Temperature', value='0.0', data_type='float')]
        additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': params,
                           "chat_subject": None}
        ency_answer, chat_subject = obj.ask_question(self.que, files=file, chat_history={},
                                                     **additional_args)
        answer = Schema.decrypt(ency_answer)
        answer = json.loads(answer)

        self.assertTrue(('apologize' in answer['response'].lower()) or
                        ('sorry' in answer['response'].lower()))
        self.assertIsInstance(answer['files'], list)
        self.assertNotEqual(len(answer['files']), 0)

    # def test_language_check(self):
    #     obj.ask_question(file)

    def test_check_empty_chunks(self):
        chat_id = None

        params = dict(name='Temperature', value='0.0', data_type='float')
        additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': [params],
                           "chat_subject": None}
        ency_answer = obj.ask_question(self.que, files=self.empty_file, chat_history={},
                                       **additional_args)
        answer, chat_subject = Schema.decrypt(ency_answer)
        answer = json.loads(answer)

        self.assertTrue(('apologize' in answer['response'].lower()) or
                        ('sorry' in answer['response'].lower()))
        self.assertIsInstance(answer['files'], list)
        self.assertEqual(len(answer['files']), 0)

    def test_extensive_test(self):
        questions = [
            'hello',
            'what is pump',
            'tell me about seal-less pumps',
            'what is the specification for the seal-less pump'
        ]
        file = [
            {'remote_id': 'flowserve/71569212_EN_AQ.pdf', 'id':2},
            # {'remote_id': 'flowserve/71569195_DE_A4.pdf'},
            # {'remote_id': 'flowserve/PS-20-1_DE_A4.pdf'},
        ]
        # obj.add_document(file)
        files = [file, None, Files, None]
        chat_id = [None, 1, 2, 3]
        chat_history = {}
        params = [dict(name='Temperature', value='0.0', data_type='float')]
        for que, fil, ch_id in zip(questions, files, chat_id):
            encrypt_que = Schema.encrypt(que)
            additional_args = {'chat_id': ch_id, 'lang': 'en', 'params': params,
                               'chat_subject': 'pumps description',
                               "extras": {"web_search_enabled": False,
                                          "tenant_id": 2},
                               }
            ency_answer = obj.ask_question(encrypt_que, files=fil, chat_history=chat_history,
                                           model_name='deepseek-v3',
                                           **additional_args)
            answer = Schema.decrypt(ency_answer)
            chat_history[time.time()] = store_message(
                Schema.encrypt(que),
                Schema.encrypt(str(answer))
            )
            print(que)
            print(answer)
        obj.executor.shutdown(wait=True)

    def test_assistant_api(self):
        chat_id = 101
        chat_history = {}

        params = [dict(name='Temperature', value='0.0', data_type='float')]
        additional_args = {'chat_id': chat_id, 'lang': 'en', 'params': params,
                           "chat_type":'E',  "chat_subject": None,
                           "model_name": 'deepseek-v3',
                           "extras":{"web_search_enabled": True},
                           "ck": Schema.encrypt("testing")
                           }
        name = 'jarvis'
        question = Schema.encrypt("what is the document about")
        description = "you are an assistant that writes in shakes peer style"
        instructions = "be concise in you description"

        obj.ask_assistant(name, question, description, instructions,
             files=[{'id':7986, }], chat_history=chat_history, **additional_args)



        ency_answer, chat_subject = obj.ask_assistant(
            self.que, files=None, chat_history=chat_history, **additional_args)

        answer = Schema.decrypt(ency_answer)
        answer = json.loads(answer)
        self.assertIsNotNone(chat_subject)
        self.assertTrue(('apologize' not in answer['response'].lower()) and
                        ('sorry' not in answer['response'].lower()))
        self.assertIsInstance(answer['files'], list)
        self.assertNotEqual(len(answer['files']), 0)


    def test_web_search(self):
        question = 'who is elon musk'
        encrypt_que = Schema.encrypt(question)
        additional_args = self.additional_args.copy()
        additional_args.update({'chat_id': 'web_test',
                                'lang': 'en',
                                "extras": {"web_search_enabled": True,
                                           "tenant_id": 2},
                                'chat_subject': 'something '})

        ency_answer, chat_subject = obj.ask_question(
            encrypt_que, files=None, chat_history='', chat_type='C', model_name='gpt-4o',
            **additional_args)

        obj.executor.shutdown(wait=True)


if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
    # loop.close()
