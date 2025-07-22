import os, sys
import json
import time
import logging
import unittest
from dotenv import load_dotenv

load_dotenv()
sys.path.append("./")

from doclm.doclm import get_interactive
# from doclm.util import parse_param
from doclm.schema import Schema
# from doclm.tools.multi_tool_tools import get_enterprise_knowledge_tool, get_web_retriever_tool, get_attach_document_tool, get_planner_tool
# from doclm.util import get_lang_detector
logging.basicConfig(level=logging.INFO)
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

# class TestSyncApisSpecific(unittest.TestCase):
#     def setUp(self):
#         self.obj = get_interactive()

#         self.que_de = Schema.encrypt(
#             "Initiale Ausrichtungsprozedur während der Installation einer dichtungslosen Pumpe")
#         self.params = [dict(name="Temperature", value="0.0", data_type="float"),
#                        dict(name="Max length (tokens)", value="3000", data_type="int")]
#         self.additional_args = {'chat_id': -100, 'lang': 'en', 'params': self.params,
#                                 "chat_subject": "New Chat",
#                                 }


#     def test_attach_file_irrelevant_question(self):
#         question = 'tell me key points about flowserve pump'
#         question = 'give me general statics for Afghanistan'
#         chat_id = 41
#         # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
#         # processor_obj.add_document(self.files)

#         additional_args = self.additional_args.copy()

#         additional_args.update({'chat_id': chat_id,
#                                 'lang': 'en',
#                                 'chat_subject': 'something',
#                                 "extras": {
#                                     "namespace_id": 4,
#                                     "app_id": 3,
#                                     "rerank_bool": True,
#                                     "tenant_id": 2,
#                                     "username": "adminqa",
#                                     "lang": "en",
#                                     "lang_basis": "i",
#                                     "is_user_exempted_from_tokens_count": False,
#                                     "web_search_enabled": False,
#                                     "user_profile": {
#                                         "name": "Osama Khan",
#                                         "designation": "Chief Data Scientist",
#                                         "company_name": "Microsoft",
#                                         "country": "Argentina"
#                                     },
#                                 }
#                                 })


#         encrypt_que = Schema.encrypt(question)
#         f_files = [
#             {'id': 5, 'summary':'''The document details observations and recommendations for the "Go-Live Permit to Work System (WPMS)" at Adhi Field. It includes various high-priority issues that need addressing, such as mandatory training, system testing, and SOP development. Named entities include Adhi Field, PPL (Pakistan Petroleum Limited), Intech, and individuals like Mr. Abdul Samad SE (E&I).'''},
#             {'id': 1,
#              'summary': '''The document appears to be a detailed log of GPS-synchronized data points collected along a specific route from KM 0.000 to KM 5.000. It includes timestamps, latitude and longitude coordinates, signal quality, number of satellites, and other related metrics. Named entities identified include "KM 0 000 IJ SV1 STATION" and "FENCE," as well as geographical coordinates and dates.'''},
#             {'id': 2,
#              'summary': ''' The document appears to be a detailed log of GPS data points collected on June 20, 2022. It includes information such as chainage, on/off times, UTC time, latitude, longitude, signal quality, number of satellites, PDOP, altitude, and other related metrics. Named entities include GPS equipment, the date "2022-06-20," and geographical coordinates (latitude and longitude).'''},
#             {'id': 3,
#              'summary': '''The document provides statistical data for countries including Afghanistan, Albania, Algeria, American Samoa, and Andorra from 2000 to 2010. It covers categories such as transit (railways, passenger cars), business (mobile phone subscribers, internet users), health (mortality, health expenditure), population (total, urban, birth rate, life expectancy), and finance (GDP, GDP per capita).'''},
#         ]

#         ency_answer, chat_subject =  self.obj.ask_question_multi_tool(
#             encrypt_que, files=f_files, chat_history='', tools_list=["web_search"], model_name='gpt-4o',
#             **additional_args)

#         print(1)
#         self.obj.executor.shutdown(wait=True)


# class TestPlanAndExecute(unittest.TestCase):
#     def setUp(self):
#         self.obj = get_interactive()
#         self.ret = self.obj.store_db_reader_chat.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"score_threshold_semantic": 0.0, "score_threshold_keyword": 0.0})
#         self.planner_args = {
#                 "web_retriever": self.obj.web_retriever,
#                 "doc_retriever": self.ret,
#                 "chat_model": 'gpt-4o',
#                 "lang_detector": get_lang_detector(self.obj.lang_detector),
#                 "reranker": None,
#                 "reranker_bool": False,
#                 "tenant_id":2,
#                 "k": 20,
#                 "doc_filter":{'id':[1,2,3,4]},
#                 "attached_files_filter": {'id':[1,2]},
#                 "attached_file_context": Schema.extract_context([{'id':1, "summary":"Amarex pump"},{'id':2, "summary":"KSB pumps catalog"}]),
#                 "tools_list": ['default']#,'enterprise','web_search','attachments']
#                 }

#     def test_planner(self):
#         user_question = "How many holidays does my company offer" if os.getenv('ENCRYPT')=='False' else Schema.encrypt( "How many holidays does my company offer")
#         planner_tool = get_planner_tool(**self.planner_args)
#         planner_result = planner_tool.invoke({"tool_call_id":1,"state":
#             {"profile_kwargs": {"Name":"Jazib Jamil"}, "retriever_token_limit":0},
#                                               "query":user_question})
#         print(planner_result)
#         self.obj.executor.shutdown(wait=True)

#     def tearDown(self):
#         del self.obj
#         return super().tearDown()
    

class TestMultitool(unittest.TestCase):
    def setUp(self):
        self.obj = get_interactive()
        self.chat_id = 41
        self.params = [
            dict(name="Temperature", value="0.0", data_type="float"),
            dict(name="Max length (tokens)", value="3000", data_type="int"),
        ]
        # obj.create_tenant(tenant_id=2, application_id=3, collection_name='langchain')
        # processor_obj.add_document(self.files)
        self.additional_args = {
            "chat_id": self.chat_id,
            "rid":1,
            "lang": "en",
            "params": self.params,
            "chat_subject": "hello world",
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
                    "designation": "Operations Manager",
                    "company_name":"Riyad Electric Company",
                    "country":"Saudi Arabia",
                    "response_customization":"Emphasize on the key parts of your response by making them bold in markdown format."
                }
            }
        }
        self.files=[
                {"id":1,"summary":"This document is an installation and operating manual for the Amarex KRT submersible motor pump manufactured by KSB. It provides detailed instructions and guidelines for the installation, commissioning, servicing, and maintenance of the pump. The manual also includes information on safety regulations, transportation, electrical systems, and troubleshooting. The document contains a glossary of terms and an index for easy reference. Named entities identified in the text include KSB (manufacturer), Amarex KRT (product), and various sizes and types of pumps."},
                {"id":5,"summary":"The document is an installation and operating manual for the \"Etanorm SYT\" thermal oil/hot water pump, produced by KSB SE & Co. KGaA. KSB, headquartered in Germany, is a global leader in pumps and valves. The manual includes legal information and emphasizes the company's expertise in various industrial applications."}
                ]

    def test_multitool_planner(self):
        user_question = """Amarex pump vs Etaline pump design specification. detailed comparison. Which pump is suitable for what use cases. Use each tool with in plan_and_execute"""
            #if os.getenv('ENCRYPT')=='False' else Schema.encrypt( "Amarex pump design specifications")
        ency_answer, chat_subject = self.obj.ask_question_multi_tool(
            user_question, 
            tools_list=['plan_and_execute','web_search',"enterprise"],
            # tools_list=['plan_and_execute','web_search',"attached_documents","enterprise","parametric"],
            # files=[{"id":70},{"id":63}], web_search, enterprise,'plan_and_execute', attached_documents
            enterprise_files=self.files,
            attachment_files=self.files,
            chat_history={}, 
            **self.additional_args
        )
        print(ency_answer,chat_subject)
        self.obj.executor.shutdown(wait=True)

    def tearDown(self):
        del self.obj
        return super().tearDown()

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # while True:
    unittest.main()
    # loop.close()
