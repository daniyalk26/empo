import os
import json
import time
import logging
import unittest
import dotenv
import pandas as pd

os.environ['LOCAL'] = 'True'
# os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres"
dotenv.load_dotenv()

from .evaluation import Evaluation
from doclm import get_interactive
from doclm.util import Filemeta

from .test_config import test_config_dict
obj = get_interactive()
import mlflow

# dotenv_file = dotenv.find_dotenv()
mlflow.set_tracking_uri("http://51.142.69.27:5000/")

# def list_files(dir, format):
#     file = []
#     for root, dirs, files in os.walk(dir):
#         for name in files:
#             file_name = os.path.join(root, name)
#             if format == 'pdf':
#                 file.append({'remote_id': file_name, 'id': 1, 'name': file_name, 'format': 'pdf', 'original_format': 'pdf'})
#             elif format == 'ppt':
#                 file.append({'remote_id': file_name, 'id': 1, 'name': file_name, 'format': 'ppt'})
#     return file

def list_files(dir, format):
    file = []
    count=0
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_name = os.path.join(root, name)
            if format == 'pdf':
                # count+=1
                file.append({'remote_id': file_name, 'id': 1, 'name': file_name, 'format': 'pdf', 'original_format': 'pdf',
                             'lang': 'en'})
            elif format == 'ppt':
                file.append({'remote_id': file_name, 'id': 'test', 'name': file_name, 'format': 'ppt'})
    return file

class TestPerformanceRetrieverPDF(unittest.TestCase):

    def setUp(self):
        doc_format = 'pdf'
        analysis_type = 'add_chunk'
        experiment_name = '9900'
        # experiment_name = 'test_structure_recursive_1200_250_plus_minus_1_test'
        self.eval_metric = 'sas'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        folder_path = './test_doc/data_add/'
        files = list_files(folder_path, doc_format)
        # remove_list = ['OPENAI_CHAT_API_VERSION', 'OPENAI_API_TYPE', 'OPENAI_API_VERSION',
        #                'OPENAI_API_BASE', 'OPENAI_CHAT_API_VERSION', 'OPENAI_API_KEY']
        # Load Evaluation set
        eval_file = 'eval_set/qa_set_clean.csv'
        eval_set = pd.read_csv(eval_file)
        evaluation_obj = Evaluation(language=self.language)
        experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Add Files to DB
            obj.add_document(files)
            if analysis_type == 'add_chunk':
                retriever = obj.store_db.as_retriever(search_kwargs={"k": 15})
            else:
                retriever = obj.store_db.as_retriever()

            self.correct_ret_file, self.correct_ret_page, self.results_dict, result_details = evaluation_obj.execute_retrieval_evaluaion(
                eval_df=eval_set, top_n=self.top_n,
                retriever=retriever, eval_metric=self.eval_metric, analysis_type=analysis_type)
            # results_json = result_details.to_json()
            # params_log = {'results': results_json}
            # mlflow.log_params(params_log)
            mlflow.log_metrics({'PER_CORR_FILE': self.correct_ret_file,
                                'PER_CORR_PAGE': self.correct_ret_page,
                                })
            mlflow.log_metrics(self.results_dict)
            result_details.to_csv('./results/' + experiment_name, index=False)

    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')


class TestPerformanceRetrieverPPT(unittest.TestCase):

    def setUp(self):
        doc_format = 'ppt'
        analysis_type = 'add_chunk'
        # experiment_name = 'sdsdd'
        experiment_name = '150_simple_default_ppt'
        self.eval_metric = 'sas'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        # folder_path = './test_doc/ppt_data/'
        folder_path = './test_doc/ppt_data/'
        files = list_files(folder_path, doc_format)
        # remove_list = ['OPENAI_CHAT_API_VERSION', 'OPENAI_API_TYPE', 'OPENAI_API_VERSION',
        #                'OPENAI_API_BASE', 'OPENAI_CHAT_API_VERSION', 'OPENAI_API_KEY']
        # Load Evaluation set
        eval_file = 'eval_set/qa_set_clean.csv'
        eval_set = pd.read_csv(eval_file)
        evaluation_obj = Evaluation(language=self.language)
        # experiment_id = mlflow.create_experiment(experiment_name)
        # with mlflow.start_run(experiment_id=experiment_id) as run:
        if True:
            # Add Files to DB
            if doc_format == 'ppt':
                obj.add_document(files, format=doc_format)
            if analysis_type == 'add_chunk':
                retriever = obj.store_db.as_retriever(search_kwargs={"k": 15})
            else:
                retriever = obj.store_db.as_retriever()

            self.correct_ret_file, self.correct_ret_page, self.results_dict, result_details = evaluation_obj.execute_retrieval_evaluaion(
                eval_df=eval_set, top_n=self.top_n,
                retriever=retriever, eval_metric=self.eval_metric, analysis_type=analysis_type)
            # results_json = result_details.to_json()
            # params_log = {'results': results_json}
            # mlflow.log_params(params_log)
            # mlflow.log_metrics({'PER_CORR_FILE': self.correct_ret_file,
            #                     'PER_CORR_PAGE': self.correct_ret_page,
            #                     })
            # mlflow.log_metrics(self.results_dict)
            result_details.to_csv('./results/' + experiment_name, index=False)

    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')


class TestPerformanceQA(unittest.TestCase):

    def setUp(self):
        self.experiment_name = 'qa_smart_scanned_doctr'
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/retry_files/'
        self.eval_file = 'eval_set/qa_set_clean.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.head(5)
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        while os.environ.get('RESP_COMP') != 'True':
            time.sleep(1)
        print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)

        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.result = self.evaluation_obj.execute_llm_evaluation(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')


class TestPerformanceQAOCR(unittest.TestCase):

    def setUp(self):
        self.experiment_name = 'qa_ocr_e2e'
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/scanned_docs/'
        self.eval_file = 'eval_set/qa_clean_scanned.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        self.eval_set = self.eval_set.head(5)
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        while os.environ.get('RESP_COMP') != 'True':
            time.sleep(1)
        print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)

        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.result = self.evaluation_obj.execute_llm_evaluation(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)

    def test_param_parser(self):
        self.assertEqual('New Chat', 'New Chat')

class TestPerformanceMapReduce(unittest.TestCase):

    def setUp(self):
        self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/retry_files/'
        self.eval_file = 'eval_set/qa_set_clean.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        while os.environ.get('RESP_COMP') != 'True':
            time.sleep(1)
        print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)

        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.result = self.evaluation_obj.execute_llm_evaluation(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestEvaluationCode(unittest.TestCase):

    def setUp(self):
        # self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.experiment_name = 'test_prompt'
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/retry_files/'
        self.eval_file = 'eval_set/qa_set_clean.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.iloc[131:]
        # self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        # self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        # while os.environ.get('RESP_COMP') != 'True':
        #     time.sleep(1)
        # print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)

        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.merged_df = self.merged_df.head(3)
        self.result = self.evaluation_obj.execute_llm_evaluation(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestStripDown(unittest.TestCase):

    def setUp(self):
        self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/data_add/'
        self.eval_file = 'eval_set/qa_set_clean.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.iloc[19:]
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        # self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        # while os.environ.get('RESP_COMP') != 'True':
        #     time.sleep(1)
        # print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)

        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.result = self.evaluation_obj.execute_llm_evaluation(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestQAFixStaging(unittest.TestCase):

    def setUp(self):
        self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/data_add/'
        self.eval_file = 'eval_set/qa_set_clean_fix_xling.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        self.eval_set = self.eval_set.head(1)
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses

        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        # while os.environ.get('RESP_COMP') != 'True':
        #     time.sleep(1)
        # # print('file written succesfully')
        # self.resp_set = pd.read_csv(self.response_file)
        #
        # # join with eval_set dataframe
        # self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        # self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        # self.result = self.evaluation_obj.execute_llm_evaluation_retr(self.merged_df)
        # self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestGeneralConversation(unittest.TestCase):

    def setUp(self):
        self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.language = os.getenv('LANGUAGE')
        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.general_conversation(query='Can you help me')

    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')


class TestMistralRAG(unittest.TestCase):

    def setUp(self):
        self.config = test_config_dict['mistral_large']
        self.experiment_name = self.config['exp_name']
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/data_add/'
        self.eval_file = 'eval_set/qa_set_clean_fix_xling.csv'
        self.response_file = self.config['res_file_name']
        os.environ["RES_FILE_NAME"] = self.response_file
        os.environ["LANGCHAIN_PROJECT"] = self.config['langchain_project']

        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.loc[130:]
        # print(str(self.eval_set.head(1)['question']))
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        ## Create an evaluation class object and generate responses
        finall_call_model= self.config['final_call_model']
        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language, finall_call_model)

        # while os.environ.get('RESP_COMP') != 'True':
        #     time.sleep(1)
        # # print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)
        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        self.result = self.evaluation_obj.execute_llm_evaluation_retr(self.merged_df)
        self.result.to_csv('./results/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')


class TestEmedderRAG(unittest.TestCase):

    def setUp(self):
        self.config = test_config_dict['embedding_large_4o_xling_wot_xset']
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/data_add/'
        self.eval_file = 'eval_set/qa_set_clean_fix_xling.csv'
        self.response_file = self.config['res_file_name']
        os.environ["RES_FILE_NAME"] = self.response_file
        os.environ["LANGCHAIN_PROJECT"] = self.config['langchain_project']
        os.environ["WITH_TRANSLATION"] = self.config['with_translation']
        os.environ["POSTGRES_CONN_STRING"] = 'postgresql://postgres:postgres@localhost:5432/postgres2'


        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.head(1)
        # self.eval_set = self.eval_set.loc[68:]
        # print(str(self.eval_set.head(1)['question']))
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        obj.add_document(self.files)
        # self.retriever = obj.store_db.as_retriever()
        # # ## Create an evaluation class object and generate responses
        # finall_call_model= self.config['final_call_model']
        # self.evaluation_obj = Evaluation(language=self.language)
        # # self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language, finall_call_model)
        #
        # # # while os.environ.get('RESP_COMP') != 'True':
        # # #     time.sleep(1)
        # # # # print('file written succesfully')
        # self.resp_set = pd.read_csv(self.response_file)
        # # join with eval_set dataframe
        # self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        # self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        # self.result = self.evaluation_obj.execute_llm_evaluation_retr(self.merged_df)
        # self.result.to_csv('./results/cross_lingual/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestEmedderRAGXset(unittest.TestCase):

    def setUp(self):
        self.config = test_config_dict['test_config']
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/data_add/'
        self.eval_file = 'eval_set/extended_qa_set_xling.csv'
        self.response_file = self.config['res_file_name']
        os.environ["RES_FILE_NAME"] = self.response_file
        os.environ["LANGCHAIN_PROJECT"] = self.config['langchain_project']
        os.environ["WITH_TRANSLATION"] = self.config['with_translation']
        # os.environ["POSTGRES_CONN_STRING"] = 'postgresql://postgres:postgres@localhost:5432/postgres'


        os.environ["RESP_COMP"] = 'False'

        # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        self.eval_set = self.eval_set.head(1)
        # self.eval_set = self.eval_set.loc[598:]
        # print(str(self.eval_set.head(1)['question']))
        self.files = list_files(self.folder_path, self.format)
        # Generate embeddings of files from test set and store embeddings in DB

        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        # # ## Create an evaluation class object and generate responses
        finall_call_model= self.config['final_call_model']
        self.evaluation_obj = Evaluation(language=self.language)
        self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language, finall_call_model)
        #
        # # # while os.environ.get('RESP_COMP') != 'True':
        # # #     time.sleep(1)
        # # # # print('file written succesfully')
        # self.resp_set = pd.read_csv(self.response_file)
        # # join with eval_set dataframe
        # self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        # self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        # self.result = self.evaluation_obj.execute_llm_evaluation_retr(self.merged_df)
        # self.result.to_csv('./results/cross_lingual/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')

class TestGenerateSummary(unittest.TestCase):

    def setUp(self):
        from documentlm.doclm.util import generate_summary_agent
        text = 'WK\n8. Mount coupling hub on the shaft with the aid of a pusher\ndevice.\n9. Align pump at the coupling (see section 8.3, ‘‘Coupling\nAlignment’’).\n10. Pull fixing bolts tight on pump feet (see section 8.5, ‘‘Final\nAlignment’’).\n11. Pack the stuffing boxes in accordance with section 12.3.1\n‘‘Assembly of Shaft Seal’’.\n12. Connect all the pipelines to the pump.13. Carry out a final coupling alignment check in accordance\nwith section 8.5 ‘‘Final Alignment’’.\n14. Fill the bearing housing (350) with oil (see sections 1.3\n‘‘Bearings’’ and 1.4 ‘‘Lubrication’’).\n15. Open isolating valve in suction line fully.16. Start up the pump (see section 10 ‘‘Commissioning’’).\n13. Operating Troubles, Causes and\nRemedies\nCaution : Before remedying operating troubles, check allmeasuring instruments used for reliability and accuracy.\n13.1 Operating Troubles Cause and suggested remedy see\nsection 13.2 & 13.3\n1. Pump fails to deliver liquid 1, 2, 3, 4, 5, 6, 8, 12,\n13\n2. Pump delivers insufficient 1, 2, 3, 4, 5, 8, 12, 13,\nliquid 19, 20, 21\n3. Total head is too low 3, 4, 5, 7, 8, 10, 12, 19,\n20, 21\n4. Sudden interruption of 1, 2, 3, 4, 9, 11, 13\ndelivery shortly after start-up\n5. Absorbed power is 6, 7, 9, 10, 17, 19\nexcessively high\n6. Excessive leakage at 14, 16, 18, 22, 23, 24,\nstuffing box 28, 29\n7. Life of packing is too short 14, 16, 18, 22, 23, 24,\n25, 26, 27, 28, 29\n8. Pump vibrates or runs noisily 2, 3, 4, 11, 13, 14, 15,\n16, 17, 18, 20, 24, 32,33\n9. Life of bearing is too short 14, 16, 17, 25, 31, 32,\n33, 34\n10. Excessively high temperature 1, 3, 4, 7, 9, 11, 13, 14,\ninside the pump. 16,Rotor fouls the casing or 18, 19, 20, 24, 30, 32seizes\n11. Too high a rate of leakage 14, 16, 18, 24, 25, 35,\nliquid at the mechanical seal 36, 37, 38, 39, 40or too short a mechanicalseal life\n13.2 Cause of Damage  (the numbers listed below\ncorrespond with the code numbers of section 13.1).Faults at the Suction End\n1. Pump not properly vented, air pocket in suction line,\nvapour bubble at suction end, lines not properly vented.\n2. Pump or suction line incompletely primed with fluid.\n3. Insufficient pressure differential between suction pressure\nand vapour pressure, NPSH required is not attained(observe rate of pressure decrease).\n4. Mouth of suction line too close to surface of liquid level in\nthe suction vessel, or liquid level in vessel too low.\nGeneral Faults in the Installation\n5. Rotational speed to low, or rate or minimum flow through\nby-pass excessive.\n6. Rotational speed too high.\n7. Reverse rotation.\n8. Total head required for the system is higher than total\nhead generated by the pump at duty point (back pressuretoo high).\n9. Total head required for the system is lower than total head\ngenerated by the pump (pump operates beyond theperformance limit curve).\n10. Specific gravity of fluid pumped is different from figure\nspecified originally (different operating temperature).\n11. Operation at very low rate of flow (fault in minimum flow\ndevice, rate of minimum flow is too low).\n12. Pumps cannot possibly operate in parallel under these\nconditions.\nMechanical Faults\n13. Foreign bodies lodged in impeller.\n14. Pump misaligned or incorrectly aligned, or shifting of\nfoundation.\n15. Resonance, or interference by other machines via the\nfoundation.\n16. Shaft is bent.\n17. Rotating elements foul the stationary elements, pump runs\nvery rough.\n18. Bearings badly worn.\n19. Casing wearing rings badly worn.20. Impeller damaged or disintegrated.\n21. Fault casing seal (excessive internal loss at throttling gap,\nrotor clearances exceeded due to wear), so that anexcessive loss arises or water leaks through the casingpartition. Water leaks out of metallic sealing face toatmosphere.\n22. Shaft or shaft protection sleeves worn or scored, O-ring\ndamaged.\n23. Stuffing box badly packed. Packing material of unsuitable\nquality.\n24. Shaft chatters because bearings are worn or because\nshaft is misaligned.\n25. Rotor vibrates.\n26. Stuffing box gland is tightened excessively, no fluid\navailable to lubricate the packing.\n27. Defect in cooling liquid supply to water-cooled stuffing\nbox gland.\n28. Excessive clearance gap between stuffing box gland and\nshaft protection sleeve. Packing is squeezed into gapbeneath the gland.\n25\nWK\n29. Dirt or sand in cooling liquid fed to stuffing box gland\ncauses scoring of shaft protection sleeve.\n30. Excessively high axial thrust.\n31. Insufficient quantity of oil in bearing housing, unsuitable\noil quality, dirty oil, water in the oil.\n32. Faulty bearing assembly (damage during assembly,\nwrong, assembly).\n33. Dirt in the bearings.34. Ingress of water into bearing housing.\n35. Rubbing faces of mechanical seal worn or scored. O-rings\ndamaged.\n36. Seal incorrectly assembled. Materials unsuitable.\n37. Surface pressure in sealing gap too high, no fluid available\nfor lubrication and cooling.\n38. Fault in cooling liquid supply system to mechanical seal.\n39. Excessively large gap between cooling housing and\nspacer sleeve. Temperature in the cooling circuit risesexcessively.\n40. Dirt in cooling circuit of mechanical seal leads to scoring\nof mechanical seal rubbing faces.\n13.3 Suggested RemediesIf, after a breakdown has occurred, one of the cause listed in\nsection 13.2 has been established as the cause, and the matterhas been put right or the cause of the trouble eliminated, it isrecommended, prior to recommissioning the set, to check theeffortless rotation of the pump rotor by hand, with the driverdisconnected (unless the pump had to be dismantled in anycase, because of the damage). Check that the pump runssmoothly and quietly after recommissioning.\nCause 1. Open vent valves or pressure gauge vent screws,\nopen isolating valves in minimum flow devicecircuit. Check layout of pipelines to ensure that fluidflows smoothly.\nCause 2. Prime pumps and piping again, and vent them\nthoroughly. Check layout of pipelines.\nCause 3. Check isolating valve and strainers in suction line.\nThe instrument readings taken must be accurate.Consult manufacturer.\nCause 4. Check water level in reservoir and examine\npossibility of altering it. Raise water level, altermouth of suction line. The nozzle should not projecttoo high inside the reservoir, and it should beshaped so as to promote favourable flowcharacteristics.\nCause 5. Increase speed, if pump it turbine-driven. Refer to\nmanufacturer, if pump is motor-driven. Checkoperation of minimum flow device.\nCause 6. Decrease speed, if pump is turbine-driven. Refer\nto manufacturer, if pump is motor driven.\nCause 7. Cross over two phase leads on the motor.\nCause 8. Increase rotational speed. Fit larger diameter\nimpellers. Increase number of stages. Refer tomanufacturer.\nCause 9. Adjust pressure conditions by means of discharge\nvalve. Alter rotation speed, alter impeller diameter.Refer to manufacturer.\nCause 10. Check temperature of fluid pumped, take steps\noutlined in 9, above.Cause 11. Check operation of minimum flow device. Refer to\nmanufacturer.\nCause 12. Check condition of individual machines. Refer to\nmanufacturer.\nCause 13. Clean out pump, check condition of suction system\n(check suction line and strainers).\nCause 14. Realign pumping set when cold.\nCause 15. Refer to manufacturer.Cause 16. Fit a new shaft. On no account straighten outt a\nbend shaft.\nCause 17. Dismantle pump.\nCause 18. Check quiet running of pumps. Check coupling\nalignment (when cold). Check oil quality andcleanlines.\nCause 19. Fit new casing wearing rings, Check out-of-round\n(true running of) rotor. Check presence of foreignbodies in the pump (see also item 16).\nCause 20. Fit new impeller. Check suction head (cavitation).\nCheck system for presence of foreign bodies (seealso item 16).\nCause 21. Replace damaged components by new ones.\nCause 22. Replace damaged components by new ones.\nCheck shaft protection sleeves for true running(out-of-round). Check suitability of packing materialused. Check that gland is not tightened askew andobserve rate of leakage.\nCause 23. Carefully repack stuffing box. Check suitability of\npacking material used.\nCause 24. Realign coupling (when cold). Fit new bearings.\nCheck rotor for signs of damage.\nCause 25. Check suction pressure (cavitation). Check\ncoupling alignment. Check pump internals forpresence of foreign bodies.\nCause 26. Repack stuffing box. Tighten gland lightly only.\nAlow slightly higher rate of gland leakage. Checksuitability of packing material used.\nCause 27. Check unobstructed flow through cooling liquid\nfeed line.\nCause 28. Fit an end ring or a new stuffing box gland. Check\ncondition of shaft protection sleeve.\nCause 29. Use treated cooling liquid. Fit filters in cooling liquid\nlines.\nCause 30. Check rotor clearances. Check axial adjustment\n(position) of rotor.\nCause 31. Check oil quality and quantity.\nCause 32. Check bearing components for signs of damage\nand assemble them correctly.\nCause 33. Throughly clean bearings, bearing housings and\ncheck condition of bearing of seal.\nCause 34. Remove all rust from b earings and bearing\nbrackets. Change the oil fill.\nCause 35. Replace damaged components by new ones.\nCheck rotating components for out-of-round. Checksuitability of materials used. Make sure all sealcomponents seat accurately, and lookout forleakage.\nCause 36. Carefully insert seal. Check materials for suitability.\nCause 37. Measure the seal anew. Refer to manufacturer.\n26\nWK\nCause 38. Check unobstructed flow through cooling liquid\nsupply line.\nCause 39. Fit a new bush or a spacer sleeve in the cooling\nhousing.\nCause 40. Use treated cooling liquid. Incorporate filters in\ncooling liquid line.14. Spare Parts\nWhen ordering spare parts, always please quote the itemnumbers, and designations of the items concerned, and theWorks serial number of the pump, in order to avoid any queriesand delays in delivery. The Works serial number of the pumpcan be obtained from the title page of the present instructionmanual, or from the rating plate on the pump.\nWe recommend keeping the following spare parts in stock in\norder to be in a position to remedy rapidly any operating troublewhich might arise.\nPart No.\nSpecial stuffingboxes HWV, VSM, VSH\nMechanical sealStandard\nstuffing box RemarkQuantity for pump\nconstruction with\n210 Shaft with keys 1 1 1 *)230 Impeller S S S *)320 Angular contact ball bearing 2 2 2 Only on heavy duty bearing bracket321 Deep groove roller bearing 1 1 1322 Cylindrical roller bearing 1 1 1400.1 Flat gasket 1 1 1400.2 Flat gasket S S S400.3 Flat gasket 2 4 4400.4 Flat gasket 2 2 2400.5 Flat gasket 1 1 1 Only on heavy duty bearing bracket400.6 Flat gasket - - 2412.2 O-ring 1 1 1412.3 O-ring 2 2 2 *)412.4 O-ring - 2 2412.7 O-ring 2 2 2 Only on heavy duty bearing bracket422.1 Felt ring 3 3 3 Only on heavy duty bearing bracket422.2 Felt ring 1 1 1 Only when bearing is sealed461.1 Stuffing box packing (in metres) 2 2 -472 Mechanical seal, complete - - 2502 Casing wearing ring S S S521 Stage sleeve S-1 S-1 S-1523.1 Shaft protection sleeve - - 1 *)524.1 Shaft protection sleeve 1 1 - *)524.2 Shaft protection sleeve 1 1 - *)525.1 Spacer sleeve 1 1 1 *)525.2 Spacer sleeve 1 1 1 *)540.1 Stage bush 1 1 1 Only on pump size 150540.2 Stage bush 1 1 1 Only on material alternative chrome steel540.3 Stage bush 1 1 1 Only on material alternative chrome steel541 Stage bush S-1 S-1 S-1 Only on material alternative chrome steel52.1 Adaptor sleeve, complete 1 1 1\nS = Number of stages\n*) Parts for complete pump rotor.The latter should be assembled and dynamically balanced, and kept in stock as a complete spare parts.\n27\n'
        print(generate_summary_agent(text, lang='en'))

    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')