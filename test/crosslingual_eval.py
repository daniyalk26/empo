import os
import json
import time
import logging
import unittest
import dotenv
import pandas as pd

os.environ['LOCAL'] = 'True'
os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres"
dotenv.load_dotenv()

from .evaluation import Evaluation
from doclm import get_interactive
obj = get_interactive()
import mlflow

# dotenv_file = dotenv.find_dotenv()
mlflow.set_tracking_uri("http://51.142.69.27:5000/")

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

# def list_files(dir, format, exception_list):
#     file = []
#     for root, dirs, files in os.walk(dir):
#         for name in files:
#             file_name = os.path.join(root, name)
#             if format == 'pdf':
#                 if file_name not in exception_list:
#                     file.append({'remote_id': file_name, 'id': 'test', 'name': file_name, 'format': 'pdf', 'original_format': 'pdf'})
#             elif format == 'ppt':
#                 file.append({'remote_id': file_name, 'id': 'test', 'name': file_name, 'format': 'ppt'})
#     return file

class TestCrosslingualBase(unittest.TestCase):

    def setUp(self):
        self.experiment_name = os.getenv("EXP_NAME", 'test_experiment')
        self.format = 'pdf'
        self.language = os.getenv('LANGUAGE')
        self.top_n = int(os.getenv('TOP_N'))
        self.folder_path = './test_doc/cross_lingual/data_english'

        self.eval_file = 'eval_set/cross_lingual/en_qa_tr_150.csv'
        self.response_file = os.getenv("RES_FILE_NAME", 'test_file')
        os.environ["RESP_COMP"] = 'False'
        # # Load Evaluation set and parser files
        self.eval_set = pd.read_csv(self.eval_file)
        # self.eval_set = self.eval_set.head(1)
        self.files = list_files(self.folder_path, self.format)
        # print(self.files )
        # print(len(self.files))
        # # Generate embeddings of files from test set and store embeddings in DB
        #
        # obj.add_document(self.files)
        self.retriever = obj.store_db.as_retriever()
        # ## Create an evaluation class object and generate responses
        #
        self.evaluation_obj = Evaluation(language=self.language)
        # self.evaluation_obj.generate_responses(self.eval_set, self.retriever, self.files, self.language)
        # # while os.environ.get('RESP_COMP') != 'True':
        # #     time.sleep(1)
        # # print('file written succesfully')
        self.resp_set = pd.read_csv(self.response_file)
        #
        # join with eval_set dataframe
        self.resp_set.rename(columns={'chat_id': 'Sr'}, inplace=True)
        self.merged_df = pd.merge(self.resp_set, self.eval_set, on="Sr")
        # print(self.merged_df.columns)
        self.result = self.evaluation_obj.execute_llm_evaluation_retr(self.merged_df)
        self.result.to_csv('./results/cross_lingual/' + self.experiment_name, index=False)


    def test_param_parsert(self):
        self.assertEqual('New Chat', 'New Chat')