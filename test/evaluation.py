import os
import ast
import time
import json
import openai
import dotenv
import pandas as pd
from rouge_score import rouge_scorer
from .test_templates import (
    GRADE_ANSWER_PROMPT
)
os.environ['LOCAL'] = 'True'
os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres"
dotenv.load_dotenv()

dotenv.load_dotenv()
import re
from langchain.document_loaders import PyPDFLoader
from langchain.evaluation import load_evaluator
from langchain.chains import LLMChain
from sentence_transformers import CrossEncoder
from doclm.util import Filemeta
from doclm.llm_chains import get_evaluation_model

from doclm import get_interactive
obj = get_interactive()

NUM_PAGES = 2


class Evaluation:

    def __init__(self, language):

        # self.qa_chain = qa_chain
        # self.retriever = retriever
        self.graded_answers = None
        self.graded_retrieval = None
        self.latency = None
        self.predictions = None
        self.qa_eval = None
        self.score_time = 0
        self.score_ins = 0
        self.model_name = os.getenv("EVAL_MODEL_NAME", "gpt4")
        self.chat_llm = get_evaluation_model(self.model_name)
        self.eval_chain = LLMChain(llm=self.chat_llm, prompt=GRADE_ANSWER_PROMPT)
        self.generated_ans = []

    def output_parser(self, text):
        verdict = None
        score = None
        match_last = re.search(r"\s*(Y|N)\s*$", text, re.IGNORECASE)
        match_first = re.search(r"^\s*(Y|N)\s*", text, re.IGNORECASE)
        match_end = re.search(r"\b(Y|N)\b\s*$", text, re.IGNORECASE)

        if match_last:
            verdict = match_last.group(1).strip()
            text = text[: match_last.start()].strip()
        elif match_first:
            verdict = match_first.group(1).strip()
            text = text[match_first.end():].strip()
        elif match_end:
            verdict = match_end.group(1).strip()
            text = text[: match_end.start()].strip()
        else:
            splits = text.strip().rsplit("\n", maxsplit=1)
            if len(splits) == 1:
                reasoning = ""
                verdict = splits[0]
            else:
                reasoning, verdict = splits

        if verdict:
            score = (
                1 if verdict.upper() == "Y" else (0 if verdict.upper() == "N" else None)
            )

        return score

    # def _compute_correctness_llm(self, question, true_ans, pred_ans):
    #
    #     evaluator = load_evaluator("labeled_criteria", llm=self.chat_llm, criteria="correctness")
    #     # We can even override the model's learned knowledge using ground truth labels
    #     eval_result = evaluator.evaluate_strings(
    #         input=question,
    #         prediction=pred_ans,
    #         reference=true_ans,
    #     )
    #     # print (eval_result)
    #     return eval_result["score"]

    def _compute_correctness_llm(self, question, true_ans, pred_ans):

        text = self.eval_chain.run(dict(question=question, predicted_answer=pred_ans, true_answer=true_ans))
        return self.output_parser(text)

    def _evaluate_text_correctness(self, true_ans, pred_ans_list, model):

        score_list = []
        if isinstance(model, CrossEncoder):
            for pred_ans in pred_ans_list:
                start = time.time()
                score = model.predict([(true_ans, pred_ans.page_content)])
                end = time.time()
                elapsed_time = end - start
                self.score_time += elapsed_time
                self.score_ins += 1
                score_list.append(score[0])

        elif isinstance(model, rouge_scorer):
            for pred_ans in pred_ans_list:
                score = model.score(true_ans, pred_ans.page_content)['rougeL'].fmeasure
                score_list.append(score)

        return max(score_list)

    def data_load(self, file_name):
        loader = PyPDFLoader(file_name)
        return loader.load()

    def evaluator(self, *args, **kwargs):
        # msg = kwargs['msg']
        print('args', args)
        gen_answer= json.loads(args[0]['msg'])
        print('gen_answer', gen_answer['response'])
        self.generated_ans.append(gen_answer['response'])
        # print('msg', args[0]['msg'])
        # print('msg', args[0]['msg']['response'])

    def generate_qa_set(self, doc_list, num_questions, chunk_size):

        stack_eval_set = []
        for doc_ in doc_list:

            data = self.data_load(doc_)
            summary_content = ''
            for page_number in range(NUM_PAGES):
                # if data[page_number].page_content is not None:
                if len(data) > NUM_PAGES:
                    summary_content += data[page_number].page_content

            if summary_content is not None:
                doc_summary = self.qa_eval.generate_summary(summary_content)
            else:
                doc_summary = 'Title of this document cannot be inferred'

            stack_eval_set += self.qa_eval.generate_eval(data, round(num_questions / len(doc_list)), chunk_size,
                                                         doc_summary, file_name=doc_)
        return stack_eval_set

    def generate_responses(self, eval_df, retriever,  file_query, lan, finall_call_model):

        eval_set = [
            {'ind': row.Sr, 'question': row['question'], 'answer': row['answer'], 'page_number': row['page_number'],
             'file_name': row['file_name']} for index, row in eval_df.iterrows()]

        chat_history = {}
        params = [dict(name="Temperature", value="0.0", data_type="float"),
                  dict(name="Max length (tokens)", value="3000", data_type="int")]
        additional_args = {'chat_id': -100, 'lang': 'en', 'params': params,
                           "chat_subject": "New Chat",
                           }

        for qa in eval_set:
            # Ask Question Using Question from eval set
            encrypt_que = Filemeta.encrypt(qa['question'])
            additional_args = additional_args.copy()
            additional_args['chat_id'] = qa['ind']
            print('index_info', qa['ind'])
            obj.ask_question(encrypt_que, files=file_query, chat_history=chat_history, model_name=finall_call_model,
                             **additional_args)
            # ency_answer, chat_subject = obj.ask_question(
            #     encrypt_que, files=None, chat_history=chat_history, **additional_args)

    def execute_llm_evaluation(self, response_df):

        response_set = [
            {'ind': row.Sr, 'question': row['question'], 'answer': row['answer'], 'page_number': row['page_number'],
             'response': row['response'], 'file_name': row['file_name']} for index, row in response_df.iterrows()]
        result_df = response_df.copy()
        result_df['correct_answer'] = 0
        result_stack = []

        for qa in response_set:
            score = self._compute_correctness_llm(qa['question'], qa['answer'], qa['response'])
            result_df.loc[qa['ind']-1, 'correct_answer'] = score
            result_stack.append(score)

        return result_df

    def execute_llm_evaluation_retr(self, response_df):

        response_set = [
            {'ind': index, 'question': row['question_de'], 'answer': row['answer'], 'page_number': row['page_number'],
             'response': row['response'], 'file_name': row['file_name'], 'files': row['files']} for index, row in response_df.iterrows()]
        result_df = response_df.copy()
        result_df['correct_answer'] = 0
        result_df['correct_file'] = 0
        result_df['correct_page'] = 0
        result_stack = []

        for qa in response_set:
            print('evaluation running for Question:', qa['ind'])
            try:
                score = self._compute_correctness_llm(qa['question'], qa['answer'], qa['response'])
            except Exception as e:
                print(f"Error computing correctness: {e}")
                score = -1  # or some default value, e.g., 0 or -1
            # Identify correct file/page
            file_names = ast.literal_eval(qa['files'])
            corr_file = 0
            corr_page = 0
            for file in file_names:
                if qa['file_name'] in file['name']:
                    corr_file += 1
                    if file['page'] in [qa['page_number']-1, qa['page_number'], qa['page_number']+1]:
                        corr_page += 1

            result_df.loc[qa['ind'], 'correct_answer'] = score
            result_df.loc[qa['ind'], 'correct_file'] = corr_file
            result_df.loc[qa['ind'], 'correct_page'] = corr_page
            result_stack.append(score)

        return result_df

    def execute_retrieval_evaluaion(self, eval_df, top_n, retriever, eval_metric, analysis_type):
        # Grade the docs retrieval
        correct_ret_file = 0
        correct_ret_page = 0
        correct_file_page = 0
        result_details = []
        eval_set = [
            {'ind': row.Sr, 'question': row['question'], 'answer': row['answer'], 'page_number': row['page_number'],
             'file_name': row['file_name']} for index, row in eval_df.iterrows()]
        result_df = eval_df.copy()
        result_df['corr_file'] = 0
        result_df['corr_page'] = 0
        total_qa = len(eval_set)
        if eval_metric == 'sas':
            use_gpu = False
            device = None if use_gpu else "cpu"
            sas_model_name_or_path = "cross-encoder/stsb-roberta-large"
            use_auth_token = True
            model = CrossEncoder(
                sas_model_name_or_path,
                device=device,
                tokenizer_args={"use_auth_token": use_auth_token},
                automodel_args={"use_auth_token": use_auth_token},
            )
        elif eval_metric == 'rough':
            # Create a rouge scorer object
            model = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for qa in eval_set:
            # Get Top_n chunks from
            if analysis_type == 'add_chunk':
                retrieved_doc = []
                docs_ = retriever.get_relevant_documents(qa['question'])
                for doc in docs_:
                    new_doc = doc.copy()
                    prev_chunk = ''
                    next_chunk = ''
                    if doc.metadata.get('previous_chunk_text') is not None:
                        prev_chunk = doc.metadata['previous_chunk_text']
                    if doc.metadata.get('next_chunk_text') is not None:
                        next_chunk = doc.metadata['next_chunk_text']
                    new_doc.page_content = prev_chunk + doc.page_content + next_chunk
                    retrieved_doc.append(new_doc)
            else:
                docs = retriever.get_relevant_documents(qa['question'])
                retrieved_doc = docs[0:top_n]

            score = self._evaluate_text_correctness(qa['question'], retrieved_doc, model)
            page_match = False
            source_match = False
            page_store = []
            doc_count_acc = {}
            count = 0
            for doc_ in retrieved_doc:
                count += 1
                # print(qa['file_name'], doc_.metadata['source'])
                if qa['file_name'] in doc_.metadata['source']:
                    source_match = True
                    page_list = [int(doc_.metadata['page']) - 1, int(doc_.metadata['page']),
                                 int(doc_.metadata['page']) + 1]
                    if int(qa['page_number']) in page_list:
                        page_match = True
                        page_store.append(int(qa['page_number']))
                if count < 4:
                    doc_count_acc['file_found_' + str(count)] = source_match
                    doc_count_acc['page_found_' + str(count)] = page_match
                    doc_count_acc['score_' + str(count)] = self._evaluate_text_correctness(qa['question'],
                                                                                           retrieved_doc[0:count], model)
            if count <4:
                temp_dict = doc_count_acc.copy()
                for count_ in range(count+1, 4):
                    doc_count_acc['file_found_' + str(count_)] = source_match
                    doc_count_acc['page_found_' + str(count_)] = page_match
                    doc_count_acc['score_' + str(count_)] = doc_count_acc['score_' + str(count)]

            # score = doc_count_acc['score_' + str(count)]
            # Check whether extracted page number is same from which question was asked
            if source_match:
                correct_ret_file += 1
                result_df.loc[qa['ind'], 'corr_file'] = 1
            if page_match:
                result_df.loc[qa['ind'], 'corr_page'] = 1
                correct_ret_page += 1
            if source_match and page_match:
                correct_file_page += 1

            result_part = dict(ind=qa['ind'], question=qa['question'], answer=qa['answer'],
                               source_file=source_match, source_page=page_match, score=score,
                               page_num_matched=page_store
                               )
            result_comp = {**doc_count_acc, **result_part}
            result_details.append(result_comp)
        # Convert list of dicts to pandas dataframe
        result_details = pd.DataFrame(result_details)
        result_cols = result_df.columns.tolist()
        rem_list = ['question', 'answer', 'file_name', 'page_number', 'corr_file', 'corr_page']
        result_cols = [val for val in result_cols if val not in rem_list]
        results_dict = {}
        for col_name in result_cols:
            results_dict[col_name + '_' + 'corr_file'] = (result_df.loc[result_df[col_name] == 1, 'corr_file'].sum() / \
                                                          result_df[col_name].sum()) * 100
            results_dict[col_name + '_' + 'corr_page'] = (result_df.loc[result_df[col_name] == 1, 'corr_page'].sum() / \
                                                          result_df[col_name].sum()) * 100
        results_dict['avg_elsapsed_time'] = self.score_time / self.score_ins
        print('elapsed time is: ', self.score_time / self.score_ins)
        if correct_ret_file != 0:
            results_dict['CON_PROB'] = correct_file_page / correct_ret_file
        return (correct_ret_file / total_qa) * 100, (correct_ret_page / total_qa) * 100, results_dict, result_details
