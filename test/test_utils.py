# from __future__ import annotations
# from typing import Any, Dict, List, Optional
# # from pydantic import Extra
# from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
# from langchain.chains import QAGenerationChain, LLMChain
# from langchain.evaluation.qa import QAEvalChain
# from langchain.llms import AzureOpenAI, OpenAI
# import os
# import time
# import json
# import random
# import itertools
# from PyPDF2 import PdfReader
#
# CHUNK_SIZE = 1500
# CHUNK_OVERLAP = 200
# LLM_COMMON_MODEL_NAME = os.getenv("LLM_COMMON_MODEL_NAME")
# LLM_LARGE_MODEL_NAME = os.getenv("LLM_LARGE_MODEL_NAME")
# LLM_COMMON_DEPLOYMENT_NAME = os.getenv("LLM_COMMON_DEPLOYMENT_NAME")
# LLM_LARGE_DEPLOYMENT_NAME = os.getenv("LLM_LARGE_DEPLOYMENT_NAME")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
# OPENAI_CHAT_API_VERSION = os.getenv("OPENAI_CHAT_API_VERSION")
# OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
#
# # llm = AzureOpenAI(
# #     deployment_name=LLM_COMMON_DEPLOYMENT_NAME,
# #     model_name=LLM_COMMON_MODEL_NAME,
# #     temperature=0.0,
# #
# #     # max_tokens=500,
# # )
# #
# # chat_llm = AzureChatOpenAI(
# #     deployment_name=LLM_LARGE_DEPLOYMENT_NAME,
# #     # model_name=LLM_LARGE_DEPLOYMENT_NAME,
# #     openai_api_base=OPENAI_API_BASE,
# #     openai_api_version=OPENAI_CHAT_API_VERSION,
# #     openai_api_type=OPENAI_API_TYPE,
# #     temperature=0.0,
# #     # max_tokens=max_tokens,
# # )
#
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # OPENAI_ORG = os.getenv("OPENAI_ORG")
# # llm = OpenAI(temperature=0)
# # chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#
# class QA_Evaluation:
#
#     def __init__(self, language):
#         if language == 'en':
#             from .test_templates import (
#                 GRADE_ANSWER_PROMPT,
#                 CHAT_PROMPT,
#                 DOCUMENT_DESCRIPTION_PROMPT,
#                 QUESTION_FILTER_PROMPT
#             )
#
#         elif language == 'de':
#             from .test_templates_de import (
#                 GRADE_ANSWER_PROMPT,
#                 CHAT_PROMPT,
#                 DOCUMENT_DESCRIPTION_PROMPT,
#                 QUESTION_FILTER_PROMPT
#             )
#         else:
#             raise ValueError("Invalid language! Supported languages are 'english' and 'german'.")
#
#         self.GRADE_ANSWER_PROMPT = GRADE_ANSWER_PROMPT
#         self.CHAT_PROMPT = CHAT_PROMPT
#         self.DOCUMENT_DESCRIPTION_PROMPT = DOCUMENT_DESCRIPTION_PROMPT
#         self.QUESTION_FILTER_PROMPT = QUESTION_FILTER_PROMPT
#
#     def generate_summary(self, first_page):
#         # llm = AzureOpenAI(temperature=0.0)
#         summary_stuff = LLMChain(llm=llm, prompt=self.DOCUMENT_DESCRIPTION_PROMPT)
#         return summary_stuff.run({'text': first_page})
#
#     def post_filter(self, ques_ans, title):
#
#         # llm = AzureOpenAI(temperature=0.0)
#         filter_chain = LLMChain(llm=llm, prompt=self.QUESTION_FILTER_PROMPT)
#         data_combined = f''' Question Answer Pair: {ques_ans}
#                                         ======================
#                                             Title: {title}
#                                         '''
#         return filter_chain.run({'text': data_combined})
#
#     def generate_eval(self, data: list, num_questions: int, chunk: int, doc_summary: str, file_name : str):
#         """
#         Generate eval set
#         @param data: List of documents to generate eval set from
#         @param num_questions: number of questions to generate
#         @param chunk: chunk size to draw question from in the doc
#         @param doc_summary: Summary of document from which piece of chunk is extracted
#         @return: eval set as JSON list
#         """
#
#         n = len(data)
#
#         starting_indices = [random.randint(3, n - chunk) for _ in range(num_questions)]
#         sub_sequences = []
#         for i in starting_indices:
#             try:
#                 text = ''
#                 page_number = []
#                 for page_num in range(i, i + chunk):
#                     # print (len(data), i)
#                     text += data[page_num].page_content
#                     page_number.append(page_num)
#                 sub_sequences.append({'content': text, 'page_num': page_number})
#             except:
#                 print('Chunk Not Available in the document')
#
#             # sub_sequences = [data[i:i + chunk] for i in starting_indices]
#         chain = QAGenerationChain.from_llm(chat_llm, prompt=self.CHAT_PROMPT)
#         eval_set = []
#         for i, text_ in enumerate(sub_sequences):
#             try:
#                 # append summary
#                 text_combined = f'''
#                                     Title: {text_['content']}
#                                     ======================
#                                     Text: {doc_summary}
#                                     '''
#                 # print (text_combined)
#                 qa = chain.run(text_combined)
#                 # Apply a filter here
#                 # filter_resp = json.loads(self.post_filter(qa, doc_summary))
#                 filter_resp = self.post_filter(qa, doc_summary)
#                 # print(doc_summary)
#                 # print(qa)
#                 # print(filter_resp)
#                 type(qa)
#                 if '1' in filter_resp:
#                     qa[0]['page_number'] = text_['page_num']
#                     qa[0]['file_name'] = file_name
#                     eval_set.append(qa)
#                     print('QA Pair Generated')
#
#             except Exception as e:
#                 print('Error generating questions')
#                 print(e)
#                 # raise e
#
#         eval_set_full = list(itertools.chain.from_iterable(eval_set))
#         return eval_set_full
#
#     def grade_model_answer(self, predicted_dataset: List, predictions: List, grade_answer_prompt: str) -> List:
#         """
#         Grades the distilled answer based on ground truth and model predictions.
#         @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
#         @param predictions: A list of dictionaries containing model predictions for the questions.
#         @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
#         @return: A list of scores for the distilled answers.
#         """
#         # Grade the distilled answer
#         # Set the grading prompt based on the grade_answer_prompt parameter
#         # if grade_answer_prompt == "Fast":
#         #     prompt = GRADE_ANSWER_PROMPT_FAST
#         # elif grade_answer_prompt == "Descriptive w/ bias check":
#         #     prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
#         # elif grade_answer_prompt == "OpenAI grading prompt":
#         #     prompt = GRADE_ANSWER_PROMPT_OPENAI
#         # else:
#         #     prompt = self.GRADE_ANSWER_PROMPT
#
#         prompt = self.GRADE_ANSWER_PROMPT
#         # Create an evaluation chain
#         eval_chain = QAEvalChain.from_llm(
#             llm=chat_llm,
#             prompt=prompt
#         )
#
#         # Evaluate the predictions and ground truth using the evaluation chain
#         graded_outputs = eval_chain.evaluate(
#             predicted_dataset,
#             predictions,
#             question_key="question",
#             prediction_key="answer"
#         )
#
#         return graded_outputs
#
#     # def grade_model_retrieval(self, gt_dataset: List, predictions: List, grade_docs_prompt: str):
#     #     """
#     #     Grades the relevance of retrieved documents based on ground truth and model predictions.
#     #     @param gt_dataset: list of dictionaries containing ground truth questions and answers.
#     #     @param predictions: list of dictionaries containing model predictions for the questions
#     #     @param grade_docs_prompt: prompt level for the grading. Either "Fast" or "Full"
#     #     @return: list of scores for the retrieved documents.
#     #     """
#     #
#     #     # Grade the docs retrieval
#     #     # Set the grading prompt based on the grade_docs_prompt parameter
#     #     prompt = GRADE_DOCS_PROMPT_FAST if grade_docs_prompt == "Fast" else GRADE_DOCS_PROMPT
#     #
#     #     # Create an evaluation chain
#     #     eval_chain = QAEvalChain.from_llm(
#     #         llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
#     #         prompt=prompt
#     #     )
#     #
#     #     # Evaluate the predictions and ground truth using the evaluation chain
#     #     graded_outputs = eval_chain.evaluate(
#     #         gt_dataset,
#     #         predictions,
#     #         question_key="question",
#     #         prediction_key="answer"
#     #     )
#     #     return graded_outputs
#
#     def run_evaluation(self, chain_obj, retriever, eval_set, file_query, grade_prompt, lan, retriever_type,
#                        num_neighbors):
#         """
#         Runs evaluation on a model's performance on a given evaluation dataset.
#         @param chain_obj: Model chain used for answering questions
#         @param retriever:  Document retriever used for retrieving relevant documents
#         @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
#         @param file_query: List of files, from where answer should be generated
#         @param grade_prompt: String prompt used for grading model's performance
#         @param lan: String either 'en' or 'de'
#         @param retriever_type: String specifying the type of retriever used
#         @param num_neighbors: Number of neighbors to retrieve using the retriever
#         @return: A tuple of four items:
#         - answers_grade: A dictionary containing scores for the model's answers.
#         - retrieval_grade: A dictionary containing scores for the model's document retrieval.
#         - latencies_list: A list of latencies in seconds for each question answered.
#         - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
#         """
#
#         predictions_list = []
#         retrieved_docs = []
#         gt_dataset = []
#         latencies_list = []
#         chat_id = 4
#         chat_history = ''
#         count = 0
#         for data in eval_set:
#             count = count + 1
#             print(count)
#             time.sleep(1)
#             # Get answer and log latency
#             start_time = time.time()
#             params = dict(name="Temperature", value="0.2", data_type="float")
#             additional_args = {
#                 "chat_id": count,
#                 "lang": lan,
#                 "params": [params],
#                 "chat_subject": "New Chat",
#             }
#
#             answer = chain_obj.ask_question(data['question'], files=file_query, chat_history=chat_history,
#                                             **additional_args)
#             print ('answer', answer)
#             predictions_list.append({'answer': answer[0]})
#             gt_dataset.append(data)
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             latencies_list.append(elapsed_time)
#
#
#         # Grade
#         answers_grade = self.grade_model_answer(gt_dataset, predictions_list, grade_prompt)
#         # retrieval_grade = grade_model_retrieval(gt_dataset, retrieved_docs, grade_prompt)
#
#         return answers_grade, latencies_list, predictions_list  # ,retrieval_grade,
#
#     async def async_generate(self, chain, query):
#         resp = await chain.arun(query)
#         return resp
#
#     def run_async_evaluation(self, chain_obj, retriever, eval_set, file_query, grade_prompt, lan, retriever_type,
#                        num_neighbors):
#         """
#         Runs evaluation on a model's performance on a given evaluation dataset.
#         @param chain_obj: Model chain used for answering questions
#         @param retriever:  Document retriever used for retrieving relevant documents
#         @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
#         @param file_query: List of files, from where answer should be generated
#         @param grade_prompt: String prompt used for grading model's performance
#         @param lan: String either 'en' or 'de'
#         @param retriever_type: String specifying the type of retriever used
#         @param num_neighbors: Number of neighbors to retrieve using the retriever
#         @return: A tuple of four items:
#         - answers_grade: A dictionary containing scores for the model's answers.
#         - retrieval_grade: A dictionary containing scores for the model's document retrieval.
#         - latencies_list: A list of latencies in seconds for each question answered.
#         - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
#         """
#
#         predictions_list = []
#         gt_dataset = []
#         latencies_list = []
#         chat_history = ''
#         count = 0
#         for data in eval_set:
#             count = count + 1
#             print(count)
#
#             start_time = time.time()
#             params = dict(name="Temperature", value="0.2", data_type="float")
#             additional_args = {
#                 "chat_id": count,
#                 "lang": lan,
#                 "params": [params],
#                 "chat_subject": "New Chat",
#             }
#
#             answer = chain_obj.ask_question(data['question'], files=file_query, chat_history=chat_history,
#                                             **additional_args)
#             print ('answer',answer)
#             predictions_list.append({'answer': answer[0]})
#             gt_dataset.append(data)
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             latencies_list.append(elapsed_time)
#
#         # Grade
#         answers_grade = self.grade_model_answer(gt_dataset, predictions_list, grade_prompt)
#         # retrieval_grade = grade_model_retrieval(gt_dataset, retrieved_docs, grade_prompt)
#
#         return answers_grade, latencies_list, predictions_list
#
#     # def grade_model_retrieval(self, eval_set: List, top_n: int, retriever):
#     #     """
#     #         Grades the relevance of retrieved documents based on ground truth and model predictions.
#     #         @param eval_set: list of dictionaries containing ground truth questions and answers.
#     #         @param top_n: Top_n samples to be matched and extracted from
#     #     """
#     #
#     #     # Grade the docs retrieval
#     #     for qa in eval_set:
#     #         extracted_question = qa['question']
#     #         page_number = qa['page_number']
#     #         # Get Top_n chunks from
#     #         docs = retriever.get_relevant_documents(extracted_question)
#     #         response = docs[0]
#     #         print (response)
