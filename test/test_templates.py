from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt = ''''As an evaluator, you are tasked with assessing the performance of a chatbot. 
The evaluation is based on the comparison of the chatbot's predicted answer and the true answer 
to a given question.

Question: "{question}"
Predicted Answer: "{predicted_answer}"
True Answer: "{true_answer}"

Please evaluate the predicted answer in relation to the true answer as either Y for correct or N for incorrect. The
evaluation should focus 
on the content and meaning of the responses, rather than their length, writing style, punctuation, or phrasing. 
It's acceptable if the predicted answer is longer and contains more information, as long as it does not contain 
any conflicting statements with the true answer. First, write out in a step by step manner your reasoning about each 
decision to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. 
Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the 
correct answer of whether the predicted answer is same as true answer. At the end, repeat just the letter again by 
itself on a new line.
'''
GRADE_ANSWER_PROMPT= PromptTemplate(input_variables=["question", "predicted_answer", "true_answer"], template=prompt)


# """

# generate_qa_human = """Please come up with a question/answer pair from the provided text. The provided text consists of
# two parts:  the first part is a title of the document, and the second part is a randomly extracted chunk from the same
# document. To create the question and answer pair,  make sure to incorporate named entities from the title as well as
# text chunk provided. Make sure that the model should generate questions that require detailed responses, rather than
# simple yes/no answers. Please make sure to include named entities from title as well as extracted chunk in the document.
#
# Please provide the question and answer pair in the following JSON format:
#
# ```
# {{
#     "question": "$YOUR_QUESTION_HERE",
#     "answer": "$THE_ANSWER_HERE"
# }}
# ```
#
# Everything between the ``` must be valid json.
#
# Here is the provided text:
# ----------------
# {text}
# """
#
# generate_qa_human = """Please come up with a question/answer pair from the provided text. Here are the instructions to
# follow:
# Instructions:
# 1. Read the provided text carefully. It consists of two parts: a title and a randomly extracted chunk from the same
#    document.
# 2. Identify named entities in the title. The named entity mentioned in the title must refer to the equipment the document
#    is focused on.
# 3. Create questions that ask about the properties, maintenance, or installation instructions etc of the named entities.
# 4. Ensure that the questions require detailed responses rather than simple yes/no answers.
# 5. Make sure to avoid generic or ambiguous questions that refer to arbitrary sections or parts of the document.
# 6. Make sure that each question must contain the full name of named entity mentioned in the title.
#
# Please provide the question and answer pair in the following JSON format:
#
# ```
# {{
#     "question": "$YOUR_QUESTION_HERE",
#     "answer": "$THE_ANSWER_HERE"
# }}
# ```
#
# Everything between the ``` must be valid json.
#
# Here is the provided text:
# ----------------
# {text}
# """
#
# CHAT_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(generate_qa_system),
#         HumanMessagePromptTemplate.from_template(generate_qa_human),
#     ]
# )
#
# #GENERATE_QA_PROMPT = PromptTemplate(input_variables=["query"], template=template)
# extract_titles_template = """A piece of extracted text from the first few pages of the document is given to you below.
# You are supposed to tell us what is optimal title of this document, include named entities if they are present in the text.
#
# If no text is provided just say 'NO CONTEXT AVAILABLE'.
# If you are unable to tell what the document is about then just summarize the text given to you.
#
# Extracted text: {text}
#
# Brief document description:"""
# DOCUMENT_DESCRIPTION_PROMPT = PromptTemplate(template=extract_titles_template, input_variables=["text"])
#
# filter_questions_template = """
# A piece of text is provided to you which consists of two parts: first part is a question answering pair generated from
# a document and second part is title of the document from which this question-answering pair is extracted. In title of
# document, there will be name of an equipment. You have to read the question and check whether the named entity mentioned
# in title is mentioned in the question or not.
# If full name of named entity from title is mentioned in the question, then provide following answer in valid JSON format:
# '''
# {{
# "Answer": "1"
# }}
# '''
# If full name of named entity from title is not mentioned in the question, then provide following answer in valid JSON format:
# '''
# {{
# "Answer": "0"
# }}
# '''
#
# Everything between the ``` must be valid json.
#
# Provided text: {text}
# """
# QUESTION_FILTER_PROMPT = PromptTemplate(template=filter_questions_template, input_variables=["text"])
