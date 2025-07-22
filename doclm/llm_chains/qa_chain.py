"""
 module implements lang chain for answering user query's from db
"""
from __future__ import annotations

import re
import os

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import Extra, Field

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import Runnable

from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.utils import (
    Input,
    Output)

from .format_retrieve_data import FormatRetrieverDocumentChain, FormatRetrieverWebChain
from ..config import supported_templates, language_abbreviation, supported_lang_abbreviation
from ..exceptions import NotSupportedException
from ..tokenizer_cal import num_tokens_from_string, _url_to_size, _count_image_tokens

log = logging.getLogger("doclogger")
log.disabled = False
MIN_QUERY_LENGTH = 5

MAX_TOKENS_ROUTER = int(os.getenv('MAX_TOKENS_ROUTER', 150))
MAX_TOKENS_REPHRASE = int(os.getenv('MAX_TOKENS_REPHRASE', 500))
MAX_TOKENS_CHAT = int(os.getenv('MAX_TOKENS_CHAT', 3000))
MAX_TOKENS_TITLE = int(os.getenv('MAX_TOKENS_TITLE', 100))

# pylint: disable=C0103,R0903
def strip_string_parser(message):
    return message.content.strip("\"\'")

def convert_to_string(chat_tuples):
    """
    :param chat_tuples:list of tuples like [(q,'1'), (r,'1'), (q,'2'), (r,'2')]
    :return:
    """
    updated_chat_tuples = []
    for c in chat_tuples:
        if c[0] in ['human']:
            if isinstance(c[1], list):
                text = ''
                for item in c[1]:
                    if 'text' in item:
                        text += item['text']

                updated_chat_tuples.append(('human', text))
            elif isinstance(c[1], str):
                updated_chat_tuples.append(c)
            else:
                raise ValueError("Invalid Chat history")
        else:
            updated_chat_tuples.append(c)

    return ''.join([f"Question: {q[1]} \n Answer: {r[1]} \n" for q, r in zip(updated_chat_tuples[::2], updated_chat_tuples[1::2])])


def get_token_calculator_for_model(model_name):
    def token_calculator(msg: str):
        token_count = 0
        if isinstance(msg, list):
            for m in msg:
                if m.get('type')=='image_url':
                    image_size = _url_to_size(m["image_url"]["url"])
                    if not image_size:
                        continue
                    token_count += _count_image_tokens(*image_size)
                elif m.get('type')=='text':
                    token_count += num_tokens_from_string(m["text"], model_name)

        elif isinstance(msg, str):
            token_count += num_tokens_from_string(msg, model_name)
        else:
            raise ValueError(f'Unknown message {msg}')
        return token_count

    return token_calculator


def reduce_chats_below_limit(chats: List[Any], token_calculator, max_tokens) -> List[Any]:
    num_chats = -len(chats)
    log.info("got %s chats", num_chats)

    tokens = [token_calculator(doc[1]) for doc in chats]
    token_count = sum(tokens[num_chats:])
    while token_count > max_tokens:
        # need to deleted whole chat tuple question and response
        num_chats += 1
        token_count -= tokens[num_chats]
        num_chats += 1
        token_count -= tokens[num_chats]

    log.info("returning %s chats", num_chats)

    return chats[num_chats:]


def rephrase_question(input_kw, chat_history, common_llm, initial_question_template,
                      chat_condenser_prompt, run_manager, max_tokens, return_chain=False):
    if not chat_history:
        # Transform the original question into a new question
        log.info("building lang chain with initial templates CustomQA chain")
        template = initial_question_template
    else:
        template = chat_condenser_prompt
        human_msg = template[-1].format(chat_history=[], **input_kw)
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'

        human_msg_tokens = common_llm.get_num_tokens_from_messages([human_msg])

        model_name = common_llm.default.metadata.get('model', None) or common_llm.default.model_name
        token_calculator = get_token_calculator_for_model(model_name)
        chat_history = reduce_chats_below_limit(chat_history, token_calculator,
                                                common_llm.default.metadata["chat_history_limit_calculator"](human_msg_tokens))
        input_kw["chat_history"] = convert_to_string(chat_history)
        # Transform the original question into a new question
        log.info("building lang chain with initial templates")

    question_chain = template | common_llm | strip_string_parser

    log.info("building rephrased_question lang chain")
    if return_chain:
        return question_chain

    config = RunnableConfig(callbacks=run_manager.get_child(),
                            metadata={"stage_name": 'rephrase_chain'},
                            configurable={"output_token_number": max_tokens})

    rephrased_question = question_chain.invoke(input_kw, config=config)

    log.debug("Rephrased question: %s", rephrased_question)

    return rephrased_question


def build_rephrase_chain(llm, initial_question_template,
                      chat_condenser_prompt):

    rephrase_with_out_history_chain = initial_question_template | llm | strip_string_parser
    rephrase_with_history_chain = chat_condenser_prompt | llm | strip_string_parser

    def wrapper(input_kwargs, chat_history, config ):
        if chat_history:
            template = chat_condenser_prompt
            human_msg = template[-1].format(question=input_kwargs['question'], chat_history='')
            assert isinstance(human_msg, HumanMessage), 'the message should be a human template, check prompt template'

            human_msg_tokens = llm.get_num_tokens_from_messages([human_msg])

            model_name = llm.default.metadata.get('model', None) or llm.default.model_name
            token_calculator = get_token_calculator_for_model(model_name)
            chat_history = reduce_chats_below_limit(chat_history, token_calculator,
                                                    llm.default.metadata["chat_history_limit_calculator"](
                                                        human_msg_tokens))

            input_kwargs["chat_history"] = convert_to_string(chat_history)

            return rephrase_with_history_chain.invoke(input_kwargs, config=config)
        else:
            return rephrase_with_out_history_chain.invoke(input_kwargs, config=config)

    return wrapper


def build_title_chain(llm, title_prompt, **kwargs):

    title_generation_chain = title_prompt | llm | strip_string_parser
    return title_generation_chain


def detect_query_language(rephrased, chat_history, lang_detector, question=None):
    supported_langs = list(supported_templates.keys())
    lang = lang_detector.predict(rephrased)
    query_length = len(rephrased.split(' '))
    if lang in supported_langs and query_length >= MIN_QUERY_LENGTH:
        return lang
    # Handle case when user input is short
    if query_length < MIN_QUERY_LENGTH and chat_history:
        prev_message = chat_history[-1][1]
        match = re.search(r"\[AIMessage: \d+\](.*)", prev_message)
        extracted_message = match.group(1) if match else ''
        if extracted_message:
            # Detect Language on First row of previous AI Message
            lang = lang_detector.predict( extracted_message.split('\n')[0])

    elif question is not None:
        lang = lang_detector.predict(question) # as second option to detect language on user question

    if lang not in supported_langs:
        raise NotSupportedException(lang, code="unsupported_language")
    return lang


class GeneralChain(Chain):
    llm: Runnable
    title_chain: Union[Runnable, None] = None
    response_template: BasePromptTemplate
    max_tokens: int = MAX_TOKENS_CHAT
    max_tokens_title: int = MAX_TOKENS_TITLE
    output_key: str = "results"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        # allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.
        :meta private:
        """
        return ["question", "subject", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.
        :meta private:
        """
        return [self.output_key]

    def _call(self,
              inputs: Dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None,
              ) -> Dict[str, Any]:
        log.info("in %s", self._chain_type)
        chat_subject = None
        model_name = self.llm.metadata.get('model', None) or self.llm.model_name
        token_calculator = get_token_calculator_for_model(model_name)

        if not inputs.get("chat_subject") and self.title_chain:
            chat_subject = self.title_chain.invoke(
                inputs['question'],
                config=RunnableConfig(callbacks=run_manager.get_child(),
                                      metadata={"stage_name": 'title_generation_chain'},
                                      configurable={"output_token_number": self.max_tokens_title}
                                      )
            )
        question_chain = self.response_template | self.llm

        # log.info("building rephrased_question lang chain")
        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'Direct_chat_chain'},
                                #metadata={'stream': True}
                                configurable={"output_token_number": self.max_tokens}
                                )

        human_msg = self.response_template[-1].format(**inputs)
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'

        human_msg_tokens = self.llm.get_num_tokens_from_messages([human_msg])

        chat_history = reduce_chats_below_limit(
            inputs['chat_history'], token_calculator,
            self.llm.metadata["chat_history_limit_calculator"](human_msg_tokens))

        log.info("running response with %s", type(self.llm))
        try:
            inputs.update({"chat_history": chat_history})
            inputs.update(self.llm.metadata.get('model_profile_for_generation'))
            inputs.update({'user_name':inputs['user_profile'].get('name','Not Provided'),
                       'user_designation':inputs['user_profile'].get('designation','Not Provided'),
                       'user_department':inputs['user_profile'].get('department','Not Provided'),
                       'user_personalization':inputs['user_profile'].get('user_personalization','Not Provided'),
                       'user_company':inputs['user_profile'].get('company_name','Not Provided'),
                       'user_specified_behaviour':inputs['user_profile'].get('response_customization',''),
                       'user_country':inputs['user_profile'].get('country','Not Provided')})
            inputs.update({'current_date': date.today()})

            if self.llm.metadata.get('stream_able', True):
                response = ''.join(
                    [chunk.content for chunk in question_chain.stream(inputs, config=config)])

            else:
                AIMessage = question_chain.invoke(inputs, config=config)
                response = AIMessage.content

        except Exception as e:
            log.error(e, exc_info=True)
            raise e

        return {
            self.output_key:
                {
                    "chat_subject": chat_subject,
                    "response": response,
                    "files": []
                    # 'answer': json.dumps({'response': response, "files": doc_source})})
                }

        }

    @property
    def _chain_type(self) -> str:
        return "DirectCallChain"


class SubjectChain(Chain):
    """
    """

    llm: Runnable
    chat_llm: Runnable
    retriever: BaseRetriever = Field()
    initial_question_template: BasePromptTemplate
    chat_condenser_prompt: BasePromptTemplate
    title_chain: Runnable
    # document_formatting_prompt: BasePromptTemplate
    extractor_prompt: BasePromptTemplate

    subject: str = "a specific field"
    output_key: str = "results"  #: :meta private:
    # lang: str = "English"
    max_tokens_title: int = MAX_TOKENS_ROUTER
    max_tokens_rephrase: int = MAX_TOKENS_REPHRASE
    max_tokens: int = MAX_TOKENS_CHAT

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        # allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.
        :meta private:
        """
        return ["question", "chat_history", "subject"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.
        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        model_name = self.chat_llm.metadata.get('model', None) or self.chat_llm.model_name
        token_calculator = get_token_calculator_for_model(model_name)

        log.info("in SubjectChain")
        reranker=inputs["reranker"]
        reranker_bool=inputs["reranker_bool"]

        tenant_id=inputs["tenant_id"]
        query = inputs["question"]
        lang = inputs["lang"] or detect_query_language(query, inputs['chat_history'], inputs["lang_detector"])
        filter=inputs['filter']
        k=inputs['k']
        lang_expanded = supported_templates[lang]

        log.info('question: %s ,language: %s', query, lang)

        subject = inputs.get("subject") or self.subject
        rephrased_query = rephrase_question({"subject": subject,
            "question": query,
            "lang": lang_expanded,
            'user_name':inputs['user_profile'].get('name','Not Provided'),
            'user_designation':inputs['user_profile'].get('designation','Not Provided'),
            'user_department':inputs['user_profile'].get('department','Not Provided'),
            'user_personalization':inputs['user_profile'].get('user_personalization','Not Provided'),
            'user_company':inputs['user_profile'].get('company_name','Not Provided'),
            'user_specified_behaviour':inputs['user_profile'].get('response_customization',''),
            'user_country':inputs['user_profile'].get('country','Not Provided')
            },
            inputs['chat_history'],
            self.llm,
            self.initial_question_template,
            self.chat_condenser_prompt,
            run_manager, self.max_tokens_rephrase) if inputs.get("query_repharsing_needed", True)==True else query

        chat_subject = None
        if not inputs.get("chat_subject"):
            chat_subject = self.title_chain.invoke(
                query,
                config=RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'title_generation_chain'},
                                configurable={"output_token_number": self.max_tokens_title}
                                )
            )

        human_msg = self.extractor_prompt[-1].format(context='', **inputs)
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'
        human_msg_tokens = self.chat_llm.get_num_tokens_from_messages([human_msg])

        document_combine_chain = FormatRetrieverDocumentChain(
            # document_formatting_prompt=self.document_formatting_prompt,
            retriever=self.retriever,
            verbose=False,
            token_calculator=token_calculator
        )

        log.info("Getting relevant documents")
        document_output_dict = document_combine_chain(
            {"query": rephrased_query, 
             "lang":lang,
             "max_tokens": self.chat_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
             "tenant_id": tenant_id,
             "filter":filter,
             "k":k,
             "reranker_bool":reranker_bool,
             "reranker":reranker},
            callbacks=run_manager.get_child() if run_manager else None,
        )

        formatted_docs = document_output_dict["context"]
        doc_source = document_output_dict["source"]
        log.info(f"Format retriver returned: {doc_source}")
        answer_generation_chain = self.extractor_prompt | self.chat_llm

        chat_history = reduce_chats_below_limit(
            inputs['chat_history'], token_calculator,
            self.chat_llm.metadata["chat_history_limit_calculator"](human_msg_tokens))

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'answer_generation_chain'},
                                configurable={"output_token_number": self.max_tokens}
                                )

        log.info("running answer chain with")
        inputs.update({
            "subject": subject,
            "question": query,
            "context": formatted_docs,
            "chat_history": chat_history,
            "lang": lang_expanded}
        )
        inputs.update(self.chat_llm.metadata.get('model_profile_for_generation'))
        inputs.update({'user_name':inputs['user_profile'].get('name','Not Provided'),
                       'user_designation':inputs['user_profile'].get('designation','Not Provided'),
                       'user_department':inputs['user_profile'].get('department','Not Provided'),
                       'user_personalization':inputs['user_profile'].get('user_personalization','Not Provided'),
                       'user_company':inputs['user_profile'].get('company_name','Not Provided'),
                       'user_specified_behaviour':inputs['user_profile'].get('response_customization',''),
                       'user_country':inputs['user_profile'].get('country','Not Provided')})
        inputs.update({'current_date': date.today()})

        if self.chat_llm.metadata.get('stream_able', True):
            response = ''.join([chunk.content for chunk in answer_generation_chain.stream(
                inputs, config=config)])

        else:
            AIMessage = answer_generation_chain.invoke(inputs, config=config)
            response = AIMessage.content
        # docx = len(set([x['name'] for x in sources]))
        # pages = len([x['page'] for x in sources])
        return {
            self.output_key:
                {
                    "chat_subject": chat_subject,
                    "response": response,
                    "files": doc_source,
                    "rephrase": rephrased_query,
                    "sources_count": [len(set([x['name'] for x in doc_source])), len([x['page'] for x in doc_source])]
                    # 'answer': json.dumps({'response': response, "files": doc_source})})
                }

        }

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        chat_subject = None
        if not inputs.get("chat_subject"):
            log.info("building title generation lang chain")
            chat_subject = await self.title_chain.ainvoke(
                inputs["input"],
                config=RunnableConfig(callbacks=run_manager.get_child(),
                                      metadata={"stage_name": 'title_generation_chain'},
                                      configurable={"output_token_number": self.max_tokens_title}
                                      )
            )
        document_combine_chain = FormatRetrieverDocumentChain(
            # document_formatting_prompt=self.document_formatting_prompt,
            retriever=self.retriever,
            verbose=False,
        )

        log.info("Getting relevant documents ")
        document_output_dict = document_combine_chain(
            {"query": inputs["input"]},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        #
        formatted_docs = document_output_dict["context"]
        doc_source = document_output_dict["source"]
        answer_generation_chain = self.extractor_prompt | self.chat_llm
        log.info("running answer chain")
        response = await answer_generation_chain.arun(
            {
                "subject": self.subject,
                "question": inputs["input"],
                "context": formatted_docs,
            },
            callbacks=run_manager.get_child() if run_manager else None,
            metadata={"stage_name": 'answer_generation_chain'},
        )
        return {
            self.output_key:
                {
                    "chat_subject": chat_subject,
                    "response": response,
                    "files": doc_source
                    # 'answer': json.dumps({'response': response, "files": doc_source})})
                }

        }

    @property
    def _chain_type(self) -> str:
        return "SubjectChain"

    def update_retriever(self, retriever):
        """
        method to update db file retrieve object
        :param retriever: new retriever of type :langchain.schema.BaseRetriever
        :return:
        """
        self.retriever = retriever


def build_attachment_router_chain(
        router_prompt:BasePromptTemplate,
        router_llm:Runnable[Input, Output],
        routes:Dict[Any,Runnable[Input, Output]],
        default_route):

    def route(info:Dict[Any,Any]):
        route_name = info["topic"].lower()
        if route_name in routes:
            log.debug("route `%s` selected", route_name)
            return routes[route_name]
        else:
            log.warning(
            'route `%s` is not present in given routes `%s`, selecting default route',route_name, routes.keys())
            return default_route

    model_name = router_llm.metadata.get('model', None) or router_llm.default.model_name
    token_calculator = get_token_calculator_for_model(model_name)

    def marginalize_chats(x):
        human_msg = router_prompt[-1].format(**x)
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'
        human_msg_tokens = router_llm.get_num_tokens_from_messages([human_msg])

        chat_history_limit = router_llm.metadata["chat_history_limit_calculator"](human_msg_tokens)
        return reduce_chats_below_limit(x['chat_history'], token_calculator, chat_history_limit)

    chain = ({"question":lambda x:  x.get("question"),
              # "chat_history": lambda x: reduce_chats_below_limit(x['chat_history'], token_calculator, chat_history_limit),
              "chat_history": marginalize_chats,
          "context": lambda x:  x.get("context")
          }|router_prompt
             | router_llm
             | StrOutputParser()
             )

    return  RunnablePassthrough.assign(topic=chain) | RunnableLambda(route)


class WebSearchChain(Chain):
    common_llm: Runnable  # may be required for rephrasing
    large_context_llm: Runnable
    web_retriever: BaseRetriever = Field()
    # document_formatting_prompt: BasePromptTemplate
    initial_question_template: BasePromptTemplate
    chat_condenser_prompt: BasePromptTemplate
    title_chain: Runnable
    response_template: BasePromptTemplate
    max_tokens: int = MAX_TOKENS_CHAT
    max_tokens_title: int = MAX_TOKENS_TITLE
    output_key: str = "results"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        # allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.
        :meta private:
        """
        # return ["input", "subject", "chat_history", "chat_subject"]
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.
        :meta private:
        """
        return [self.output_key]

    def _call(self,
              inputs: Dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None,
              ) -> Dict[str, Any]:
        log.info("in %s", self._chain_type)
        chat_subject = None
        model_name = self.large_context_llm.metadata.get('model', None) or self.large_context_llm.model_name
        token_calculator = get_token_calculator_for_model(model_name)
        question = inputs['question']

        if not inputs.get("chat_subject"):
            chat_subject = self.title_chain.invoke(
                question,
                config=RunnableConfig(callbacks=run_manager.get_child(),
                                      metadata={"stage_name": 'title_generation_chain'},
                                      configurable={"output_token_number": self.max_tokens_title}
                                      )
            )
        lang = inputs['lang'] or detect_query_language(question, inputs['chat_history'], inputs["lang_detector"])
        rephrased_query = rephrase_question({"subject": "subject",
            "question": question,
            "lang":  lang,
            'user_name':inputs['user_profile'].get('name','Not Provided'),
            'user_designation':inputs['user_profile'].get('designation','Not Provided'),
            'user_department':inputs['user_profile'].get('department','Not Provided'),
            'user_personalization':inputs['user_profile'].get('user_personalization','Not Provided'),
            'user_company':inputs['user_profile'].get('company_name','Not Provided'),
            'user_specified_behaviour':inputs['user_profile'].get('response_customization',''),
            'user_country':inputs['user_profile'].get('country','Not Provided')
            },
            inputs['chat_history'],
            self.common_llm, self.initial_question_template,
            self.chat_condenser_prompt, run_manager, 50) if inputs.get("query_repharsing_needed", True)==True else question

        # need to get documents from the web
        human_msg: HumanMessage = self.response_template[-1].format(context='', **inputs)
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'

        human_msg_tokens = self.large_context_llm.get_num_tokens_from_messages([human_msg])

        web_document_combine_retrieve_chain = FormatRetrieverWebChain(
            retriever=self.web_retriever,
            # max_tokens=self.large_context_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
            token_calculator=token_calculator,
            verbose = False,
        )
        formatted_document: dict = web_document_combine_retrieve_chain(
            {"query": rephrased_query,
             "max_tokens": self.large_context_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
             "lang": lang},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        context, doc_source = formatted_document["context"], formatted_document["source"]

        question_chain = self.response_template | self.large_context_llm

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'web_search_chain'},
                                # metadata={'stream': True}
                                configurable={"output_token_number": self.max_tokens}
                                )
        chat_history = reduce_chats_below_limit(
            inputs['chat_history'], token_calculator,
            self.large_context_llm.metadata["chat_history_limit_calculator"](human_msg_tokens))
        log.info("running response with %s", type(self.large_context_llm))
        try:
            inputs.update(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "context": context,
                }
            )
            inputs.update(self.large_context_llm.metadata.get('model_profile_for_generation'))
            inputs.update({'user_name':inputs['user_profile'].get('name','Not Provided'),
                       'user_designation':inputs['user_profile'].get('designation','Not Provided'),
                       'user_department':inputs['user_profile'].get('department','Not Provided'),
                       'user_personalization':inputs['user_profile'].get('user_personalization','Not Provided'),
                       'user_company':inputs['user_profile'].get('company_name','Not Provided'),
                       'user_specified_behaviour':inputs['user_profile'].get('response_customization',''),
                       'user_country':inputs['user_profile'].get('country','Not Provided')})
            inputs.update({'current_date': date.today()})

            if self.large_context_llm.metadata.get('stream_able', True):
                response = ''.join([chunk.content for chunk in question_chain.stream(inputs, config=config)])
            else:
                AIMessage = question_chain.invoke(inputs, config=config)
                response = AIMessage.content

        except Exception as e:
            log.error(e, exc_info=True)
            raise e

        return {
            self.output_key:
                {
                    "chat_subject": chat_subject,
                    "response": response,
                    "files": doc_source,
                    "rephrase": rephrased_query,
                    "sources_count": [len(set([x['name'] for x in doc_source])),
                                      len([x['page'] for x in doc_source])]
                }
            }

    @property
    def _chain_type(self) -> str:
        return "WebSearchChain"
