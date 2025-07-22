"""
 This is the main execution file. it implements ask_question and add_document to add pdf file
 and ask question from those files using openapi large-language-models
"""
import os
import re
import json
import time
import asyncio
import logging
from logging.config import dictConfig
from typing import List, Dict, Optional

import pycountry
import concurrent.futures as cf
from threading import Lock
from pydantic.json import pydantic_encoder
from pydantic import BaseModel, Field, Json
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_core.runnables import RunnableConfig
from .config import supported_images, map_to_local, backend_azure_map
from .templates.util import get_templates
from .vector_store import DocumentPGVector, WebPGStore, DocumentPGKeyword, DocumentPGHybrid
from .vector_store.web import web_retriever
from .external_endpoints.embedding import get_embeddings
from .external_endpoints.reranker import get_reranker
from .custom_models.fast import FAST_LANG_MODEL
from .templates.default import user_specified_behaviour_instruction
from .models.models import OutBEMsgMultiTool,OutBEMultiTool, OutFEChatMetaMsg,OutFEErrorMsg

from .callback import (
    get_logger,
    TokenCalculationCallback,
    AzureWebPubSubCallback,
    AsyncAzureWebPubSubCallback,
    ResponseTimeCalculation,
    AzureWebPubSubCallbackMultiTool,
)
from .util import (
    parse_chat,
    parse_param,
    gtp_bot,
    gtp_direct,
    gtp_web_search,
    gpt_attached_docs,
    parse_output,
    parse_tool_output,
    timeit,
    clean_text_for_common_string,
    regex_filter,
    build_doc_inspector_graph,
    language_translator,
    chat_bot,
    process_user_profile,
    get_templates,
)
from .schema import Schema
from .exceptions import get_error_message
from .logger import configuration

from .tools.multi_tool_tools import EnterpriseTool, WebTool, AttachmentTool, PlannerTool, GreetingTool


dictConfig(configuration)

log = logging.getLogger("doclogger")
log.disabled = False
posgres_conn = os.getenv("POSTGRES_CONN_STRING")
thread_pool_size = int(os.getenv("MY_APP_THREADS",10))

# pylint: disable=W0718,C0103,R0914
#TODO: Remove inheritance relation from here.


DOC_ANALYZER_LLM_NAME = os.getenv("LLM_DOC_ANALYZER", "gpt-4o").lower()

RETRIEVED_DOCS = int(os.getenv("RETRIEVED_DOCS", "20"))
CHAT_DOC_RETRIEVER_TYPE = os.getenv("DOC_RETRIEVER_TYPE_CHAT", "hybrid")
ADV_SEARCH_DOC_RETRIEVER_TYPE = os.getenv("DOC_RETRIEVER_TYPE_SEARCH", "hybrid")

WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", 'us')

RERANKER_BOOL = True if os.getenv("RERANKER_BOOL", 'True').lower()=='true' else False
RERANKER_URL = os.getenv("RERANKER_URL")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME","BAAI/bge-reranker-v2-m3")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", None)

DOC_RETRIEVER_MAP = {"semantic":0,
"keyword":1,
"hybrid":2}

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "gpt-4o")


class Interactive:
    """
    main Class implements the functionality
    takes no arguments
    """

    executor = cf.ThreadPoolExecutor(max_workers=thread_pool_size)
    loop = asyncio.get_event_loop()

    def __init__(self, **kwargs):
        embedding_object = get_embeddings(os.getenv("OPENAI_API_TYPE", "openai"))
        store_db_readers = self._create_file_store_readers(embedding_object)
        self.store_db_reader_chat = store_db_readers[DOC_RETRIEVER_MAP[CHAT_DOC_RETRIEVER_TYPE]]
        self.store_db_reader_adv_search = store_db_readers[DOC_RETRIEVER_MAP[ADV_SEARCH_DOC_RETRIEVER_TYPE]]
        self.reranker = get_reranker(RERANKER_URL, RERANKER_MODEL_NAME, top_n=RETRIEVED_DOCS, token=RERANKER_API_KEY, **kwargs)

        # self.store_db_reader_keyword = store_db_readers[KEYWORD_DB_READER]
        # self.store_db_reader_hybrid = store_db_readers[HYBRID_DB_READER]

        self.web_store = self._create_web_store(posgres_conn)
        self.web_retriever = web_retriever(vectorstore=self.web_store, search_region=WEB_SEARCH_REGION,
                                           lang_detector=FAST_LANG_MODEL)
        self.lang_detector = FAST_LANG_MODEL

    def translate(self, user_input, target_language='en'):
        detected_lang = self.lang_detector.predict(user_input)
        if detected_lang == target_language:
            return user_input

        target_language = pycountry.languages.get(alpha_2=target_language).name
        result = language_translator(user_input, target_language, TRANSLATION_MODEL_NAME)
        return result

    def ask_assistant(self, name, user_input: str,
                      description, instructions, chat_history = "", chat_type = 'E',
                      files=None,  model_name=None,  **kwargs):


        additional_kwargs = {
            "assistant_name": clean_text_for_common_string(name),
            "assistant_description": clean_text_for_common_string(description),
            "assistant_instructions":clean_text_for_common_string(instructions),
        }
        # templates_to_use = ["REPHRASE", "ASSISTANT", "DOC_FORMATTING", "TITLE"]
        templates_to_use = ["REPHRASE", "ASSISTANT", "TITLE"]

        return self.ask_question(user_input, chat_history=chat_history, chat_type=chat_type, files=files,
                                 model_name=model_name, subject='a specific field',
                                 templates_to_use=templates_to_use,
                                 additional_kwargs=additional_kwargs,
                                 **kwargs)

    def ask_agent_question(self, user_input: List[Dict], files=None, lang: str = None,
                           model_name=None, **kwargs):
        """ used for DOC-ANALYST

        :param user_input:
        :param files:
        :param lang:
        :param model_name:
        :param kwargs:
        :return:
        """
        use_model = backend_azure_map[model_name] or DOC_ANALYZER_LLM_NAME
        tenant_id = kwargs['extras']['tenant_id']

        extras = kwargs.get("extras", {})
        user_profile = kwargs['extras'].get('user_profile',{})
        reranker_bool = kwargs['extras'].get('reranker_bool', RERANKER_BOOL)
        
        analysis_id = extras.get("analysis_id", time.time())
        log.info("making file filter")
        filters = Schema.make_filter(files)
        log.info(filters)
        log.info("getting retriever with the filter applied")
        ret = self.store_db_reader_chat.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                           "score_threshold_semantic":0.12,
                           "score_threshold_keyword": 0.0})

        call_back = kwargs.get("cb", lambda *x: print(x))
        doc_inspector = build_doc_inspector_graph(ret, use_model)
        token_callback = TokenCalculationCallback()
        chain_callbacks = [token_callback]
        if os.getenv("LOGDIR", None):
            log.info("creating callback for debugging chat")
            chain_callbacks += [get_logger(analysis_id, "doc_inspector")]
        log.info("running chain with question ")

        run_config = RunnableConfig(callbacks=chain_callbacks,
                                    metadata={"Run type": 'document inspector'})
        log.debug("getting language param %s", lang)
        for u_q in user_input:
            query_question = json.loads(Schema.decrypt(u_q))
            try:
                result = doc_inspector.graph.invoke({
                    'question': query_question['question'],
                    "max_revisions": 2,
                    "revision_number": 1,
                    "answer_score": 0,
                    "k": RETRIEVED_DOCS,
                    "filter": filters,
                    "subject_chain_kwargs": {
                        "lang_detector": self.lang_detector,
                        "lang": lang,
                        "tenant_id": tenant_id,
                        "user_profile": user_profile,
                        "chat_subject": 'doc analysis',
                        "reranker": self.reranker,
                        "reranker_bool": reranker_bool,
                        "chat_history":[],
                        "subject": "a specific field",
                        "k": RETRIEVED_DOCS,
                        "filter": filters,
                    }

                }, config=run_config)
                response, cited_sources = regex_filter(result['answer'], result['files'])
                answer = {"response": response, 'files': cited_sources}
                confidence = result['answer_score']
                status_details=None

            except Exception as e:
                log.error(e, exc_info=True)
                status_details = get_error_message(e)
                confidence = None
                answer = {"response": '', 'files': []}

            query_question.update({"answer": Schema.encrypt(json.dumps(answer))})
            payload = {"question": query_question,
                        "confidence":confidence,
                        "extra": extras,
                        "status": False if status_details else True,
                        "status_details":status_details,
                        "token_usage": list(token_callback.tokens_usage.values()),
                    }
            call_back(payload)
        return True

    def ask_question(self, user_input, chat_history="", chat_type='E', files=None, lang=None,
                     model_name=None, templates_to_use=None,
                     additional_kwargs=None,
                     subject='a specific field',
                     **kwargs):
        """

        :param additional_kwargs:
        :param templates_to_use:
        :param chat_type:
        :param user_input: user query
        :param chat_history: chat history object default ''
        :param files: list for files (name) to be searched from default None
        :param model_name: string from [Mistral-Large, llama-3-70B-Instruct, gpt-3.5-turbo-16k,  gpt-4-32k
        :param subject: {lang:language, chat_id, chat_subject, qid, ck: encryption key , cb:callback}
        :param kwargs: {lang:language, chat_id, chat_subject, qid, ck: encryption key , cb:callback}
        :return:
        """
        # decrypt encryption key using standard methods
        user_input = Schema.decrypt(user_input)
        user_profile = kwargs['extras'].get('user_profile', {})
        if behaviour_instructions := user_profile.get('response_customization'):
            user_profile['response_customization'] = user_specified_behaviour_instruction.format(
                user_specified_behaviour=behaviour_instructions)
            # llm_profile = pass
        web_search = kwargs['extras'].get('web_search_enabled', False)
        tenant_id = kwargs['extras']['tenant_id']
        reranker_bool = kwargs['extras'].get('reranker_bool', RERANKER_BOOL)
        log.info("Parsing chat history")
        chat_history, _ = parse_chat(chat_history)
        log.info("Parsing params")
        _, model_params, chat_subject = parse_param(kwargs)

        log.info("cleaning inputs")
        user_input = clean_text_for_common_string(user_input)
        chat_subject = clean_text_for_common_string(chat_subject)
        subject = clean_text_for_common_string(subject)

        images = kwargs.get('images', {})

        log.info("making file filter")
        filters = Schema.make_filter(files)
        log.info(filters)
        log.info("getting retriever with the filter applied")
        model_name = map_to_local(model_name)
        ret = self.store_db_reader_chat.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold_semantic": 0.12, "score_threshold_keyword": 0.0})
        # model_params["lang"] = lang
        # "k": RETRIEVED_DOCS, "filter": filters,

        additional_img_kwargs = {}

        if chat_type.upper() == "C":
            if web_search:
                log.info("calling web_search chain builder function")
                bot = gtp_web_search(self.web_retriever, model_name=model_name, model_params=model_params,
                                     templates_to_use=templates_to_use,
                                     )
            else:
                if images:
                    image_count = 0
                    for k, v in images.items():
                        img_data = v['file_data']
                        image_type = v['original_format']
                        if image_type and image_type not in supported_images:
                            return False
                        additional_img_kwargs.update({
                            f"image_type{image_count or ''}": image_type,
                            f"image_data{image_count or ''}": img_data,
                        })
                        image_count += 1

                    templates_to_use = ["REPHRASE", "SUBJECT", "TITLE", "IMAGE"]

                    if files:
                        # templates_to_use = ["REPHRASE", "SUBJECT", "TITLE", "PARAMETRIC", "IMAGE"]
                        log.info("calling custom chain builder function")
                        bot = gpt_attached_docs(ret, model_name=model_name, model_params=model_params,
                                                templates_to_use=templates_to_use, image_count=image_count)
                        additional_img_kwargs["context"] = Schema.extract_context(files)
                    else:
                        bot = gtp_direct(model_name=model_name, model_params=model_params,
                                         templates_to_use=templates_to_use, image_count=image_count)

                elif files:
                    # calling the RAG for files
                    log.info("calling custom chain builder function")
                    bot = gpt_attached_docs(ret, model_name=model_name, model_params=model_params,
                                            templates_to_use=templates_to_use)
                    additional_img_kwargs["context"] = Schema.extract_context(files)

                else:
                    # check if history contains any image source
                    is_image_in_history = False
                    for res in chat_history:
                        if isinstance(res, tuple):
                            for r in res:
                                if isinstance(r, list):
                                    is_image_in_history = True

                    if is_image_in_history:
                        templates_to_use = ["REPHRASE", "SUBJECT", "TITLE", "PARAMETRIC"]

                    log.info("calling direct chain builder function")
                    bot = gtp_direct(model_name=model_name, model_params=model_params,
                                     templates_to_use=templates_to_use)

        elif chat_type.upper() == "E":
            log.info("calling custom chain builder function")
            bot = gtp_bot(ret, model_name=model_name, model_params=model_params,
                          templates_to_use=templates_to_use,
                          )
        else:
            raise ValueError(f'{chat_type} is not supported')

        bot_kwargs = {
            "question": user_input,
            "subject": subject,
            "chat_history": chat_history,
            "lang": lang,
            "filter": filters,
            "user_profile": user_profile,
            "lang_detector": self.lang_detector,
            "reranker": self.reranker,
            "reranker_bool": reranker_bool,
            "k": RETRIEVED_DOCS,
            "chat_subject": chat_subject if chat_subject != "New Chat" else None,
            "tenant_id": tenant_id,

        }
        if additional_kwargs:
            assert isinstance(additional_kwargs, dict), ValueError("only dictionary type of arguments are supported")
            bot_kwargs.update(additional_kwargs)

        if additional_img_kwargs:
            assert isinstance(additional_img_kwargs, dict), ValueError(
                "only dictionary type of arguments are supported")
            bot_kwargs.update(additional_img_kwargs)

        log.debug("ask question called")
        if os.getenv("ASYNC_MODE", None) == "True":
            log.debug("using async mode to answer question")
            try:
                async_stream_callback = AsyncAzureWebPubSubCallback(
                    encrypt_func=Schema.encrypt,
                    encryption_key=Schema.decrypt(kwargs.get("ck")),
                    **kwargs,
                )
                chain_callbacks = [TokenCalculationCallback(), async_stream_callback, ResponseTimeCalculation()]

                future = self.loop.create_task(
                    self.async_execute_bot(
                        bot,
                        bot_kwargs,
                        chain_callbacks,
                        model_params,
                        *list(kwargs.values()),
                    ),
                )
                future.add_done_callback(
                    lambda x: self.loop.create_task(
                        self.async_handle_ask_question_response(x.result(), **kwargs)
                    )
                )
                log.info("Successfully submitted task to thread")
                return True, None
            except Exception as e:
                if isinstance(async_stream_callback, AsyncAzureWebPubSubCallback):
                    async_stream_callback.close()
                log.error(e, exc_info=True)
                self.loop.close()
                return False, e
        else:
            log.debug("using sync mode to answer question")
            stream_callback = None
            try:
                chat_type = "W" if web_search else chat_type

                # TODO: we donot need to make this connection for each request. This takes alot of time. The only \
                # issue here is the ck param that we get from be with each request. Discussion required on this with be.
                stream_callback = AzureWebPubSubCallback(
                    chat_type=chat_type,
                    encrypt_func=Schema.encrypt,
                    encryption_key=Schema.decrypt(kwargs.get("ck")),
                    **kwargs,
                )
                chain_callbacks = [TokenCalculationCallback(), stream_callback, ResponseTimeCalculation()]
                # TODO: make a container object for callbacks so that grace-ful closing can be offloaded to that object
                future = self.executor.submit(
                    self.sync_execute_bot,
                    bot,
                    bot_kwargs,
                    chain_callbacks,
                    # model_params,
                    **kwargs,
                )
                future.add_done_callback(
                    lambda x: log.info("**** sync_ask_question returned %s", str(x.result()))
                )
                future.add_done_callback(
                    lambda x: self.handle_ask_question_response(x, chain_callbacks, chat_type=chat_type, **kwargs)
                )
                # future.add_done_callback(
                #     lambda _: stream_callback.close()
                # )
                log.info("Successfully submitted task to thread")
                return True, None
            except Exception as e:
                if isinstance(stream_callback, AzureWebPubSubCallback):
                    stream_callback.close()
                log.error(e, exc_info=True)
                return False, e


    def ask_assistant_multi_tool(self, name, user_input: str,
                      description, instructions, chat_history = "", tools_list=None,
                      enterprise_files=None, attachment_files=None,  model_name=None,  **kwargs):


        additional_kwargs = {
            "assistant_name": clean_text_for_common_string(name),
            "assistant_description": clean_text_for_common_string(description),
            "assistant_instructions":clean_text_for_common_string(instructions),
        }
        # templates_to_use = ["REPHRASE", "ASSISTANT", "DOC_FORMATTING", "TITLE"]
        # templates_to_use = ["REPHRASE", "ASSISTANT", "TITLE"]

        return self.ask_question_multi_tool(user_input, chat_history=chat_history, tools_list=tools_list,
                                 enterprise_files=enterprise_files,attachment_files=attachment_files,
                                 model_name=model_name, subject='a specific field',
                                 assistant=True,
                                 additional_kwargs=additional_kwargs,
                                 **kwargs)

    def ask_question_multi_tool(self, user_input, chat_history="", tools_list=None, enterprise_files=None, attachment_files=None,lang=None,
                     model_name=None,
                     additional_kwargs=None,
                     subject='a specific field',
                     **kwargs):
        """

        :param user_input: user query
        :param chat_history: chat history object default ''
        :param tools_list: list of tools that are available should be in `web_search, enterprise, attached_documents, plan_and_execute`
        :param enterprise_files: list for files (name) to be searched from default None
        :param attachment_files: list for files (name) to be searched from attachment_files None
        :param model_name: string from [Mistral-Large, llama-3-70B-Instruct, gpt-3.5-turbo-16k,  gpt-4-32k
        :param additional_kwargs:
        :param subject: {lang:language, chat_id, chat_subject, qid, ck: encryption key , cb:callback}
        :param kwargs: {lang:language, chat_id, chat_subject, qid, ck: encryption key , cb:callback}
        :return:
        """
        # decrypt encryption key using standard methods
        user_input = Schema.decrypt(user_input)
        user_profile = kwargs['extras'].get('user_profile',{})
        if behaviour_instructions:=user_profile.get('response_customization'):
            user_profile['response_customization']=user_specified_behaviour_instruction.format(user_specified_behaviour=behaviour_instructions)

        # web_search = kwargs['extras'].get('web_search_enabled', False)
        tenant_id = kwargs['extras']['tenant_id']
        reranker_bool = kwargs['extras'].get('reranker_bool', RERANKER_BOOL)
        log.info("Parsing chat history")
        chat_history, is_image_in_history = parse_chat(chat_history)
        log.info("Parsing params")
        _, model_params, chat_subject = parse_param(kwargs)

        log.info("cleaning inputs")
        user_input = clean_text_for_common_string(user_input)
        chat_subject = clean_text_for_common_string(chat_subject)
        subject = clean_text_for_common_string(subject)

        assistant = kwargs.get('assistant', False)
        images = kwargs.get('images', {})

        log.info("making file filter")
        enterpise_filters = Schema.make_filter(enterprise_files)
        attachment_filters = Schema.make_filter(attachment_files)
        log.info(enterpise_filters)
        log.info(attachment_filters)
        log.info("getting retriever with the filter applied")
        model_name = map_to_local(model_name)
        ret = self.store_db_reader_chat.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold_semantic": 0.12, "score_threshold_keyword": 0.0})
        # model_params["lang"] = lang
        # additional_img_kwargs = {}
        profile_kwargs = process_user_profile(user_profile)

        available_tools = []

        if tools_list is None:
            tools_list = []
        greet_tool = GreetingTool(template=get_templates(
            ["PARAMETRIC"], "default", model_name)["response_template"])
        available_tools.append(greet_tool)

        if 'web_search' in tools_list:
            if assistant:
                template = get_templates(["ASSISTANT"], 'default', model_name)["web_search"]
            else:
                template = get_templates(["WEBCHAT"], 'default', model_name)["web_search"]

            web_tool = WebTool(
                self.web_retriever,
                chat_model=model_name,
                lang_detector=self.lang_detector,
                tenant_id=tenant_id,
                k=RETRIEVED_DOCS,
                template=template
            )
            available_tools.append(web_tool)

        if 'attached_documents' in tools_list and len(attachment_files)>0:

            if assistant:
                template = get_templates(["ASSISTANT"], 'default', model_name)["enterprise_knowledge"]
            else:
                template = get_templates(["SUBJECT"], 'default', model_name)["extractor_prompt"]

            attachment_tool = AttachmentTool(
                ret,
                reranker=self.reranker if reranker_bool else None,
                reranker_bool=reranker_bool,
                tenant_id=tenant_id,
                chat_model=model_name,
                lang_detector=FAST_LANG_MODEL,
                file_context=Schema.extract_context(attachment_files),
                filter=attachment_filters,
                k=RETRIEVED_DOCS,
                template=template
                )
            available_tools.append(attachment_tool)

        if 'enterprise' in tools_list:
            if assistant:
                template = get_templates(["ASSISTANT"], 'default', model_name)["enterprise_knowledge"]
            else:
                template = get_templates(["SUBJECT"], 'default', model_name)["extractor_prompt"]
            adv_retriever = self.store_db_reader_adv_search.as_retriever(
                search_type="advanced_search_with_threshold",
                search_kwargs={"score_threshold_semantic": 0.1, "score_threshold_keyword": 0.0})
            adv_retrieved_docs = adv_retriever.invoke(
                input={
                    "query": user_input,
                    "k": 15,
                    "filter": enterpise_filters,
                    "lang": lang,
                    "tenant_id": tenant_id
                }
            )
            rag_tool = EnterpriseTool(
                ret,
                enterprise_doc_summaries='\n\t- '.join([f.doc_summary for f in adv_retrieved_docs]),
                reranker=self.reranker if reranker_bool else None,
                reranker_bool=reranker_bool,
                tenant_id=tenant_id,
                chat_model=model_name,
                lang_detector=FAST_LANG_MODEL,
                filter=enterpise_filters,
                k=RETRIEVED_DOCS,
                template=template
                )
            available_tools.append(rag_tool)

        if 'plan_and_execute' in tools_list:
            planner_tool_args={
                "web_retriever": self.web_retriever,
                "doc_retriever": ret,
                "advanced_search_retriever": self.store_db_reader_adv_search.as_retriever(
                    search_type="advanced_search_with_threshold",
                    search_kwargs={"score_threshold_semantic": 0.12, "score_threshold_keyword": 0.0}),
                "chat_model": model_name,
                "lang_detector": FAST_LANG_MODEL,
                "reranker": self.reranker if reranker_bool else None,
                "reranker_bool": reranker_bool,
                "tenant_id":tenant_id,
                "attached_file_context":Schema.extract_context(attachment_files),
                "doc_filter":enterpise_filters,
                "attached_files_filter":attachment_filters,
                "tools_list": tools_list
                }
            planner_tool = PlannerTool(**planner_tool_args)
            available_tools.append(planner_tool)


        bot = chat_bot(available_tools, model_name, model_params=model_params,
                       images=images, image_in_history=is_image_in_history, assistant=assistant)

        if additional_kwargs:
            assert isinstance(additional_kwargs, dict), ValueError("only dictionary type of arguments are supported")
            profile_kwargs.update(additional_kwargs)

        bot_kwargs = {
            "question": user_input,
            "subject": subject,
            "chat_history": chat_history,
            "lang": lang,
            "chat_subject": chat_subject,  # if chat_subject != "New Chat" else None,
            "tenant_id": tenant_id,
            "tool_name":None,
            "profile_kwargs": profile_kwargs,
        }

        log.debug("ask question called")
        if os.getenv("ASYNC_MODE", None) == "True":
            log.debug("using async mode to answer question")
            try:
                async_stream_callback = AsyncAzureWebPubSubCallback(
                    encrypt_func=Schema.encrypt,
                    encryption_key=Schema.decrypt(kwargs.get("ck")),
                    **kwargs,
                )
                chain_callbacks = [TokenCalculationCallback(), async_stream_callback, ResponseTimeCalculation()]

                future = self.loop.create_task(
                    self.async_execute_bot(
                        bot,
                        bot_kwargs,
                        chain_callbacks,
                        model_params,
                        *list(kwargs.values()),
                    ),
                )
                future.add_done_callback(
                    lambda x: self.loop.create_task(
                        self.async_handle_ask_question_response(x.result(), **kwargs)
                    )
                )
                log.info("Successfully submitted task to thread")
                return True, None
            except Exception as e:
                if isinstance(async_stream_callback, AsyncAzureWebPubSubCallback):
                    async_stream_callback.close()
                log.error(e, exc_info=True)
                self.loop.close()
                return False, e
        else:
            log.debug("using sync mode to answer question")
            stream_callback = None
            try:
                # chat_type = "W" if web_search else chat_type
                chat_type = 'tool'
                # TODO: we donot need to make this connection for each request. This takes alot of time. The only \
                # issue here is the ck param that we get from be with each request. Discussion required on this with be.
                stream_callback = AzureWebPubSubCallbackMultiTool(
                    chat_type=chat_type,
                    encrypt_func=Schema.encrypt,
                    encryption_key=Schema.decrypt(kwargs.get("ck")),
                    **kwargs,
                )
                chain_callbacks = [TokenCalculationCallback(), stream_callback, ResponseTimeCalculation()]
                # TODO: make a container object for callbacks so that grace-ful closing can be offloaded to that object
                future = self.executor.submit(
                    self.sync_execute_bot,
                    bot,
                    bot_kwargs,
                    chain_callbacks,
                    # model_params,
                    **kwargs,
                )
                future.add_done_callback(
                    lambda x: log.info("**** sync_ask_question returned %s", str(x.result()))
                )
                future.add_done_callback(
                    lambda x: self.handle_ask_multitool_response(x, chain_callbacks, chat_type=chat_type, **kwargs)
                )
                log.info("Successfully submitted task to thread")
                return True, None
            except Exception as e:
                if isinstance(stream_callback, AzureWebPubSubCallbackMultiTool):
                    stream_callback.close()
                log.error(e, exc_info=True)
                return False, e

    @timeit
    def sync_execute_bot(self, bot, bot_kwargs, chain_callbacks, **kwargs):
        """

        :param bot: string question
        :param bot_kwargs: dictionary of dictionary with structure as
                   {id:     {'q': {'id': 309, 'message': q, ...},
                            'r': {'id': 310, 'message': r, ...},
                            ...
                           }
                   }
        :param chain_callbacks: List of Dict of following structure
                [ {'remote_id: 'your-file-id'}
                    .
                    .
                ]

        :param kwargs:
                chat_id,
                params = parma_map {refer to Schema in util.py},
                lang define language for answer string
        :return:
            String (encrypted if encryption is enabled)
        """
        chat_id = kwargs.get("chat_id", -1)

        if os.getenv("LOGDIR", None):
            log.info("creating callback for debugging chat")
            chain_callbacks += [get_logger(chat_id)]
        log.info("running chain with question ")

        config = RunnableConfig(callbacks=chain_callbacks,
                                metadata={"stage_name": 'main_chain'},
                           )
        response = bot.invoke(
            bot_kwargs,
            config=config
            # callbacks=chain_callbacks,
        )
        return response

    async def async_execute_bot(self, bot, bot_kwargs, chain_callbacks, *args):
        """

        :param bot: Chain object with retriever configured
        :param bot_kwargs: additional arguments for chain,  question
        :param chain_callbacks: additional arguments for chain,  question
        :param args:
        :return:
        """
        # TODO: make async consistent with sync
        chat_id = args[0]
        # try:
        if os.getenv("LOGDIR", None):
            log.info("creating callback for debugging chat")
            chain_callbacks += [get_logger(chat_id)]
        log.info("running chain with question ")

        response = await bot.arun(
            bot_kwargs,
            callbacks=chain_callbacks,
        )
        return response, chain_callbacks

    def handle_ask_question_response(self, result, callback_responses, **kwargs):

        """
        function to handel response from ask_question sync function, transforms output and sends to
        backend
        :param result: tuple from ask_question
        :param callback_responses: additional arguments {cb:callback, lan: language, rid: response_id,
        chat_id
        :return: None
        """
        cb = kwargs.get("cb", log.debug)
        chat_subject = None
        tokens = []
        message = {}

        try:
            if result.exception():
                log.error(result.exception())
                raise result.exception()
            chat_response = result.result()

            event = 'chat_meta'
            message, tokens = parse_output(chat_response, callback_responses[0].tokens_usage)
            chat_subject = message.get("chat_subject")

            fe_message = message
            status = True
            status_details = None
        except Exception as e:
            log.error(e, exc_info=True)
            event = 'error'
            status = False
            status_details = get_error_message(e)
            fe_message = get_error_message(e)

        #TODO: make a dataclass out of payloads
        payload = {
            "rid": kwargs.get("rid"),
            "time_stamps": callback_responses[2].time_log,  #TODO: this is bad implementation
            "tokens_usage": tokens,
            "msg": message,
            "new_subject": chat_subject,
            "chat_id": kwargs.get("chat_id"),
            'status': status,
            'status_details': status_details
        }
        pubsub_callback: AzureWebPubSubCallback = callback_responses[1]
        fe_message['type'] = kwargs["chat_type"]
        pubsub_callback.send_msg({"msg": fe_message,
                                  "new_subject": chat_subject,
                                  }, event=event)
        pubsub_callback.close()
        log.debug(payload)

        payload["extras"] = kwargs.get("extras")
        payload["msg"] = Schema.encrypt(json.dumps(payload.get("msg")))
        cb(payload)

    async def async_handle_ask_question_response(self, function_response, **kwargs):
        """
        function to handel response from ask_question asynchronously  function, transforms
         output and sends to backend
        :param function_response: tuple from async_ask_question
        :param kwargs: additional arguments {cb:callback, lan: language, rid: response_id,
        chat_id
        :return: None
        """
        cb = kwargs.get("cb", log.debug)
        log.debug("val of function response is:\n %s", function_response)
        chat_response, callback_responses = function_response

        chat_subject = None
        tokens = []
        try:
            event = 'chat_meta'

            message, tokens = parse_output(
                *function_response
            )

            status = True
            status_details = None

        except Exception as e:
            log.error(e, exc_info=True)
            event = 'error'
            status = False
            status_details = get_error_message(e)
            message = {}

        payload = {
            "rid": kwargs.get("rid"),
            "tokens_usage": tokens,
            "time_stamps": callback_responses[2].time_log,
            "msg": message,
            "new_subject": chat_subject,
            "chat_id": kwargs.get("chat_id"),
            'status': status,
            'status_details': status_details
        }

        # message, chat_subject = post_process(result, **kwargs)
        await self.async_stream_callback.send_msg({"msg": message,
                                                   "new_subject": chat_subject,
                                                   }, event=event)

        payload["extras"] = kwargs.get("extras")
        payload["msg"] = Schema.encrypt(json.dumps(payload.get("msg")))
        cb(payload)
        # if self.async_stream_callback is not None:
        #     await self.async_stream_callback.close()
        # self.time_stamps = None


    def handle_ask_multitool_response(self, result, callback_responses, **kwargs):

        """
        function to handel response from ask_question sync function, transforms output and sends to
        backend
        :param result: tuple from ask_question
        :param callback_responses: additional arguments {cb:callback, lan: language, rid: response_id,
        chat_id
        :return: None
        """
        cb = kwargs.get("cb", log.debug)
        msg = OutBEMsgMultiTool(task_id='0')
        payload: OutBEMultiTool = OutBEMultiTool(rid=kwargs.get("rid"), chat_id= kwargs.get("chat_id"), msg=msg)

        try:
            event = 'chat_meta'
            if result.exception():
                log.error(result.exception())
                raise result.exception()
            tool_response = result.result()

            message = parse_tool_output(tool_response)
            fe_message = OutFEChatMetaMsg(
                task_id='0', step_name='final', **message)

            payload.new_subject = tool_response.get("chat_subject")
            payload.tokens_usage=list(callback_responses[0].tokens_usage.values())
            # msg: OutBEMsgMultiTool=OutBEMsgMultiTool(task_id='0',**message)
            msg.files=message['files']
            msg.response=message['response']
            msg.tool_name=message['tool_name']
            msg.additional_steps=message.get('additional_steps', {})
            msg.search_phrase = message['search_phrase']

        except Exception as e:
            event = 'error'
            log.error(e, exc_info=True)
            payload.status = False
            payload.status_details = get_error_message(e)
            fe_message = OutFEErrorMsg(**get_error_message(e))

        payload.time_stamps = callback_responses[2].time_log
        pubsub_callback: AzureWebPubSubCallbackMultiTool = callback_responses[1]

        pubsub_callback.send_msg(fe_message.model_dump(), event=event)

        pubsub_callback.close()
        log.debug(payload)
        payload.extras = kwargs.get("extras")
        payload.msg = Schema.encrypt(payload.msg.model_dump_json())
        cb(payload.model_dump())

    @staticmethod
    def _create_file_store_readers(embedding_object):
        """
        build connection to vector store
        :param persist_directory: str directory path
        :return:
        """
        embeddings = embedding_object

        # if posgres_conn:
        log.info("Connecting PgVector dB")

        db_store_vector = DocumentPGVector(
            posgres_conn,
            embedding_function=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
            pool_size=thread_pool_size,
            pool_pre_ping=True,
            pool_recycle=3600,  # this line might not be needed
            connect_args={
                "keepalives": 1,
                # "keepalives_idle": 30,
                # "keepalives_interval": 10,
                # "keepalives_count": 5,
            }
        )
        db_store_keyword = DocumentPGKeyword(
            posgres_conn,
            pool_size=thread_pool_size,
            pool_pre_ping=True,
            pool_recycle=3600,  # this line might not be needed
            connect_args={
                "keepalives": 1,
                # "keepalives_idle": 30,
                # "keepalives_interval": 10,
                # "keepalives_count": 5,
            }
        )
        db_store_hybrid = DocumentPGHybrid(
            db_store_vector,
            db_store_keyword
        )
        return db_store_vector, db_store_keyword, db_store_hybrid

    @staticmethod
    def _create_web_store(conn_string):
        return WebPGStore(
            conn_string,
            pool_size=thread_pool_size,
            pool_pre_ping=True,
            pool_recycle=3600,  # this line might not be needed
            connect_args={
                "keepalives": 1,
                # "keepalives_idle": 30,
                # "keepalives_interval": 10,
                # "keepalives_count": 5,
            }
        )

    def advance_search(self, user_input, files=None, token_callback=None,**kwargs):
        user_input = Schema.decrypt(user_input)
        tenant_id = kwargs['extras']['tenant_id']
        lang = kwargs['extras'].get('lang', None)

        log.info("cleaning inputs")
        user_input = clean_text_for_common_string(user_input)
        log.info("making file filter")
        filters = Schema.make_filter(files)
        log.info(filters)
        lang = lang or self.lang_detector.predict(user_input)
        # callbacks = [TokenCalculationCallback()]
        token_callback = token_callback or TokenCalculationCallback()
        retriever = self.store_db_reader_adv_search.as_retriever(
            search_type="advanced_search_with_threshold",
            search_kwargs={"score_threshold_semantic": 0.5, "score_threshold_keyword": 0.0})
        config = RunnableConfig(callbacks=[token_callback],
                                metadata={"stage_name": 'adv_search_retriever',
                                          "retriever": retriever},
                           )
        retrieved_docs = retriever.invoke(
            input={
                "query": user_input,
                "k": 15,
                "filter": filters,
                "lang":lang,
                "tenant_id":tenant_id
            }, config=config
            )
        return json.dumps({'retrieved_docs': retrieved_docs,'token_usage': list(token_callback.model_tokens.values())}, default=pydantic_encoder)
        # return json.dumps(retrieved_docs, default=pydantic_encoder)


obj = None
lock = Lock()

def get_interactive():
    global obj
    with lock:
        if obj is None:
            obj = Interactive()
    return obj
