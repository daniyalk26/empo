
import os
import re
import asyncio
import logging
from logging.config import dictConfig
from typing import List, Dict,  Any
from threading import Lock

from langchain_core.runnables import RunnableConfig


from .schema import Schema
from .config import supported_templates
from .file_loader import load_reader
from .agents import SummaryGenerator
from .external_endpoints.azure_llm import get_llm_common, get_chat_model
from .meta_function import detect_language, get_timestamp
from .templates.util import get_templates
from .tokenizer_cal import _get_encoding_model

# from .vector_store.base import async_concurrent_requests
from .preprocessing import enrich_split_document, get_supported_extension, document_processor
from .operations import RecordOperation

from .callback import (
    get_logger,
    TokenCalculationCallback,
)
from .exceptions import RetryableException, ScannedDocException, get_error_message, NotSupportedException
from .logger import configuration
from .util import timeit


dictConfig(configuration)

log = logging.getLogger("doclogger")
log.disabled = False
posgres_conn = os.getenv("POSTGRES_CONN_STRING")
thread_pool_size = int(os.getenv("MY_APP_THREADS",10))

LLM_ADD_META = os.getenv('LLM_ADD_META', "gpt-4o")
MAX_TOKENS_SUMMARY = os.getenv('MAX_TOKENS_SUMMARY', 500)
MAX_SUMMARY_REVISIONS = os.getenv('MAX_REVISIONS', 2)
PDF_PARSER_TYPE = os.getenv("PARSER_TYPE")

# pylint: disable=W0718,C0103,R0914


class DocumentProcessor(RecordOperation):

    def __init__(self, **kwargs):
        super().__init__()
        self.blob_reader = load_reader(os.getenv("READ_FROM", "azure_blob"))

    def add_document(self, files, initial_text_size=2000, **kwargs):
        """
        main call function to add document (async or sync)to database after sone processing
        :param files: list of dict containing 'remote_id', "id", "name", secret:for file decryption,
        :param num_pages: pages to be used for title/summary extraction
        :param kwargs: {index, cb:callback from front_end}
        :return:
        """
        log.debug("add document called")
        if os.getenv("ASYNC_MODE", None) == "True":
            log.debug("using async mode to add document")
            raise ValueError("depricated")
            # self.async_add_document(files, initial_text_size=initial_text_size, **kwargs)

        else:
            log.debug("using sync mode to add document")
            self.sync_add_document(files, initial_text_size=initial_text_size, **kwargs)


    def sync_add_document(self, files: List[dict], initial_text_size=2000, **kwargs):
        """
        adds new documents to db backend for querying from
        :param files: list of dict as explained in ask_question
        :param initial_text_size: int, Number of text size in document for summary generation
        :param kwargs: additional arguments see add_document
        :return:
        """
        # if self.reader is None:
        #     self.reader = load_reader(os.getenv("READ_FROM", "azure_blob"))

        index = kwargs.get("index", None)
        log.info(" 'Processing ' for files %s", len(files))

        be_call_back = kwargs.get("cb", lambda *x: print(x))
        for item in files:
            status_details = {"code": 'success', "description": "successfully processed"}
            token_callback = TokenCalculationCallback()
            if self.store_db_writer.if_exists(item):
                status_details["description"] = "already processed"
                item['status'] = True
                item["status_details_update"] =  status_details
                item["token_usage"] = list(token_callback.tokens_usage.values())  #
                item["lang"] = None
                item["metadata"] = None
                item["retry_flag"] = False
                continue

            f_path = item[Schema.file_tag]
            tenant_id = item[Schema.tenant_id_tag]
            encry_key = item.pop("secret", None)

            chain_callbacks = [token_callback]
            if os.getenv("LOGDIR", None):
                log.info("creating callback for debugging chat")
                chain_callbacks += [get_logger(item[Schema.id_tag], "doc_processing")]

            parser_type = item.get(Schema.processing_step, 0)
            # item["extras"] = kwargs.get("extras")
            try:
                log.debug("getting file with %s", item)
                log.info("loading Initial data from file %s", f_path)
                file_type = get_supported_extension(item[Schema.format_tag])
                initial_info, data, meta_info_be = load_initial_and_data(
                    self.blob_reader, f_path, file_type, initial_text_size, encry_key, parser_type=parser_type
                )
                log.info("parsing document %s", f_path)
                ai_extracted_meta = MetaExtractor.meta_extractor_ai(
                    initial_info, file_type=file_type, callbacks=chain_callbacks
                )  # language and summary
                # add additional meta for front end source representation
                fe_extracted_meta = MetaExtractor.meta_extractor_fe(item)
                meta_info_be.update(ai_extracted_meta)

                ai_extracted_meta.update(
                    fe_extracted_meta
                )
                texts, meta_data = enrich_split_document(
                    data, file_type, ai_extracted_meta
                )
                lang = ai_extracted_meta[Schema.lang_param]
                try:
                    self.store_db_writer.add_texts(
                        texts=texts, metadatas=meta_data, tenant_id=tenant_id, callbacks=chain_callbacks,
                        extracted_meta=ai_extracted_meta
                    )
                except Exception as r_e:
                    log.error(str(r_e), exc_info=True)
                    raise RetryableException(r_e)
                item["lang"] = lang
                item["status"] = True
                item["metadata"] = meta_info_be
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = False

            except ScannedDocException as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = True if PDF_PARSER_TYPE!='simple' else False
                item["next_step"] = parser_type + 1
                log.error(" Exception occurred", exc_info=True)

            except RetryableException as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = True
                item["next_step"] = parser_type
                log.error(" Exception occurred", exc_info=True)

            except Exception as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = False
                log.error(" Exception occurred", exc_info=True)

            item["status_details_update"] = status_details

            # finally:
            #     if self.reader is not None:
            #         self.reader.close()
        if os.getenv('DLM_JOB_MODE', "").lower() != 'azure-queue-storage':
            be_call_back(files, index)  # , kwargs.get('extras'))

        log.info("Completed for files %s", str(files))

    async def async_add_document(self, files, initial_text_size=2000, **kwargs):
        """
        adds new documents to db backend asynchronously for querying from
        :param files: list of dict as explained in ask_question
        :param initial_text_size: int, Number of words in document for summary generation
        :param kwargs: additional arguments see add_document
        :return:
        """
        call_back = kwargs.get("cb", lambda *x: print(x))
        index = kwargs.get("index", None)

        # if self.reader is None:
        #     self.reader = load_reader(os.getenv("READ_FROM", "azure_blob"))

        files_to_process_concurrently = asyncio.Semaphore(3)

        # for each file follow the following steps
        # @async_concurrent_requests(1)
        async def aadd_single_document(item: dict, semaphore):
            """
            adds single document to db
            :param file: dict -> 'remote_id', "id", "name", secret:for file decryption,
            :param semaphore: no of concurrent docs to process
            :return:
            """
            await semaphore.acquire()
            tokens_usage = {}
            # file["extras"] = file.get("extras")
            token_callback = TokenCalculationCallback()
            chain_callbacks = [token_callback]
            parser_type = item.get(Schema.processing_step, 0)

            try:
                file_type = get_supported_extension(item[Schema.format_tag])
                f_path = item[Schema.file_tag]
                tenant_id = item[Schema.tenant_id_tag]
                encry_key = item.pop("secret", None)

                if os.getenv("LOGDIR", None):
                    log.info("creating callback for debugging chat")
                    chain_callbacks += [get_logger(item[Schema.id_tag], "doc_processing")]

                print("|", end="")
                # gather data from pdf/other file
                # initial_info, data, meta_info_be = load_initial_and_data(
                #     self.blob_reader, f_path, file_type, initial_text_size, encry_key, parser_type=parser_type
                # )
                initial_pages, docs, meta_info_be = await aload_initial_and_data(
                    self.blob_reader,
                    f_path,
                    file_type,
                    initial_text_size,
                    encry_key,
                    parser_type=parser_type
                )
                log.debug("getting file with %s", item)

                # detect language, generate_summary
                ai_extracted_meta = await MetaExtractor.ameta_extractor_ai(
                    initial_pages, callbacks=chain_callbacks
                )  # language
                fe_extracted_meta = await  MetaExtractor.ameta_extractor_fe(item)
                ai_extracted_meta.update(
                    fe_extracted_meta
                )
                meta_info_be.update(ai_extracted_meta)

                texts, meta_data = enrich_split_document(
                    docs, file_type, ai_extracted_meta
                )
                lang = ai_extracted_meta[Schema.lang_param]

                try:
                    await self.store_db_writer.aadd_texts(
                        texts=texts, metadatas=meta_data,  tenant_id=tenant_id, callbacks=chain_callbacks,
                        extracted_meta=ai_extracted_meta

                    )
                except Exception as r_e:
                    log.error(str(r_e), exc_info=True)
                    raise RetryableException(r_e)

                item["status"] = True
                item["metadata"] = meta_info_be
                item["token_usage"] = list(tokens_usage.values())  # tokens_usage

                # return file
                item["lang"] = lang
                item["status"] = True
                item["metadata"] = meta_info_be
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = False

            except ScannedDocException as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = True if PDF_PARSER_TYPE!='simple' else False
                item["next_step"] = parser_type + 1
                log.error(" Exception occurred", exc_info=True)

            except RetryableException as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = True
                item["next_step"] = parser_type
                log.error(" Exception occurred", exc_info=True)

            except Exception as e:
                item["status"] = False
                item["metadata"] = None
                status_details = get_error_message(e)
                item["token_usage"] = list(token_callback.tokens_usage.values())  # tokens_usage
                item["retry_flag"] = False
                log.error(" Exception occurred", exc_info=True)
            finally:
                semaphore.release()
            return item

        # tasks = [aload_initial_and_data(self.reader, file, num_pages) for file in files]
        # files = await asyncio.gather(*tasks, return_exceptions=True)
        # client_context = self.blob_reader.get_async_container_client()
        process_result = await asyncio.gather(
            *[aadd_single_document(f, files_to_process_concurrently) for f in files],
            return_exceptions=True,
        )
        # loop = asyncio.get_event_loop()
        # process_result = loop.run_until_complete(files_to_process_task)
        log.debug("testing concurrency")
        if os.getenv('DLM_JOB_MODE', "").lower() != 'azure-queue-storage':
            call_back(process_result, index)  # , kwargs.get('extras'))
        # client_context.close()


def document_word_count(document):
    return len(
        [word for word in re.sub(r'\s+', ' ', document.page_content).split(' ') if len(word)])


@timeit
def load_initial_and_data(
        reader, file_path, file_type, initial_text_size=2000, encry_key=None, **kwargs
):
    # with reader(file_path, encry_key) as f:
    stream = None
    try:
        log.info("Reading %s with %s", file_path, str(reader))
        stream = reader(file_path, encry_key)
        # document_processor = extension_specific_processor[file_type]
        initial_text, docs, meta_info = document_processor(
            stream, file_path, file_type, initial_text_size, Schema.name_tag, **kwargs
        )
        meta_info['document_word_count'] = sum([document_word_count(doc) for doc in docs])
        return initial_text, docs, meta_info
    except ScannedDocException as e:
        log.error(e, exc_info=True)
        raise ScannedDocException(e)
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
    finally:
        if stream is not None:
            stream.close()
            log.info("Closed stream handle")


async def aload_initial_and_data(
        reader, file_path, file_type, initial_num_pages, encry_key=None, **kwargs
):
    stream = None

    try:
        stream = await reader.__acall__(file_path, encry_key)
        initial_text, docs, meta_info = document_processor(
            stream, file_path, file_type, initial_num_pages, Schema.name_tag, **kwargs
        )
        return initial_text, docs, meta_info
    except ScannedDocException as e:
        log.error(e, exc_info=True)
        raise ScannedDocException(e)
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
    finally:
        if stream is not None:
            stream.close()
            log.info("Closed stream handle")


def aggregate_token(tokens: Dict, additional_tokens: Dict):
    model_name = additional_tokens["model_name"]
    if model_name in tokens.keys():
        for k_y in additional_tokens.keys() - ["model_name"]:
            tokens[model_name][k_y] += additional_tokens[k_y]
    else:
        tokens[model_name] = additional_tokens


def get_num_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def generate_summary_agent(first_page, **kwargs):
    log.info("Generating Summary")

    summary_llm = get_llm_common(temperature=0.0, max_tokens=800)
    callbacks = kwargs.get("callbacks", None)
    file_type=kwargs["file_type"]
    config = RunnableConfig(callbacks=callbacks,
                            metadata={"stage_name": 'Summary_chain'},
                            configurable={"output_token_number": MAX_TOKENS_SUMMARY})

    try:
        doc_lang = kwargs['lang']
        
        if doc_lang not in supported_templates:
            raise NotSupportedException(doc_lang, code="unsupported_language")

        if file_type == 'HTML' and len(first_page.strip())==0:
            return 'NO TEXT FOUND'

        doc_lang = supported_templates[doc_lang]
        templates = get_templates(["SUMMARIZATION"], 'default', model_name=summary_llm.metadata['model'])
        desc_gen = SummaryGenerator(model=summary_llm,
                                    **templates,
                                    tokenizer=_get_encoding_model,
                                    max_tokens=MAX_TOKENS_SUMMARY,
                                    max_revisions=MAX_SUMMARY_REVISIONS,
                                    )
        summary = desc_gen.generate_summary(first_page, doc_lang, config)

        if summary == 'NO CONTEXT AVAILABLE':
            raise RetryableException('Unable to Generate summary')
        log.info("Summary generated leaving summary generation")
        return summary

    except Exception as e:
        # log.error("e")
        log.error("Exception occurred", exc_info=True)
        raise e


async def agenerate_summary(first_page, **kwargs):
    log.info("Generating Summary async")
    summary_llm = get_llm_common(temperature=0.0, max_tokens=800)

    callbacks = kwargs.get("callbacks", None)

    file_type=kwargs["file_type"]
    config = RunnableConfig(callbacks=callbacks,
                            metadata={"stage_name": 'Summary_chain'},
                            configurable={"output_token_number": MAX_TOKENS_SUMMARY})
    try:
        doc_lang = kwargs['lang']
        if doc_lang not in supported_templates:
            raise NotSupportedException(doc_lang, code="unsupported_language")

        if file_type == 'HTML' and len(first_page.strip())==0:
            return 'NO TEXT FOUND'
        doc_lang = supported_templates[doc_lang]
        templates = get_templates(["SUMMARIZATION"], 'default', model_name=summary_llm.metadata['model'])
        desc_gen = SummaryGenerator(model=summary_llm,
                                    **templates,
                                    tokenizer=_get_encoding_model,
                                    max_tokens=MAX_TOKENS_SUMMARY,
                                    max_revisions=MAX_SUMMARY_REVISIONS,
                                    )
        summary = await desc_gen.agenerate_summary(first_page, doc_lang, config)
        if summary == 'NO CONTEXT AVAILABLE':
            raise ValueError('Unable to Generate summary')
        log.info("Summary generated leaving summary generation")
        return summary

    except Exception as e:
        # log.error(f"{e}")
        log.error("Exception occurred", exc_info=True)
        raise e


def additional_ai_meta_functional_call(text, **kwargs) -> Dict[Any, Any]:
    from .templates.default import AdditionalAiMeta
    try:
        file_type=kwargs["file_type"]
        if len(text.strip())==0:
            return dict(AdditionalAiMeta())
        callbacks = kwargs.get("callbacks", None)
        config = RunnableConfig(callbacks=callbacks,
                                metadata={"stage_name": 'additional_ai_meta_extraction'})

        model = get_chat_model(LLM_ADD_META, stream=False)
        structured_llm = model.default.with_structured_output(AdditionalAiMeta)
        log.debug("keyword and meta extracted")
        result = structured_llm.invoke(text, config=config)
        return dict(result)
    except Exception as e:
        log.error(e, exc_info=True)
        raise e

def detect_language_file_type(file_text, *args, **kwargs):
    try:
        file_type = kwargs['file_type']
        return detect_language(file_text, *args, **kwargs)
    except Exception as e:
        if file_type=='HTML':
            return 'en'
        raise e
class MetaExtractor:
    meta_extract = [
        ("lang", detect_language_file_type),  #TODO: ensure to use fasttext for language detection.
        ("document_summary", generate_summary_agent),
        ('processing_time', get_timestamp),
        ('keyword', additional_ai_meta_functional_call)
    ]
    meta_extract_async = [
        ("lang", detect_language_file_type),
        ("document_summary", agenerate_summary),
        ('processing_time', get_timestamp),
        ('keyword', additional_ai_meta_functional_call)
    ]
    # meta_extraction_seq = ["lang", "document_summary"]


    @staticmethod
    def meta_extractor_ai(first_pages, **kwargs):
        meta = {}
        for key, func in MetaExtractor.meta_extract:
            log.info("extracting %s in meta_extractor", key)
            value = func(first_pages, **meta, **kwargs)
            if isinstance(value, dict):
                meta.update(value)
                continue
            meta[key] = value
        return meta

    @staticmethod
    async def ameta_extractor_ai(first_pages, **kwargs):
        meta = {}
        for key, func in MetaExtractor.meta_extract_async:
            log.info("extracting %s in meta_extractor", key)
            if asyncio.iscoroutinefunction(func):
                meta[key] = await func(first_pages, **meta, **kwargs)
            else:
                meta[key] = func(first_pages, **meta, **kwargs)
        return meta

    @staticmethod
    def meta_extractor_fe(be_file_dict):
        meta = {}
        for key in Schema.front_end_required_keys:
            value = be_file_dict[key]
            meta[key] = value
        return meta

    @staticmethod
    async def ameta_extractor_fe(be_file_dict):
        meta = {}
        for key in Schema.front_end_required_keys:
            value = be_file_dict[key]
            meta[key] = value
        return meta


writer_obj = None
lock = Lock()


def get_processor():
    global writer_obj
    with lock:
        if writer_obj is None:
            writer_obj = DocumentProcessor()
    return writer_obj
