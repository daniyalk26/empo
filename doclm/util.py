"""
 this module implements utility functions and MetaExtractor class which contains the
  basic functionalities required in other modules
"""
import re
import os
import json
import time
from datetime import date

import logging
from functools import wraps

from typing import Tuple, Dict, Any, List
from langchain_core.utils.function_calling import convert_to_openai_tool

from .schema import Schema
from .llm_chains.qa_chain import (GeneralChain, SubjectChain,
                                  WebSearchChain, build_title_chain, build_attachment_router_chain)
from .llm_chains.qa_chain import build_rephrase_chain

from .templates.util import get_templates, get_router_template
from .templates.util import attachment_router_prompt

from .external_endpoints.azure_llm import get_chat_model, get_llm_common, get_llm_router
from .exceptions import NotSupportedException
from .config import supported_templates
from .agents.doc_inspector import DocInspector
from .agents.multi_tool_agent import GraphExecutor

log = logging.getLogger("doclogger")
log.disabled = False

__all__ = [
    "parse_param",
    "parse_chat",
    "gtp_bot",
    "gtp_direct",
    "gtp_web_search",
    "gpt_attached_docs",
    "parse_output",
    "clean_text_for_prompts",
    "language_translator",
]

MAX_TOKENS_ROUTER = os.getenv('MAX_TOKENS_ROUTER', 800)
MAX_TOKENS_REPHRASE = os.getenv('MAX_TOKENS_REPHRASE', 500)
MAX_TOKENS_CHAT = os.getenv('MAX_TOKENS_CHAT', 3000)
MAX_TOKENS_TITLE = os.getenv('MAX_TOKENS_TITLE', 100)

DEFAULT_PARAMS = {"temperature": 0.0,
                  "max_tokens": 3000}


# pylint: disable=R0914,C0103
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        log.info("Function %s Took %s seconds", func.__name__, total_time)
        return result

    return timeit_wrapper


def clean_text_for_common_string(input_str: Any):
    if isinstance(input_str, str):
        input_str = re.sub('}', '}}', re.sub('{', '{{', input_str))
    return input_str


def clean_text_for_prompts(input_str: Any):
    if isinstance(input_str, str):
        input_str = clean_text_for_common_string(input_str)
        input_str = re.sub(r"\[.+?:\s*(\d+.*?)\]", "", input_str)
    return input_str


def parse_output(response: dict, tokens: Dict):
    log.info("parse_output running")

    log.debug("trying to json parse response")
    result = response['results']  #json.loads(response)
    log.debug("filtering files from sources")

    answer, filtered_files = regex_filter(result['response'], result.get('files', []))
    result["response"] = answer
    result["files"] = filtered_files

    log.debug("returning results")
    return result, list(tokens.values())


def parse_tool_output(response: dict):
    log.info("parse_output running")
    log.debug("trying to json parse response")
    result = response['results']
    result["tool_name"] = response["tool_name"]
    result['response'], result["files"] = regex_filter(result['response'], result.get('files', []))

    if response["tool_name"] in ['plan_and_execute']:
        result["additional_steps"] = response['message_board'][-1].additional_kwargs
    log.debug("returning results")
    return result


def regex_filter(response, files) -> Tuple[Any, list]:
    # reg_exp = reg_expression(lang)
    # sources_tuple = []
    try:
        #reomove AIMessage from teh chat response
        response = re.sub(r"\[(?i)AI.+?:\s*(\d+.*?)\]", "", response)

        reg_exp_get_content_between_braces = r'\[.*?:(.*?\d+)\]'
        reg_exp_get_doc_nums = r'\d+'
        intermediate_result = [r for r in re.findall(reg_exp_get_content_between_braces, response)]
        # reg_exp = r'\[.+?:\s*(\d+.*?)\]'
        # reg_exp = r'\[.+?:\s*(.+?)\]'
        # reg_exp = r'\[.+?:\s*"(.+?)"\s*,.+?\s*(\d+).*?\]'
        cited_docs = []
        # log.debug(response)
        for doc in intermediate_result:
            cited_docs.extend(re.findall(reg_exp_get_doc_nums, doc))
        log.debug('extracted documents %s', str(cited_docs))

        log.debug("document parse in regex")
        log.debug(cited_docs)
        cited_docs = set(cited_docs)

        cited_sources = []
        for file in files:
            if file[Schema.citation_tag] in cited_docs:  # and file not in results:
                cited_sources.append(file)
            else:
                log.debug(f'{file[Schema.citation_tag]} not in {cited_docs}')
        return response, cited_sources

    except Exception as e:
        log.error('Can not apply regex', exc_info=True)
        return response, []


# @retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(10))


def parse_param(kwargs):
    log.info("Parsing input parameters")
    ch_id = kwargs["chat_id"]
    chat_subject = kwargs["chat_subject"]
    param = {}
    for p_r in kwargs["params"]:
        key = Schema.parma_map[p_r["name"]]
        val = p_r["value"]
        d_type = p_r["data_type"]
        assert d_type in Schema.conv_ops, f"Unknown type data_type in param {d_type}"
        param[key] = Schema.conv_ops[d_type](val)
    log.info("Parameters parsed")

    return ch_id, param, chat_subject


# pylint: disable=C0103

def parse_chat(chat_history, chat_buffer_size=3):
    chat_tuple = []
    is_image_in_history = False
    log.info("Parsing chat parameters")
    supported_formats = ['jpeg', 'jpg', 'png', 'svg']

    if not isinstance(chat_history, dict):
        return chat_tuple, is_image_in_history

    try:
        for idx, k_y in enumerate(sorted(chat_history.keys()), start=1):
            chat_x = chat_history[k_y]
            q = Schema.decrypt(chat_x["q"]["message"])
            resp_decrypt = Schema.decrypt(chat_x["r"]["message"]) or ''
            try:
                resp = json.loads(resp_decrypt) or {'response': ''}
                planner_result = ''
                if add_steps := resp.get('additional_steps', {}):
                    task_info = add_steps.get('task_info')
                    task_result = add_steps.get('task_result')
                    for t in task_info:
                        planner_result += f"task_{t}: {task_info[t]['description']} \n"
                        planner_result +=  f"task_{t} response: {task_result[t]['content']['response']} \n"

                    r = planner_result
                else:
                    r = resp.get('response')
            except json.JSONDecodeError as j_e:
                r = resp_decrypt
            if r == '':
                continue
            q = clean_text_for_prompts(q)
            r = clean_text_for_prompts(r)
            # check the image files
            res_img = chat_x['q']['resources']

            image_human_msg = []

            for img in res_img:
                if img['original_format'] in supported_formats and 'file_data' in img and 'original_format' in img:
                    image_human_msg.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img['original_format']};base64,{img['file_data']}"},
                    })
                #todoK: query should not be place for evert image

            if len(image_human_msg) > 0:
                image_human_msg.append({
                    "type": "text",
                    "text": f"[HumanMessage: {idx}]{q}"
                })
                chat_tuple.append(("human", image_human_msg))
                is_image_in_history = True
            else:
                chat_tuple.append(("human", f"[HumanMessage: {idx}]" + q))

            # chat_tuple.append(("human", f"[HumanMessage: {idx}]" + q))
            chat_tuple.append(("ai", f"[AIMessage: {idx}]" + r))
            # chat_tuple.append((q, r))
            # chat_string += f"Question: {q} \n Answer: {r} \n"

    except TypeError:
        log.warning("could not parse Chat parameters completely")

    return chat_tuple, is_image_in_history


def build_subject_chain(retriever, common_llm, large_context_llm,
                        templates_to_use: List[str], **kwargs):
    assert isinstance(templates_to_use, list), ValueError("got empty templates")
    use_templates = get_templates(templates_to_use, 'default', kwargs['model_name'])

    return SubjectChain(
        llm=common_llm,
        chat_llm=large_context_llm,
        # document_formatting_prompt=use_templates['document_formatting_prompt'],
        extractor_prompt=use_templates['extractor_prompt'],
        initial_question_template=use_templates['initial_question_template'],
        chat_condenser_prompt=use_templates['chat_condenser_prompt'],
        title_chain=build_title_chain(common_llm, use_templates["title_prompt"]),
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        max_tokens_rephrase=MAX_TOKENS_REPHRASE,
        retriever=retriever,
    )


def language_translator(input_text, target_language, model_name, model_param=None):
    if model_param is None:
        model_param = {}
    use_templates = get_templates(['TRANSLATION'], 'default', model_name)
    llm = get_chat_model(name=model_name, stream=False, **model_param)
    llm_chain = use_templates['response_template'] | llm
    return llm_chain.invoke({'text': input_text, 'target_language': target_language}).content


def chat_bot(allowed_tools, model_name, model_params, images=None, image_in_history=False, assistant=False):

    common_llm = get_llm_common(max_tokens=MAX_TOKENS_ROUTER)
    chat_llm = get_chat_model(name=model_name, **model_params, )

    title_chain = build_title_chain(common_llm,
                                    get_templates(["TITLE"], 'default', model_name)["title_prompt"])
    if assistant:
        default_template = get_templates(["ASSISTANT"], 'default', model_name)["default"]

    elif image_in_history or images:
        default_template = get_templates(["IMAGE"], 'default', model_name, images)["response_template"]

    else:
        default_template = get_templates(["PARAMETRIC"], 'default', model_name)["response_template"]


    router_template = get_templates(['MULTITOOL'], 'default', model_name)['router']

    tool_description = " \n- ".join(
        [f"""{t.name}: {t.description}\t[args:{str(convert_to_openai_tool(
            t)['function']['parameters']['properties']).replace('{', '{{').replace('}', '}}')}]{t.example}"""
         for t in allowed_tools])

    return GraphExecutor(
        tool_llm=common_llm,
        chat_llm=chat_llm,
        tools=allowed_tools,
        router_template=router_template.partial(tool_description=tool_description),
        default_template=default_template,
        title_chain=title_chain,
    )


def gtp_web_search(web_retriever, model_name, model_params=None, templates_to_use=None,
                   verbose=False):
    if templates_to_use is None:
        # templates_to_use = ["DOC_FORMATTING", "WEBCHAT", "TITLE", "REPHRASE", "PARAMETRIC"]
        templates_to_use = ["WEBCHAT", "TITLE", "REPHRASE", "PARAMETRIC"]
    if model_params is None:
        model_params = {}

    common_llm = get_llm_common()
    use_templates = get_templates(templates_to_use, 'default', model_name)

    large_context_llm = get_chat_model(name=model_name, **model_params)
    conversational_chain = GeneralChain(
        llm=large_context_llm,
        response_template=use_templates["response_template"],
        title_chain=build_title_chain(common_llm, use_templates["title_prompt"]),
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        verbose=verbose,
    )

    web_chain = WebSearchChain(
        common_llm=common_llm,
        large_context_llm=large_context_llm,
        # document_formatting_prompt=use_templates["web_document_formatting_prompt"],
        response_template=use_templates["web_search"],
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        web_retriever=web_retriever,
        title_chain=build_title_chain(common_llm, use_templates["title_prompt"]),
        initial_question_template=use_templates['initial_question_template'],
        chat_condenser_prompt=use_templates['chat_condenser_prompt'],
        verbose=verbose,
        # title_prompt=use_templates["title_prompt"]
    )

    router_prompt = get_router_template("W")
    router_llm = get_llm_router(max_tokens=MAX_TOKENS_ROUTER)
    routes = {
        "greetings": conversational_chain,
        "enterprise_specific": web_chain
    }

    final_chain = build_attachment_router_chain(router_prompt, router_llm, routes, web_chain)
    # final_chain = RouterChain(
    #     conversational_chain=conversational_chain,
    #     field_specific_chain=subject_chain,
    #     default_chain=subject_chain,
    #     llm=common_llm,
    #     router_prompt=get_route_template("W"),
    #     max_tokens=MAX_TOKENS_ROUTER,
    #     # max_tokens_rephrase=MAX_TOKENS_REPHRASE,
    #     verbose=verbose,
    # )
    log.info("leaving CustomQAChain retrival function")

    return final_chain


def gtp_direct(model_name, templates_to_use=None, model_params=None, verbose=False, **kwargs):
    if templates_to_use is None:
        templates_to_use = ["TITLE", "PARAMETRIC"]
    # if model_params is None:
    #     model_params = DEFAULT_PARAMS

    common_llm = get_llm_common()
    image_count = kwargs.get("image_count", 0)
    use_templates = get_templates(templates_to_use, "default", model_name, user_images=image_count)
    large_context_llm = get_chat_model(name=model_name, **model_params)

    return GeneralChain(
        llm=large_context_llm,
        response_template=use_templates["response_template"],
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        title_chain=build_title_chain(common_llm, use_templates["title_prompt"]),
        verbose=verbose
    )


def gpt_attached_docs(retriever, model_name, model_params=None,
                      templates_to_use=None, verbose=False, **kwargs):
    log.info("Entered CustomQAChain retrival function")
    if templates_to_use is None:
        # templates_to_use = ["REPHRASE", "SUBJECT", "DOC_FORMATTING", "TITLE"]
        templates_to_use = ["REPHRASE", "SUBJECT", "TITLE"]
    if model_params is None:
        model_params = {}

    image_count = kwargs.get("image_count", 0)
    chat_template_to_use = ['PARAMETRIC']

    if image_count > 0:
        chat_template_to_use = ["REPHRASE", "SUBJECT", "TITLE", 'IMAGE']

    # final_templates = get_final_templates('en')
    chat_templates = get_templates(chat_template_to_use, 'default', model_name=model_name, user_images=image_count)

    common_llm = get_llm_common()

    large_context_llm = get_chat_model(name=model_name, **model_params)
    conversational_chain = GeneralChain(
        llm=large_context_llm,
        response_template=chat_templates['response_template'],
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        verbose=verbose
    )

    subject_chain = build_subject_chain(retriever, common_llm, large_context_llm,
                                        templates_to_use=templates_to_use,
                                        model_name=model_name)

    router_prompt = attachment_router_prompt()
    router_llm = get_llm_router(max_tokens=MAX_TOKENS_ROUTER)
    routes = {
        "greetings": conversational_chain,
        "attachments": subject_chain
    }

    final_chain = build_attachment_router_chain(router_prompt, router_llm, routes, conversational_chain)

    return final_chain


def gtp_bot(retriever, model_name, model_params=None,
            templates_to_use=None, verbose=False, **kwargs):
    log.info("Entered CustomQAChain retrival function")
    if templates_to_use is None:
        # templates_to_use = ["REPHRASE", "SUBJECT", "DOC_FORMATTING", "TITLE"]
        templates_to_use = ["REPHRASE", "SUBJECT", "TITLE"]
    if model_params is None:
        model_params = {}

    image_count = kwargs.get("image_count", 0)
    chat_template_to_use = ['PARAMETRIC']

    if image_count > 0:
        chat_template_to_use = ["REPHRASE", "SUBJECT", "TITLE", 'IMAGE']

    # final_templates = get_final_templates('en')
    chat_templates = get_templates(chat_template_to_use, 'default', model_name=model_name, user_images=image_count)

    common_llm = get_llm_common()

    large_context_llm = get_chat_model(name=model_name, **model_params)
    conversational_chain = GeneralChain(
        llm=large_context_llm,
        response_template=chat_templates['response_template'],
        max_tokens=MAX_TOKENS_CHAT,
        max_tokens_title=MAX_TOKENS_TITLE,
        verbose=verbose
    )

    subject_chain = build_subject_chain(retriever, common_llm, large_context_llm,
                                        templates_to_use=templates_to_use,
                                        model_name=model_name)

    router_prompt = get_router_template("S")
    router_llm = get_llm_router(max_tokens=MAX_TOKENS_ROUTER)
    routes = {
        "greetings": conversational_chain,
        "enterprise_specific": subject_chain
    }

    final_chain = build_attachment_router_chain(router_prompt, router_llm, routes, subject_chain)

    log.info("leaving CustomQAChain retrieval function")

    return final_chain


def build_doc_inspector_graph(retriever, model_name, templates_to_use=None,
                              **kwargs):
    if templates_to_use is None:
        templates_to_use = ["REPHRASE", "SUBJECT", "TITLE"]
    common_llm = get_llm_common()

    large_context_llm = get_chat_model(name=model_name, **kwargs)
    reflection_model = get_chat_model(name=model_name, stream=False, **kwargs)

    reflection_prompt_temp = get_templates(["DOC_ANALYZER"], "default", model_name)['reflection']
    subject_chain = build_subject_chain(retriever, common_llm, large_context_llm,
                                        templates_to_use=templates_to_use,
                                        model_name=model_name)
    return DocInspector(subject_chain, reflection_model.default, reflection_prompt_temp)


def remove_extra_whitespaces(text):
    extracted_text = text.strip()
    return extracted_text


def remove_non_printable(text):
    printable_text = re.sub(r"[^\x20-\x7E\n\r\t]", "", text)
    # printable_text = re.sub(r"(?<!(\n|\r))(\r\n|\n|\r)(?!(\r\n|\n|\r))", "\n", text)
    return printable_text


def remove_page_numbers_headers_footers(text):
    # Replace page numbers (e.g., "Page 1 of 10")
    text = re.sub(r"Page \d+ of \d+", "\n", text)

    # Replace headers and footers with specific patterns
    text = re.sub(r"HEADER_PATTERN", "\n", text)
    text = re.sub(r"FOOTER_PATTERN", "\n", text)

    return text


def clean_data(text):
    text = remove_extra_whitespaces(text)
    text = remove_non_printable(text)
    text = remove_page_numbers_headers_footers(text)
    return text


def list_files(path):
    r = []
    for root, _, files in os.walk(path):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# def rephrase_chain():
#     model = get_llm_common()
#     templates = get_templates(['REPHRASE'], 'default', model.metadata['model'])
#
#     return build_rephrase_chain(model, **templates)


def process_user_profile(profile: Dict[str, str]):
    return {'user_name': profile.get('name', 'Not Provided'),
            'user_designation': profile.get('designation', 'Not Provided'),
            'user_department': profile.get('department', 'Not Provided'),
            'user_personalization': profile.get('user_personalization', 'Not Provided'),
            'user_company': profile.get('company_name', 'Not Provided'),
            'user_specified_behaviour': profile.get('response_customization', ''),
            'user_country': profile.get('country', 'Not Provided'),
            'current_date': date.today()}
