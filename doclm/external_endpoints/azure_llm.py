"""
    implements llm model chains and LLM model endpoints
"""
import os
import logging

from functools import lru_cache

import tiktoken

from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.runnables import ConfigurableField

from .azure_endpoint_client import CustomFormatter, AzureMLChatEndpoint
from ..config import chat_model_config

log = logging.getLogger("doclogger")

# pylint: disable=invalid-name,W0613:


def open_ai_model(deployment_name_envar: str,
                  model_name_envar: str,
                  temperature: float = 0.0,
                  max_tokens: int = 800,
                  metadata=None,
                  **kwargs) -> Runnable:
    return AzureChatOpenAI(
        deployment_name=os.environ[deployment_name_envar],
        model_name=os.environ[model_name_envar],
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata,
        model_kwargs=kwargs,
    ).configurable_fields(
        max_tokens=ConfigurableField(
            id="output_token_number",
            name="Max tokens in the output",
            description="The maximum number of tokens in the output",
        )
    )

def google_model(
                  model_name_envar: str,
                  temperature: float = 0.0,
                  max_tokens: int = 3000,
                  metadata=None,
                  **kwargs) -> Runnable:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as I_e:
        log.error(I_e, exc_info=True)
        raise

    return ChatGoogleGenerativeAI(
        model=os.environ['DIRECT_CHAT_MODEL_NAME_flash_2o'],
        temperature=temperature,
        max_output_tokens=max_tokens,
        metadata=metadata,
    )

def azure_endpoint_models(base_url_envar: str, api_key_envar: str, end_point_version,
                          stop=None, deployment_name=None,
                          metadata=None, max_tokens=None, **kwargs) -> Runnable:
    encoding = tiktoken.get_encoding("cl100k_base")

    return AzureMLChatEndpoint(
        stop=stop,
        max_tokens=max_tokens,
        endpoint_url=os.environ[base_url_envar] + end_point_version,  # '/v1/chat/completions',
        endpoint_api_type="serverless",
        endpoint_api_key=os.environ[api_key_envar],
        content_formatter=CustomFormatter(),
        deployment_name=deployment_name,
        metadata=metadata,
        model_kwargs=kwargs,
        custom_get_token_ids=lambda x: encoding.encode(x),
    ).configurable_fields(
        max_tokens=ConfigurableField(
            id="output_token_number",
            name="Max tokens in the output",
            description="The maximum number of tokens in the output",
        )
    )

def azure_openai(deployment_name_envar: str, model_name_envar: str,
                  metadata=None, temperature=1,
                  **kwargs) -> Runnable:
    client = AzureChatOpenAI(
        deployment_name=os.environ[deployment_name_envar],
        model_name=os.environ[model_name_envar],
        temperature=temperature,
        api_version="2024-12-01-preview",
        model_kwargs=kwargs,
        metadata=metadata,
    )
    return client

def azure_eval_model(temperature=0.0, max_tokens=3000, **kwargs):
    """

    :param temperature: 0.0-2.0
    :param max_tokens:
    :param kwargs:
    :return:
    """

    llm_common_deployment_name = os.getenv("LLM_EVALUATION_DEPLOYMENT_NAME",
                                           os.environ["LLM_EVALUATION_DEPLOYMENT_NAME"])
    llm_common_model_name = os.getenv("LLM_EVALUATION_MODEL_NAME",
                                      os.environ["LLM_EVALUATION_MODEL_NAME"])

    return AzureChatOpenAI(
        deployment_name=llm_common_deployment_name,
        model_name=llm_common_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        # streaming=True,
    )


def azure_translate_model(temperature=0.0, max_tokens=800, **kwargs):
    """

    :param temperature: 0.0-2.0
    :param max_tokens:
    :param kwargs:
    :return:
    """

    llm_common_deployment_name = os.environ["LLM_TRANSLATION_DEPLOYMENT_NAME"]
    llm_common_model_name = os.environ["LLM_TRANSLATION_MODEL_NAME"]

    return AzureChatOpenAI(
        deployment_name=llm_common_deployment_name,
        model_name=llm_common_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        # streaming=True,
    )


evaluation_model = {
    "gpt4o": azure_eval_model,
    "gemini-2.0-flash": google_model,
}

translation_model = {
    "gpt4": azure_translate_model,
}


def get_evaluation_model(name="gpt4o", **kwargs):
    """
    get function for evaluation model
    :param name: azure/openai
    :param kwargs: config parameters for chat model
    :return: *ChatOpenAI object
    """
    assert name in evaluation_model
    log.info(" loading %s chat model", name)

    return evaluation_model[name](**kwargs)


def get_translation_model(name="gpt4", **kwargs):
    """
    get function for translation model
    :param name: azure/openai
    :param kwargs: config parameters for chat model
    :return: *ChatOpenAI object
    """
    assert name in evaluation_model
    log.info(" loading %s chat model", name)

    return translation_model[name](**kwargs)


def get_llm_router(**kwargs):
    name = os.getenv("LLM_ROUTER", "gpt-4o")
    return get_chat_model(name=name, stream=False, **kwargs)


def get_llm_common(**kwargs):
    new_kwargs = kwargs.copy()
    new_kwargs.update({"stream": False})
    return get_chat_model(os.environ["LLM_COMMON"], **new_kwargs)


@lru_cache
def get_chat_model(name, **kwargs):
    assert name in chat_model_config, f'{name} not in {chat_model_config.keys()}'
    log.info("loading %s chat model large", name)
    model_config = chat_model_config[name].copy()
    model_type = model_config['type']
    model_kwargs = model_config['model_kwargs'].copy()
    model_kwargs.update({k_y: kwargs[k_y] for k_y in model_kwargs if k_y in kwargs})
    model_config.pop("model_kwargs")
    model_config['model'] = name

    if model_type == 'openai':
        return open_ai_model(**model_kwargs, metadata=model_config)
    elif model_type == 'azure':
        return azure_endpoint_models(**model_kwargs, metadata=model_config)
    elif model_type == "azure-openai":
        return azure_openai(**model_kwargs, metadata=model_config)
    elif model_type == "google":
        return google_model(**model_kwargs, metadata=model_config)
    raise ValueError("could not determine model type pleas check `chat_model_large` dict")
