"""
 This module contains embedding model
"""
import os
import logging

from  langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

log = logging.getLogger("doclogger")
log.disabled = False
EMBEDDING_BATCH_SIZE = min(16, int(os.getenv('EMBEDDING_BATCH_SIZE', '16')))

# @retry(stop=stop_after_attempt(7), wait=wait_random_exponential(multiplier=0.5, min=10, max=180))

def openai_embedding():
    return OpenAIEmbeddings()


def ms_embedding():

    return AzureOpenAIEmbeddings(
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment=os.environ["EMBED_DEPLOYMENT_NAME"],
        model=os.environ["EMBED_MODEL_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        dimensions=1536,
        chunk_size=EMBEDDING_BATCH_SIZE,
        max_retries=7,
        retry_max_seconds=10,
        retry_min_seconds=180,
    )


embeddings = {'openai': openai_embedding,
              'azure': ms_embedding}


def get_embeddings(name='azure'):
    """
    function to get embedding model from list of selectable models
    :param name: Embedding name  can be {openai, azure}
    :return: OpenAIEmbeddings object
    """
    assert name in embeddings
    log.info("loading %s embeddings", name)
    return embeddings[name]()
