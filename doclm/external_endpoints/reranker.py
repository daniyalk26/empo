import os
import logging
import httpx
from typing import List, Union

from infinity_client import Client, AuthenticatedClient
from infinity_client.models import RerankInput, ReRankResult
from infinity_client.api.default import rerank
from unidecode import unidecode

from langchain_core.documents import Document

log = logging.getLogger("doclogger")

SUMMARY_WORD_LIMIT = 60


class InfinityRerank:
    def __init__(self, url, model_name, **kwargs):
        timeout = httpx.Timeout(2.0, connect=5.0)
        if kwargs.get('token', None):
            self.client = AuthenticatedClient(base_url=url, token=kwargs['token'], timeout=timeout)
        else:
            self.client = Client(base_url=url, timeout=timeout)
        self.model_name: str = model_name
        self.top_n = kwargs['top_n']
        self.relevance_threshold = kwargs.get('relevance_threshold', 0)

    async def arerank(self, query: str, doc_list: List[str], top_n=None, **kwargs):
        top_n = top_n or self.top_n
        return_documents = kwargs.get('return_documents', False)
        raw_score = kwargs.get('raw_score', False)
        # with self.client as client:
        reranked_results = await rerank.asyncio(client=self.client, body=RerankInput.from_dict({
            "query": query,
            "documents": doc_list,
            "model": self.model_name,
            "top_n": top_n,
            "return_documents": return_documents,
            "raw_score": raw_score
        }))
        return reranked_results

    def rerank(self, query: str, doc_list: List[str], top_n=None, **kwargs):
        top_n = top_n or self.top_n
        return_documents = kwargs.get('return_documents', False)
        raw_score = kwargs.get('raw_score', False)
        # with self.client as client:
        reranked_results = rerank.sync(client=self.client, body=RerankInput.from_dict({
            "query": query,
            "documents": doc_list,
            "model": self.model_name,
            "top_n": top_n,
            "return_documents": return_documents,
            "raw_score": raw_score
        }))
        return reranked_results

    def unaccent(self, input_str: Union[str, List[str]]):
        if isinstance(input_str, List):
            return [unidecode(in_str) for in_str in input_str]
        elif isinstance(input_str, str):
            return unidecode(input_str)
        else:
            raise TypeError(f'Expected a list of str of str, got {type(input_str)}')

    def rerank_documents(self, query, unranked_documents: List[Document], **kwargs):
        if len(unranked_documents) == 0:
            return []
        relevance_threshold = kwargs.get('relevance_threshold', self.relevance_threshold)
        txts_to_rerank = [
            ' '.join(doc.metadata['document_summary'].split(' ')[:SUMMARY_WORD_LIMIT]) + ' \n'
            + doc.metadata['name'] + ' \n'
            + doc.metadata['author'] + ' \n'
            + doc.page_content
            for doc in unranked_documents
        ]
        reranked_results: ReRankResult = self.rerank(query, txts_to_rerank, **kwargs)
        result = [(unranked_documents[reranked_result.index], reranked_result.relevance_score) for reranked_result in
                  reranked_results.results if reranked_result.relevance_score >= relevance_threshold]
        return result

    async def arerank_documents(self, query, unranked_documents: List[Document], **kwargs):
        if len(unranked_documents) == 0:
            return []
        relevance_threshold = kwargs.get('relevance_threshold', self.relevance_threshold)
        txts_to_rerank = [
            ' '.join(doc.metadata['document_summary'].split(' ')[:SUMMARY_WORD_LIMIT]) + ' \n'
            + doc.metadata['name'] + ' \n'
            + doc.metadata['author'] + ' \n'
            + doc.page_content
            for doc in unranked_documents
        ]
        reranked_results: ReRankResult = await self.arerank(query, txts_to_rerank, **kwargs)
        result = [(unranked_documents[reranked_result.index], reranked_result.relevance_score) for reranked_result in
                  reranked_results.results if reranked_result.relevance_score >= relevance_threshold]
        return result

    def close(self):
        self.client.__exit__()


def get_reranker(*args, **kwargs):
    return InfinityRerank(*args, **kwargs)
