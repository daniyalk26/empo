from __future__ import annotations

# import asyncio
# import contextlib
import os
import enum
import logging
from math import exp
from typing import (
    # TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    ClassVar,
    Collection,
    # Optional,
    # Tuple,
    Type,
)
from typing import Optional, Tuple
import numpy as np
import warnings

from langchain_core.documents import Document
from langchain_community.utils.math import cosine_similarity
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
    )
from langchain_core.pydantic_v1 import Field, root_validator
from .custom_pgkeyword import DocumentPGKeyword
from .custom_pgvector import DocumentPGVector

from .base import DocumentPGBase


sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")

# pylint: disable=W0622,invalid-name


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class MergingStrategy(str, enum.Enum):
    """Enumerator of the Ranking strategies."""

    RRF = "rrf"
    RESCALE = "rescale"
    DUMB = "dumb"


DEFAULT_MERGING_STRATEGY = MergingStrategy.RRF

class DocumentPGHybrid(DocumentPGBase):
    def __init__(
            self,
            vector_store: DocumentPGVector,
            keyword_store: DocumentPGKeyword,
            merging_strategy: MergingStrategy = DEFAULT_MERGING_STRATEGY,
            logger: Optional[logging.Logger] = None,
            merging_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_store= keyword_store
        self._merging_strategy = merging_strategy
        self.logger = logger or logging.getLogger("doclogger")
        self.override_merging_fn = merging_fn
        
    def _simple_merging_fn(self,semantic_docs: List[Tuple[Document, float]], keyword_docs: List[Tuple[Document, float]]):
        sematic_uuids = [sem_doc[0].metadata['uuid'] for sem_doc in semantic_docs]
        result = semantic_docs+[doc for doc in keyword_docs if doc[0].metadata['uuid'] not in sematic_uuids]
        result.sort(key=lambda x:x[1], reverse=True)
        return result
    
    def _rescale_merging_fn(self, semantic_docs: List[Tuple[Document, float]], keyword_docs: List[Tuple[Document, float]]):
        uuid_semantic_score_rescaled = [distance for _, distance in semantic_docs]
        uuid_keyword_score_rescaled = [distance for _, distance in keyword_docs]
        semantic_normalization_factor = sum(uuid_semantic_score_rescaled)
        keyword_normalization_factor = sum(uuid_keyword_score_rescaled)
        uuid_semantic_score_rescaled = [score/semantic_normalization_factor for score in uuid_semantic_score_rescaled]
        uuid_keyword_score_rescaled = [score/keyword_normalization_factor for score in uuid_keyword_score_rescaled]

        uuid_dict = {doc.metadata['uuid']: (doc,rescaled_score) for (doc, _), rescaled_score in zip(semantic_docs, uuid_semantic_score_rescaled)}
        for (doc, _), rescaled_score in zip(keyword_docs, uuid_keyword_score_rescaled):
            doc_uuid=doc.metadata['uuid'] 
            if doc_uuid not in uuid_dict:
                uuid_dict[doc_uuid] = (doc, rescaled_score)
            else:
                uuid_dict[doc_uuid] = (doc, uuid_dict[doc_uuid][1]+rescaled_score)

        result = list(uuid_dict.values())
        result.sort(key=lambda x:x[1], reverse=True)
        return result
    
    def _rrf_merging_fn(self,semantic_docs, keyword_docs, lamda=0.5):
        semantic_reciprocal_rank = [lamda/(60+rank) for rank in range(1,len(semantic_docs)+1)]
        keyword_reciprocal_rank = [(1-lamda)/(60+rank) for rank in range(1,len(keyword_docs)+1)]
        uuid_dict = {doc.metadata['uuid']: (doc,rescaled_score) for (doc, _), rescaled_score in zip(semantic_docs, semantic_reciprocal_rank)}

        for (doc, _), rescaled_score in zip(keyword_docs, keyword_reciprocal_rank):
            doc_uuid = doc.metadata['uuid']
            if doc_uuid not in uuid_dict:
                uuid_dict[doc_uuid] = (doc, rescaled_score)
            else:
                uuid_dict[doc_uuid] = (uuid_dict[doc_uuid][0], uuid_dict[doc_uuid][1]+rescaled_score)
        
        result = list(uuid_dict.values())
        result.sort(key=lambda x:x[1], reverse=True)
        return result
    

    def _select_merging_fn(self):
        if self.override_merging_fn is not None:
            return self.override_merging_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._merging_strategy == MergingStrategy.RRF:
            return self._rrf_merging_fn
        elif self._merging_strategy == MergingStrategy.DUMB:
            return self._simple_merging_fn
        elif self._merging_strategy == MergingStrategy.RESCALE:
            return self._rescale_merging_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._merging_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
    def hybrid_search(
            self,
            query: str,
            lang: str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding_data = self.vector_store._get_embeddings(query)
        docs_and_scores_vector = self.vector_store.similarity_search_with_score_by_vector(
            embedding=embedding_data, lang=lang,tenant_id=tenant_id, k=k, filter=filter,
            # fetch_k=kwargs['fetch_k'],
            )
        docs_and_scores_keyword = self.keyword_store.keyword_search_with_score_by_query(
            query=query, lang=lang, tenant_id=tenant_id, k=k, filter=filter,

        )
        merge_func=self._select_merging_fn()
        docs_and_scores=merge_func(docs_and_scores_vector, docs_and_scores_keyword)
        
        # return _results_to_docs(docs_and_scores)
        return docs_and_scores

    def hybrid_search_with_score(
            self,
            query: str,
            lang: str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        docs = self.hybrid_search(
            query=query,lang=lang,tenant_id=tenant_id, k=k, filter=filter, **kwargs
        )
        return docs

    def hybrid_search_with_score_by_query_and_vector(
            self,
            query: str,
            lang: str,
            embedding: dict,
            tenant_id: int,
            k: int = 20,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            Defaults to 20.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.

        """
        self.logger.debug('printing for query ')
        self.logger.debug(query)
        self.logger.debug('printing for embedding ')
        self.logger.debug(embedding)

        results_keyword = self.keyword_store.__query_collection(query=query, lang=lang, tenant_id=tenant_id,k=k, filter=filter)
        results_vector= self.vector_store.__query_collection(embedding=embedding,lang=lang,tenant_id=tenant_id, k=k, filter=filter)
        
        embedding_list = [result.EmbeddingStore.embedding for result in results]
        results = self._results_to_docs_and_scores(results)
        # results = self._remove_duplicate_docs(results)
        results = self._diverse_docs_filter(results, embedding_list)
        results = self._rescale_scores(results)
        results = self._filter_results(results, k)
        return results

    def _results_to_docs_and_scores(self, results: Iterable) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                self._prepare_document(result),
                1 - result.distance,
            )
            for result in results
        ]
        self.logger.info(
            "similarity score (to query) of retrieved chunks %s",
            str([doc[1] for doc in docs])
        )
        return docs

    #TODO: Need to find a faster way for checking overlap
    def _overlap_between_strings(self, current_str: str, next_str: str):
        for ele in range(1, len(current_str)):
            if next_str.startswith(current_str[ele:]):
                return current_str[ele:]
        return ""

    def _remove_overlap_from_prev_string(self, prev_str:str, current_str:str):
        result=prev_str
        if len(current_str) == 0:
            return result
        overlap_length = len(self._overlap_between_strings(prev_str, current_str))
        result = prev_str[: len(prev_str) - overlap_length]
        return result

    def _remove_overlap_from_next_string(self, current_str:str, next_str:str):
        result=next_str
        if len(current_str) == 0:
            return result
        overlap_length = len(self._overlap_between_strings(current_str, next_str))
        result = next_str[overlap_length:]
        return result
        
    # This function does not handle heading boundaries which would occur in case of structured parser.
    def _remove_chunk_text_overlap_from_doc_metadata(self, page_content, metadata: dict):
        metadata[self.previous_chunk_tag]=self._remove_overlap_from_prev_string(metadata[self.previous_chunk_tag], page_content)
        metadata[self.next_chunk_tag]=self._remove_overlap_from_next_string(page_content, metadata[self.next_chunk_tag])
        return metadata

    def _remove_duplicate_docs(self, results):
        deduplicated_results = []
        score_list = []
        for doc, doc_score in results:
            if len(score_list) == 0:
                score_list.append(doc_score)
                deduplicated_results.append((doc, doc_score))
            else:
                if doc_score not in score_list:
                    score_list.append(doc_score)
                    deduplicated_results.append((doc, doc_score))
        return deduplicated_results

    def _diverse_docs_filter(self, results, embeddings_list: List):
        diverse_results = []
        idx_list = []
        for idx, (doc, doc_score) in enumerate(results):
            if len(diverse_results) == 0:
                idx_list.append(idx)
                diverse_results.append((doc, doc_score))
            else:
                max_interchunk_similarity = max(
                    cosine_similarity(
                        np.array([embeddings_list[idx_list[result_idx]]]),
                        np.array([embeddings_list[idx]]),
                    )[0]
                    for result_idx, _ in enumerate(diverse_results)
                )
                if max_interchunk_similarity > 0.98:
                    self.logger.info(
                        "found similar chunk, skipping chunk number %d with source %s, "
                        "page number %s, max interchunk similarity score %f",
                        idx,
                        doc.metadata[self.name_tag],
                        doc.metadata[self.page_tag],
                        max_interchunk_similarity,
                    )
                else:
                    idx_list.append(idx)
                    diverse_results.append((doc, doc_score))
        self.logger.info("diverse chunks indexes %s", idx_list)
        return diverse_results

    def _rescale_scores(self, results, method="softmax"):
        # "linear" or "softmax" allowed
        scores_list = [doc_score for _, doc_score in results]
        self.logger.debug("Scores for documents are: %s", scores_list)
        if method == "linear":
            sum_of_scores = sum(scores_list)
            _results = [
                (doc, scores_list[idx] / sum_of_scores)
                for idx, (doc, _) in enumerate(results)
            ]
        else:
            exp_scores_list = [exp(doc_score) for _, doc_score in results]
            exp_sum_of_scores = sum(exp_scores_list)
            _results = [
                (doc, exp_scores_list[idx] / exp_sum_of_scores)
                for idx, (doc, _) in enumerate(results)
            ]
        self.logger.info("rescaled scores are %s", str([_res[1] for _res in _results]))
        return _results

    # @staticmethod
    def _filter_results(self, results, max_num_of_results, coverage_threshold=1.0):
        accumulator = 0.0
        new_result = []

        for result in results:
            if (accumulator < coverage_threshold) and (
                    len(new_result) < max_num_of_results
            ):
                accumulator += result[1]
                new_result.append(result)
            else:
                break
        self.logger.info("Coverage of %s is %s", len(new_result), accumulator)
        return new_result

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string = kwargs.get("connection_string",
                                       os.environ['PGVECTOR_CONNECTION_STRING'])

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the PGVECTOR_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def connection_string_from_db_params(
            cls,
            driver: str,
            host: str,
            port: int,
            database: str,
            user: str,
            password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"

    # def _select_relevance_score_fn(self) -> Callable[[float], float]:
    #     """
    #     The 'correct' relevance function
    #     may differ depending on a few things, including:
    #     - the distance / similarity metric used by the VectorStore
    #     - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    #     - embedding dimensionality
    #     - etc.
    #     """
    #     if self.override_relevance_score_fn is not None:
    #         return self.override_relevance_score_fn

    #     # Default strategy is to rely on distance strategy provided
    #     # in vectorstore constructor
    #     if self._ranking_strategy == RankingStrategy.SIMPLE_RANK:
    #         return self._simple_rank_relevance_score_fn
    #     elif self._ranking_strategy == RankingStrategy.CD_RANK:
    #         return self._cd_rank_relevance_score_fn
    #     else:
    #         raise ValueError(
    #             "No supported normalization function"
    #             f" for distance_strategy of {self._ranking_strategy}."
    #             "Consider providing relevance_score_fn to PGVector constructor."
    #         )
    
    def _get_retriever_tags(self) -> List[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        # if self.embeddings:
        #     tags.append(self.embeddings.__class__.__name__)
        return tags
    
    def hybrid_search_with_relevance_scores(
        self,
        query: str,
        lang: str,
        tenant_id: int,
        k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold_keyword: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        score_threshold_keyword = kwargs.pop("score_threshold_keyword", None)
        score_threshold_semantic = kwargs.pop("score_threshold_semantic", None)

        docs_and_similarities_semantic = self.vector_store._similarity_search_with_relevance_scores(
            query,lang=lang,tenant_id=tenant_id, k=k, **kwargs
        )
        docs_and_similarities_keyword = self.keyword_store._similarity_search_with_relevance_scores(
            ' '.join(query.split(' ')[:40]+query.split(' ')[-30:]),lang=lang,tenant_id=tenant_id, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities_semantic
        ):
            warnings.warn(
                "Relevance scores for semantic must be between"
                f" 0 and 1, got {docs_and_similarities_semantic}"
            )

        if score_threshold_semantic is not None:
            docs_and_similarities_semantic = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities_semantic
                if similarity >= score_threshold_semantic
            ]
            if len(docs_and_similarities_semantic) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold_semantic}"
                )
        
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities_keyword
        ):
            warnings.warn(
                "Relevance scores for semantic must be between"
                f" 0 and 1, got {docs_and_similarities_keyword}"
            )

        if score_threshold_keyword is not None:
            docs_and_similarities_keyword = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities_keyword
                if similarity >= score_threshold_keyword
            ]
            if len(docs_and_similarities_keyword) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold_keyword}"
                )
        merge_func=self._select_merging_fn()
        docs_and_scores=merge_func(docs_and_similarities_semantic, docs_and_similarities_keyword)
        return docs_and_scores
    
    def as_retriever(self, **kwargs: Any) -> DocumentPGHybridRetriever:
        """Return CustomKeywordStoreRetriever initialized from this VectorStore.

        Args:
            **kwargs: Keyword arguments to pass to the search function.
                Can include:
                search_type (Optional[str]): Defines the type of search that
                    the Retriever should perform.
                    Can be "similarity" (default), "mmr", or
                    "similarity_score_threshold".
                search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                    search function. Can include things like:
                        k: Amount of documents to return (Default: 4)
                        score_threshold: Minimum relevance threshold
                            for similarity_score_threshold
                        fetch_k: Amount of documents to pass to MMR algorithm
                            (Default: 20)
                        lambda_mult: Diversity of results returned by MMR;
                            1 for minimum diversity and 0 for maximum. (Default: 0.5)
                        filter: Filter by document metadata

        Returns:
            VectorStoreRetriever: Retriever class for VectorStore.

        Examples:

        .. code-block:: python

            # Retrieve more documents with higher diversity
            # Useful if your dataset has many similar documents
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            )

            # Fetch more documents for the MMR algorithm to consider
            # But only return the top 5
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 50}
            )

            # Only retrieve documents that have a relevance score
            # Above a certain threshold
            docsearch.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            )
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return DocumentPGHybridRetriever(store=self, tags=tags, **kwargs)

    def close_connection(self):
        try:
            self._conn.close()
        except Exception as e:
            self.logger.error(e, exc_info=True, stack_info=True)
    
    def results_merger_advance_search(self, semantic_docs, keyword_docs, lamda=0.3):
        # semantic_docs.sort(key=lambda x:x.score, reverse=True)
        # keyword_docs.sort(key=lambda x:x.score, reverse=True)
        semantic_reciprocal_rank = [lamda/(60+rank) for rank in range(1,len(semantic_docs)+1)]
        keyword_reciprocal_rank = [(1-lamda)/(60+rank) for rank in range(1,len(keyword_docs)+1)]
        id_dict = {doc.doc_id: (doc,rescaled_score) for doc, rescaled_score in zip(keyword_docs, keyword_reciprocal_rank)}

        for doc, rescaled_score in zip(semantic_docs, semantic_reciprocal_rank):
            doc_id = doc.doc_id
            if doc_id not in id_dict:
                id_dict[doc_id] = (doc, rescaled_score)
            else:
                id_dict[doc_id] = (id_dict[doc_id][0], id_dict[doc_id][1]+rescaled_score)
        
        result = list(id_dict.values())
        result.sort(key=lambda x:x[1], reverse=True)
        return result
    
    def adv_search_query_collection(self, query_embedding, user_query, tenant_id, lang, filter, k):
        results_semantic = self.vector_store.adv_search_query_collection(query_embedding,tenant_id, lang, filter, k)
        results_keyword = self.keyword_store.adv_search_query_collection(' '.join(user_query.split(' ')[:40]+user_query.split(' ')[-30:]),tenant_id, lang, filter, k)
        merge_docs=self.results_merger_advance_search(results_semantic, results_keyword, lamda=0.3)
        return merge_docs
    
    def advance_search(self, user_query, tenant_id, lang, filter, k, **kwargs):
        query_embedding =  self.vector_store.embedding_function.embed_query(user_query)
        docs_and_scores = self.adv_search_query_collection(query_embedding, user_query, tenant_id, lang, filter, k)
        docs = []
        for doc, score in docs_and_scores[:k]:
            doc.score = score
            docs.append(doc)
        return docs

    def advance_search_with_threshold(self, user_query, tenant_id, lang, filter, k, score_threshold_semantic, score_threshold_keyword, **kwargs):
        # query_embedding =  self.vector_store.embedding_function.embed_query(user_query)
        self.logger.debug(f'user_query: {user_query}, filter: {filter} and k: {k}')
        results_semantic = self.vector_store.advance_search(user_query,tenant_id, lang, filter, k,callbacks = kwargs['callbacks'])
        results_keyword = self.keyword_store.advance_search(' '.join(user_query.split(' ')[:40]+user_query.split(' ')[-30:]),tenant_id, lang, filter, k)
        self.logger.info(f'got {len(results_semantic)} docs from semantic and {len(results_keyword)} from keyword search')
        results_semantic = [result for result in results_semantic if result.score>score_threshold_semantic]
        results_keyword = [result for result in results_keyword if result.score>score_threshold_keyword]
        docs_and_scores=self.results_merger_advance_search(results_semantic, results_keyword, lamda=0.3)
        docs = []
        for doc, score in docs_and_scores[:k]:
            doc.score = score
            docs.append(doc)
        return docs

    

class DocumentPGHybridRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    store: DocumentPGHybrid
    """VectorStore to use for retrieval."""
    # name: str = "hybrid"
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "advanced_search",
        "advanced_search_with_threshold"
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type.

        Args:
            values: Values to validate.

        Returns:
            Values: Validated values.

        Raises:
            ValueError: If search_type is not one of the allowed search types.
            ValueError: If score_threshold is not specified with a float value(0~1)
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            raise ValueError(
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
        if search_type in ["similarity_score_threshold", "advanced_search_with_threshold"]:
            score_threshold_semantic = values.get("search_kwargs", {}).get("score_threshold_semantic")
            score_threshold_keyword = values.get("search_kwargs", {}).get("score_threshold_keyword")
            if (score_threshold_semantic is None) or (not isinstance(score_threshold_semantic, float))\
             or (score_threshold_keyword is None) or (not isinstance(score_threshold_keyword, float)):
                raise ValueError(
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        return values

    

    def _get_relevant_documents(
        self, inputs: dict,*, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        lang=inputs['lang']
        tenant_id=inputs['tenant_id']
        query=inputs['query']
        search_kwargs = self.search_kwargs.copy()
        search_kwargs.update(
            {
                'filter': inputs['filter'],
                'k' : inputs['k']
            }
        )


        if self.search_type == "similarity":
            docs = self.store.hybrid_search(query,lang, tenant_id, **search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.store.hybrid_search_with_relevance_scores(
                    query,lang, tenant_id, **search_kwargs
                )
            )
            docs = docs_and_similarities
            # docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "advance_search":
            callbacks=run_manager.get_child().handlers

            docs = (
                self.store.advance_search(
                    query, tenant_id,lang, **search_kwargs, callbacks = callbacks
                )
            )
        elif self.search_type == "advanced_search_with_threshold":
            callbacks=run_manager.get_child().handlers

            docs = (
                self.store.advance_search_with_threshold(
                    query, tenant_id, lang, **search_kwargs,callbacks = callbacks
                )
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.store.ahybrid_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.store.ahybrid_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
