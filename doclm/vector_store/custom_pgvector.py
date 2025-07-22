from __future__ import annotations

import asyncio
from asyncio import current_task
import os
import enum
import logging
from math import exp
import math
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Collection,
    ClassVar,
    # Optional,
    # Tuple,
    Type,
)
from typing import Optional, Tuple
import numpy as np
import warnings

import sqlalchemy
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.sql import text

# from sqlalchemy.dialects.postgresql import to_tsvector, REGCONFIG
from sqlalchemy.ext.asyncio import async_scoped_session
# from sqlalchemy.types import UserDefinedType

from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.utils.math import cosine_similarity
from langchain.vectorstores.utils import maximal_marginal_relevance

from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

from .base import DocumentPGBase

# pylint: disable=W0622,invalid-name




SUMMARY_WORD_LIMIT = 60

def try_page_to_page_range(page_no: str, number_of_pages: int):
    try:
        if number_of_pages>0:
            return page_no+' - '+str(int(page_no)+number_of_pages)
        return page_no
    except:
        return page_no

def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

_DEFAULT_COLLECTION_NAME = "langchain"

# VectorStore
class DocumentPGVector(DocumentPGBase):
    def __init__(self, connection_string: str, embedding_function: Embeddings,
                 distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
                 logger: Optional[logging.Logger] = None,
                 relevance_score_fn: Optional[Callable[[float], float]] = None,
                 **db_kwargs) -> None:

        super().__init__(connection_string, logger, db_kwargs, embedding_function)
        self._distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn

    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the Euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - distance / math.sqrt(2)

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""

        return 1.0 - distance

    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        if distance > 0:
            return 1.0 - distance

        return -1.0 * distance
    
    def similarity_search(
            self,
            query: str,
            lang:str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding_data = self._get_embeddings(query)
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding_data, lang=lang,tenant_id=tenant_id, k=k, filter=filter,
            # fetch_k=kwargs['fetch_k'],

        )
        # return _results_to_docs(docs_and_scores)
        return docs_and_scores


    def similarity_search_with_score(
            self,
            query: str,
            lang:str,
            tenant_id:int,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs

    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, lang=lang, tenant_id=tenant_id, k=k, filter=filter
        )
        return docs

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def similarity_search_with_score_by_vector(
            self,
            embedding: dict,
            lang:str,
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
        self.logger.info('printing embeddings for query ')
        self.logger.debug(embedding)

        results = self.__query_collection(embedding=embedding, lang=lang,
                                          tenant_id=tenant_id, k=k, filter=filter)
        embedding_list = [result.EmbeddingStore.embedding for result in results]
        results = self._results_to_docs_and_scores(results)
        # results = self._remove_duplicate_docs(results)
        results = self._diverse_docs_filter(results, embedding_list)
        # results = self._rescale_scores(results)
        # results = self._filter_results(results, k)
        return results

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                self._prepare_document(result),
                1 - result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]
        self.logger.info(
            "similarity score (to query) of retrieved chunks %s",
            str([doc[1] for doc in docs])
        )
        return docs

    def __query_collection(
            self,
            embedding: dict,
            lang:str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self.logger.info('Requesting a connection from db pool')
        try:
            with scoped_session(self._session_factory)() as session:
                collection = self.get_collection_by_tenant_id(session, tenant_id)
                session.execute(text("SET LOCAL hnsw.ef_search = 1000;"))
                if not collection:
                    raise ValueError("Collection not found")

                files = filter['id']
                stmt = text(
                    f"""select * from semantic_nolang('{embedding}'::vector, {k}, {str(collection.collection_id)}, VARIADIC ARRAY{files});""")
                results: List[Any] = [self.EmbeddingStore.apply_schema(*r) for r in session.execute(stmt).all()]
        except Exception as e:
            self.logger.error(e, exc_info=True,stack_info=True)
            raise e
        self.logger.info('db connection closed')
        return results

    async def asimilarity_search_with_relevance_scores(
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
                score_threshold_semantic: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        score_threshold = kwargs.pop("score_threshold_semantic", None)
        # relevance_score_fn = self._select_relevance_score_fn()
        # embedding = self.embedding_function.embed_query(query)
        embedding = await self.embedding_function.aembed_query(query)
        #todo : find why asynvc version is not working
        # embedding = await self.embedding_function.aembed_documents([query])

        docs_and_similarities = await self.asimilarity_search_with_score_by_vector(
           embedding=embedding, lang=lang, tenant_id=tenant_id, k=k,
            filter=kwargs.get("filter")
       )

        if any(
                similarity < 0.0 or similarity > 1.0
                for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities


    async def asimilarity_search_with_score_by_vector(
            self,
            embedding: List,
            lang: str,
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
        self.logger.info('printing embeddings for query ')
        self.logger.debug(embedding)

        results = await self.async_query_collection(
            embedding=embedding, lang=lang,
            tenant_id=tenant_id, k=k, filter=filter)

        embedding_list = [result.EmbeddingStore.embedding for result in results]
        results = self._results_to_docs_and_scores(results)
        # results = self._remove_duplicate_docs(results)
        results = self._diverse_docs_filter(results, embedding_list)
        # results = self._rescale_scores(results)
        # results = self._filter_results(results, k)
        return results


    async def async_query_collection(
            self,
            embedding: List,
            lang:str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self.logger.info('Requesting a connection from db pool')
        try:

            async with async_scoped_session(self._async_session_factory, scopefunc=current_task)() as session:

                collection = await self.async_get_collection_by_tenant_id(session, tenant_id)
                await session.execute(text("SET LOCAL hnsw.ef_search = 1000;"))
                if not collection:
                    raise ValueError("Collection not found")

                files = filter['id']
                stmt = text(
                    f"""select * from semantic_nolang('{embedding}'::vector, {k}, {str(collection.collection_id)}, VARIADIC ARRAY{files});""")
                xx =  await session.execute(stmt)
                results: List[Any] = [self.EmbeddingStore.apply_schema(*r) for r in xx.scalar()]

        except Exception as e:
            self.logger.error(e, exc_info=True,stack_info=True)
            raise e
        self.logger.info('db connection closed')
        return results

    def similarity_search_by_vector(
            self,
            embedding: dict,
            lang:str,
            tenant_id: int,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, lang=lang,tenant_id=tenant_id, k=k, filter=filter,
            # fetch_k=kwargs['fetch_k'],
        )
        return docs_and_scores

    def _get_embeddings(self, input_query):
        return self.embedding_function.embed_query(text=input_query)

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

    def _diverse_docs_filter(self, results, embeddings_list):
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

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        lang:str,
        tenant_id: int,
        k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        # relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query=query,lang=lang,tenant_id=tenant_id, k=k, **kwargs)
        # return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        return docs_and_scores
    
    def similarity_search_with_relevance_scores(
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
                score_threshold_semantic: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        score_threshold = kwargs.pop("score_threshold_semantic", None)

        docs_and_similarities = self._similarity_search_with_relevance_scores(
            query,lang=lang,tenant_id=tenant_id, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities

    def max_marginal_relevance_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 20,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)

        embedding_list = [result.EmbeddingStore.embedding for result in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 20,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
            self,
            query: str,
            k: int = 20,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 20,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

        return _results_to_docs(docs_and_scores)

    # async def amax_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 20,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     filter: Optional[Dict[str, str]] = None,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     """Return docs selected using the maximal marginal relevance."""
    #
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(
    #         self.max_marginal_relevance_search_by_vector,
    #         embedding,
    #         k=k,
    #         fetch_k=fetch_k,
    #         lambda_mult=lambda_mult,
    #         filter=filter,
    #         **kwargs,
    #     )
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    def close_connection(self):
        try:
            self._conn.close()
        except Exception as e:
            self.logger.error(e, exc_info=True, stack_info=True)

    def _get_retriever_tags(self) -> List[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags
    
    def as_retriever(self, **kwargs: Any) -> DocumentPGVectorRetriever:
        """Return VectorStoreRetriever initialized from this VectorStore.

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
                        score_threshold_semantic: Minimum relevance threshold
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
                search_kwargs={'score_threshold_semantic': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            )
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return DocumentPGVectorRetriever(store=self, tags=tags, **kwargs)
    
    def adv_search_query_collection(self, query_embedding, tenant_id, lang, filter, k):
        self.logger.info('Requesting a connection from db pool')
        try:
            with scoped_session(self._session_factory)() as session:
                collection = self.get_collection_by_tenant_id(session, tenant_id)
                session.execute(text("SET LOCAL hnsw.ef_search = 1000;"))
                if not collection:
                    raise ValueError("Collection not found")
                files = filter['id']
                stmt = text(
                    f"""select * from adv_search_semantic_nolang('{query_embedding}'::vector, {k}, {str(collection.collection_id)}, VARIADIC ARRAY{files});""")
                results: List[Any] = [self.AdvSearchReadSchema.apply_schema(*r) for r in session.execute(stmt).all()]
        except Exception as e:
            self.logger.error(e, exc_info=True,stack_info=True)
            raise e
        self.logger.info('db connection closed')
        for result in results:
            result.search_type='semantic'
            if result.encrypted=='true':
                result.chunk_text,_ = self.decrypt_data(result.chunk_text, {})
                result.doc_summary,_=self.decrypt_data(result.doc_summary, {})
                result.encrypted='false'
            result.score = 1-result.score
            result.page_no = try_page_to_page_range(result.page_no, len(result.jmeta.get('page_breaks_char_offset',[])))
        return results
    
    def advance_search(self, user_query, tenant_id, lang, filter, k, **kwargs):
        query_embedding =  self.embedding_function.embed_query(user_query)
        callbacks=kwargs['callbacks']
        _, tokens, ind = self.embedding_function._tokenize(user_query, 1)
        for callback in callbacks:
            try:
                if hasattr(callback, "on_embedding"):
                    callback.on_embedding(self.embedding_function.model, tokens)
            except Exception:
                self.logger.error('could not add embedding tokens', exc_info=True)
        docs = self.adv_search_query_collection(query_embedding, tenant_id, lang, filter, k)
        return docs
    
    def advance_search_with_threshold(self, user_query, tenant_id, lang, filter, k, score_threshold_semantic, **kwargs):
        docs = self.advance_search(user_query, tenant_id, lang, filter, k, **kwargs)
        docs = [result for result in docs if result.score>score_threshold_semantic]
        return docs
    

class DocumentPGVectorRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    store: DocumentPGVector
    """VectorStore to use for retrieval."""
    # name: str = "vector"
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
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
            score_threshold = values.get("search_kwargs", {}).get("score_threshold_semantic")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold_semantic` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        return values

    def _get_relevant_documents(
        self, inputs: dict, *, run_manager: CallbackManagerForRetrieverRun,**kwargs
    ) -> List[Document]:
        lang=inputs['lang']
        tenant_id=inputs['tenant_id']
        query=inputs['query']
        # lang=kwargs['lang']
        # tenant_id=kwargs['tenant_id']
        search_kwargs = self.search_kwargs.copy()
        search_kwargs.update({'filter': inputs['filter'],
                         'k' : inputs['k']})

        if self.search_type == "similarity":
            docs = self.store.similarity_search(query, lang, tenant_id, **search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.store.similarity_search_with_relevance_scores(
                    query,lang,tenant_id, **search_kwargs
                )
            )
            # docs = [doc for doc, _ in docs_and_similarities]
            docs = docs_and_similarities
        elif self.search_type == "mmr":
            docs = self.store.max_marginal_relevance_search(
                query, **search_kwargs
            )
        elif self.search_type == "advanced_search":
            token_callback=kwargs['token_callback']
            docs = self.store.advance_search(
                query, tenant_id, lang, **search_kwargs, token_callback = token_callback
            )
        elif self.search_type == "advanced_search_with_threshold":
            token_callback=kwargs['token_callback']

            docs = self.store.advance_search_with_threshold(
                    query, tenant_id, lang, **search_kwargs, token_callback=token_callback
                )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
            self, query: str, lang: str, tenant_id: int, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.store.asimilarity_search(
                query, lang, tenant_id, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.store.asimilarity_search_with_relevance_scores(
                    query, lang, tenant_id, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.store.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    