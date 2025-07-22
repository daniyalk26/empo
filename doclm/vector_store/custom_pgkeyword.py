from __future__ import annotations

# import asyncio
# import contextlib
import os
import enum
import logging
import contextlib
from math import exp
import warnings
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

import sqlalchemy
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import column, text, case

from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy import func, or_
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY, TEXT, ts_headline,to_tsvector

from langchain_core.documents import Document
from langchain.utils.math import cosine_similarity
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
    )
from langchain_core.pydantic_v1 import Field, root_validator
# import sqlalchemy.dialects.postgresql
# from ._pgvector_data_models import DocumentStore, CollectionStore, EmbeddingStore, AdvSearchReadSchema
# pylint: disable=W0622,invalid-name



from .base import DocumentPGBase


sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")



def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class RankingStrategy(str, enum.Enum):
    """Enumerator of the Ranking strategies."""

    SIMPLE_RANK = "simple"
    CD_RANK = "cd"


DEFAULT_RANKING_STRATEGY = RankingStrategy.CD_RANK

class TSRank(GenericFunction):
    package = 'full_text'
    name = 'ts_rank'
    inherit_cache=True
class TSCDRank(GenericFunction):
    package = 'full_text'
    name = 'ts_rank_cd'
    inherit_cache=True

class StripTSVector(GenericFunction):
    package = 'full_text'
    name = 'strip'

# VectorStore
class DocumentPGKeyword(DocumentPGBase):
    def __init__(
            self,
            connection_string: str,
            ranking_strategy: RankingStrategy = DEFAULT_RANKING_STRATEGY,
            logger: Optional[logging.Logger] = None,
            relevance_score_fn: Optional[Callable[[float], float]] = None,
            **db_kwargs
    ) -> None:
        super().__init__(connection_string, logger, db_kwargs, None)
        self.connection_string = connection_string
        self._ranking_strategy = ranking_strategy
        self.logger = logger or logging.getLogger("doclogger")
        self.override_relevance_score_fn = relevance_score_fn
        # self.__post_init__(**db_kwargs)

    # def __post_init__(
    #         self,**db_kwargs
    # ) -> None:
    #     """
    #     Initialize the store.
    #     """
    #     # self._conn = self.connect(**db_kwargs)
        # self._session_factory = self.create_session_factory()

        # self.CollectionStore = CollectionStore
        # self.DocumentStore = DocumentStore
        # self.EmbeddingStore = EmbeddingStore
        # self.AdvSearchReadSchema = AdvSearchReadSchema

    def connect(self, **db_kwargs) -> sqlalchemy.engine:
        engine = sqlalchemy.create_engine(self.connection_string, **db_kwargs)
        return engine

    def create_session_factory(self):
        return sessionmaker(self._conn)

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    # def get_collection_by_name(self, session: Session, collection_name: str) -> Optional["CollectionStore"]:
    #     return self.CollectionStore.get_by_name(session, collection_name)
    #
    # def get_collection_by_tenant_id(self, session: Session, tenant_id:int) -> Optional["CollectionStore"]:
    #     return self.CollectionStore.get_by_tenant_id(session, tenant_id)
    
    def keyword_search(
            self,
            query: str,
            lang: str,
            tenant_id:int,
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
        docs_and_scores = self.keyword_search_with_score_by_query(
            query=query, lang=lang, tenant_id=tenant_id, k=k, filter=filter,

        )
        # return _results_to_docs(docs_and_scores)
        return docs_and_scores

    def keyword_search_with_score(
            self,
            query: str,
            k: int = 20,
            filter: Optional[dict] = None,

    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        docs = self.keyword_search_with_score_by_query(
            query=query, k=k, filter=filter
        )
        return docs

    @property
    def ranking_strategy(self) -> Any:
        if self._ranking_strategy == RankingStrategy.SIMPLE_RANK:
            return func.full_text.ts_rank
        elif self._ranking_strategy == RankingStrategy.CD_RANK:
            return func.full_text.ts_rank_cd
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._ranking_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in RankingStrategy])}."
            )

    def keyword_search_with_score_by_query(
            self,
            query: str,
            lang:str,
            tenant_id:int,
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

        results = self.__query_collection(query=query, lang=lang, tenant_id=tenant_id, k=k, filter=filter)
        embedding_list = [result.EmbeddingStore.embedding for result in results]
        results = self._results_to_docs_and_scores(results)
        # results = self._remove_duplicate_docs(results)
        results = self._diverse_docs_filter(results, embedding_list)
        # results = self._rescale_scores(results)
        # results = self._filter_results(results, k)
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

    def __query_collection(
            self,
            query: str,
            lang: str,
            tenant_id:int,
            k: int = 20,
            filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self.logger.info('Requesting a connection from db pool')
        current_chunk=aliased(self.EmbeddingStore)
        next_chunk=aliased(self.EmbeddingStore)
        previous_chunk=aliased(self.EmbeddingStore)
        try:
            with scoped_session(self._session_factory)() as session:
                collection = self.get_collection_by_tenant_id(session, tenant_id=tenant_id)
                if not collection:
                    raise ValueError("Collection not found")
                collection_filter = [self.DocumentStore.collection_id == collection.collection_id]
                concatenated_ts_query=func.cast(
                    func.regexp_replace(
                        func.cast(
                            func.plainto_tsquery(
                                func.cast(
                                    lang,REGCONFIG
                                    ),query
                                ).concat(func.plainto_tsquery(
                                'usimple',query
                                )),TEXT
                            ),'&','|','g'
                        ),TSQUERY
                    )

                unnested_query_cte = session.query(
                    func.unnest(
                        to_tsvector(func.cast(
                            lang,REGCONFIG
                                ),
                        query).concat(
                            to_tsvector(
                                'usimple',query
                                )
                            )
                        ).table_valued("lexeme",name="unnested_query").render_derived()).cte("unnested_query")
                keyword_filter_clauses = [
                        current_chunk.keywords_ts_vector.op("@@")(concatenated_ts_query),
                        current_chunk.keywords_ts_vector.op("@@")(func.cast(unnested_query_cte.c.lexeme,TSQUERY))
                    ]
                if filter:
                    if "id" in filter:
                        collection_filter.append(current_chunk.doc_id.in_(filter["id"]))
                term_frequency_cte_filter = sqlalchemy.and_(*collection_filter, *keyword_filter_clauses)
                term_frequency_inner_query=session.query(
                    current_chunk.uuid, 
                    unnested_query_cte.c.lexeme.label("QUERY_KEYWORD"),
                    func.count().over(partition_by=unnested_query_cte.c.lexeme).label("DOCUMENT_FREQUENCY"),
                    func.full_text.ts_rank(current_chunk.keywords_ts_vector,func.cast(unnested_query_cte.c.lexeme,TSQUERY)).label("TERM_FREQUENCY")
                    ).select_from(current_chunk)\
                        .join(self.DocumentStore,  sqlalchemy.and_(collection.collection_id == self.DocumentStore.collection_id,
                                                       current_chunk.doc_id == self.DocumentStore.id))\
                        .filter(term_frequency_cte_filter).subquery().lateral()
                
                term_freq_cte=session.query(
                    term_frequency_inner_query
                        ).select_from(unnested_query_cte,term_frequency_inner_query).cte("term_freq_cte")
                
                constants_cte=session.query(
                    func.count().label("NUMBER_OF_CHUNKS")).select_from(current_chunk)\
                        .join(self.DocumentStore,  sqlalchemy.and_(collection.collection_id == self.DocumentStore.collection_id,
                                                       current_chunk.doc_id == self.DocumentStore.id))\
                        .filter(
                            sqlalchemy.and_(
                                *collection_filter,
                                current_chunk.keywords_ts_vector != None)
                            ).cte("constants")

                denormalized_cte=session.query(term_freq_cte, constants_cte).cte("denormalized_cte")
                bm25_score_inner_subquery = session.query(denormalized_cte.c.uuid,
                                               func.sum((denormalized_cte.c.TERM_FREQUENCY/(denormalized_cte.c.TERM_FREQUENCY+1)
                                                         * (
                                                             func.ln(1+((denormalized_cte.c.NUMBER_OF_CHUNKS-denormalized_cte.c.DOCUMENT_FREQUENCY+0.5)/(denormalized_cte.c.DOCUMENT_FREQUENCY+0.5)))
                                                         ))).label("BM25_SCORE")).select_from(denormalized_cte).group_by(denormalized_cte.c.uuid).subquery()
                bm25_score_cte=session.query(bm25_score_inner_subquery.c.uuid,
                                             bm25_score_inner_subquery.c.BM25_SCORE).select_from(bm25_score_inner_subquery).order_by(sqlalchemy.desc(bm25_score_inner_subquery.c.BM25_SCORE)).limit(k).cte("bm25_score_cte")
            
                results =session.query(
                    current_chunk.embedding,
                    current_chunk.document,
                    current_chunk.jmeta,
                    current_chunk.page,
                    current_chunk.uuid,
                    current_chunk.doc_id,
                    current_chunk.lang,
                    current_chunk.chunk_num,
                    current_chunk.chunk_len_in_chars,
                    self.DocumentStore.document_summary,
                    self.DocumentStore.name,
                    1-(bm25_score_cte.c.BM25_SCORE/(bm25_score_cte.c.BM25_SCORE+1)).label("distance"),
                    self.DocumentStore.encrypted,

                    self.DocumentStore.original_format,
                    self.DocumentStore.format,
                    previous_chunk.document,
                    previous_chunk.jmeta,
                    next_chunk.document,
                    next_chunk.jmeta,
                    func.coalesce(self.DocumentStore.author, 'Unknown').label("author")
                ).select_from(current_chunk) \
                    .join(bm25_score_cte, bm25_score_cte.c.uuid == current_chunk.uuid) \
                    .join(self.DocumentStore, sqlalchemy.and_(collection.collection_id == current_chunk.collection_id,
                                                              self.DocumentStore.id == current_chunk.doc_id)) \
                    .join(previous_chunk, sqlalchemy.and_(previous_chunk.collection_id == current_chunk.collection_id,
                                                          previous_chunk.doc_id == current_chunk.doc_id,
                                                          current_chunk.chunk_num == previous_chunk.chunk_num + 1),
                          isouter=True) \
                    .join(next_chunk, sqlalchemy.and_(next_chunk.collection_id == current_chunk.collection_id,
                                                      next_chunk.doc_id == current_chunk.doc_id,
                                                      current_chunk.chunk_num == next_chunk.chunk_num - 1),
                          isouter=True) \
                    .order_by(sqlalchemy.desc(bm25_score_cte.c.BM25_SCORE)).all()
            
            results = [self.EmbeddingStore.apply_schema(*r) for r in results]
        except Exception as e:
            self.logger.error(e, exc_info=True,stack_info=True)
            raise e
        self.logger.info('db connection closed')
        return results

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
        if self._ranking_strategy == RankingStrategy.SIMPLE_RANK:
            return self._simple_rank_relevance_score_fn
        elif self._ranking_strategy == RankingStrategy.CD_RANK:
            return self._cd_rank_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._ranking_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
        
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
        # embedding = self.embedding_function.embed_query(query)
        docs = self.keyword_search_with_score_by_query(
            query=query, lang=lang, tenant_id=tenant_id, k=k, filter=filter
        )
        return docs
    
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
    
    def keyword_search_with_relevance_scores(
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
        score_threshold = kwargs.pop("score_threshold_keyword", None)

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

    def _get_retriever_tags(self) -> List[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        # if self.embeddings:
        #     tags.append(self.embeddings.__class__.__name__)
        return tags
    
    def as_retriever(self, **kwargs: Any) -> DocumentPGKeywordRetriever:
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
                        score_threshold_keyword: Minimum relevance threshold
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
                search_kwargs={'score_threshold_keyword': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            )
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return DocumentPGKeywordRetriever(store=self, tags=tags, **kwargs)

    def close_connection(self):
        try:
            self._conn.close()
        except Exception as e:
            self.logger.error(e, exc_info=True, stack_info=True)

    
    def adv_search_query_collection(self, query, tenant_id, lang, filter, k):
        self.logger.info('Requesting a connection from db pool')
        current_chunk=aliased(self.EmbeddingStore)
        try:
            with scoped_session(self._session_factory)() as session:
                collection = self.get_collection_by_tenant_id(session, tenant_id=tenant_id)
                if not collection:
                    raise ValueError("Collection not found")
                collection_filter = [current_chunk.collection_id == collection.collection_id]
                usimple_ts_query=func.cast(
                    func.regexp_replace(
                        func.cast(func.plainto_tsquery(
                                'usimple',query
                                ),TEXT
                            ),'&','|','g'
                        ),TSQUERY
                    )
                
                unnested_query_cte = session.query(
                    func.unnest(
                        to_tsvector(
                            'usimple',query
                            )
                        ).table_valued("lexeme",name="unnested_query").render_derived()).cte("unnested_query")
                keyword_filter_clauses = [
                        current_chunk.keywords_ts_vector.op("@@")(usimple_ts_query),
                        current_chunk.keywords_ts_vector.op("@@")(func.cast(unnested_query_cte.c.lexeme,TSQUERY))
                    ]
                if filter:
                    if "id" in filter:
                        collection_filter.append(current_chunk.doc_id.in_(filter["id"]))
                proximity_cte_filter = sqlalchemy.and_(*collection_filter, *keyword_filter_clauses[:1])
                proximity_cte=session.query(
                    current_chunk.uuid, 
                    func.full_text.ts_rank_cd(current_chunk.keywords_ts_vector,usimple_ts_query,32).label("PROXIMITY_SCORE")
                ).select_from(current_chunk)\
                    .filter(proximity_cte_filter).cte("proximity_cte")

                term_frequency_cte_filter = sqlalchemy.and_(*collection_filter, *keyword_filter_clauses)
                term_frequency_inner_query=session.query(
                    current_chunk.uuid, 
                    current_chunk.doc_id,
                    unnested_query_cte.c.lexeme.label("QUERY_KEYWORD"),
                    func.count().over(partition_by=unnested_query_cte.c.lexeme).label("DOCUMENT_FREQUENCY"),
                    func.full_text.ts_rank(current_chunk.keywords_ts_vector,func.cast(unnested_query_cte.c.lexeme,TSQUERY)).label("TERM_FREQUENCY")
                    ).select_from(current_chunk)\
                        .filter(term_frequency_cte_filter).subquery().lateral()
                
                term_freq_cte=session.query(
                    term_frequency_inner_query
                        ).select_from(unnested_query_cte,term_frequency_inner_query).cte("term_freq_cte")
                constants_cte=session.query(
                    func.count().label("NUMBER_OF_CHUNKS")).select_from(current_chunk)\
                        .filter(
                            sqlalchemy.and_(
                                *collection_filter,
                                current_chunk.keywords_ts_vector != None)
                            ).cte("constants")
                denormalized_cte=session.query(term_freq_cte, constants_cte).cte("denormalized_cte")
                bm25_score_inner_subquery = session.query(
                    denormalized_cte.c.uuid,
                    denormalized_cte.c.doc_id,
                    func.sum(
                        (
                            denormalized_cte.c.TERM_FREQUENCY/(denormalized_cte.c.TERM_FREQUENCY+1)
                            * (
                                func.ln(1+((denormalized_cte.c.NUMBER_OF_CHUNKS-denormalized_cte.c.DOCUMENT_FREQUENCY+0.5)/(denormalized_cte.c.DOCUMENT_FREQUENCY+0.5)))
                            )
                        )
                    ).label("BM25_SCORE")).select_from(denormalized_cte).group_by(denormalized_cte.c.uuid,denormalized_cte.c.doc_id).subquery()
                bm25_score_subquery=session.query(
                    bm25_score_inner_subquery.c.uuid,
                    bm25_score_inner_subquery.c.doc_id,
                    ((0.8*(bm25_score_inner_subquery.c.BM25_SCORE/(bm25_score_inner_subquery.c.BM25_SCORE+1)))+(0.2*func.coalesce(proximity_cte.c.PROXIMITY_SCORE,0))).label("BM25_SCORE"),
                    func.row_number().over(partition_by=bm25_score_inner_subquery.c.doc_id,order_by=sqlalchemy.desc((0.8*bm25_score_inner_subquery.c.BM25_SCORE)+(0.2*func.coalesce(proximity_cte.c.PROXIMITY_SCORE,0)))).label("CHUNK_RANK_PER_DOC")
                ).select_from(bm25_score_inner_subquery).join(proximity_cte,bm25_score_inner_subquery.c.uuid == proximity_cte.c.uuid,isouter=True).subquery()
                bm25_score_cte=session.query(
                    bm25_score_subquery.c.uuid,
                    bm25_score_subquery.c.BM25_SCORE).select_from(
                        bm25_score_subquery
                    ).filter(bm25_score_subquery.c.CHUNK_RANK_PER_DOC==1).order_by(
                        sqlalchemy.desc(bm25_score_subquery.c.BM25_SCORE)
                    ).limit(
                        k
                    ).cte("bm25_score_cte")
                results=session.query(current_chunk.document,
                    self.DocumentStore.document_summary,
                    current_chunk.jmeta,
                    current_chunk.page,
                    current_chunk.doc_id,
                    current_chunk.lang,
                    (1-bm25_score_cte.c.BM25_SCORE).label("distance"),
                    self.DocumentStore.encrypted).select_from(
                    bm25_score_cte
                ).join(current_chunk,current_chunk.uuid==bm25_score_cte.c.uuid).join(
                    self.DocumentStore, sqlalchemy.and_(self.DocumentStore.collection_id ==current_chunk.collection_id, self.DocumentStore.id==current_chunk.doc_id)
                ).all()
            results = [self.AdvSearchReadSchema.apply_schema(*r) for r in results]
            for result in results:
                if result.encrypted=='true':
                    result.chunk_text,_ = self.decrypt_data(result.chunk_text, {})
                    result.doc_summary,_=self.decrypt_data(result.doc_summary, {})
                    result.encrypted='false'
        except Exception as e:
            self.logger.error(e, exc_info=True,stack_info=True)
            raise e
        self.logger.info('db connection closed')
        if len(results)>0:
            results_spllited = self.adv_search_split_page_wise(results)
            highlights = self.adv_search_highlight_text([(result.chunk_text, result.lang, result.doc_id, result.page_no) for result in results_spllited], query, lang)
            highlights.sort(key=lambda x: (x[0],x[1]))

            final_highlighted_text_dict={}
            final_page_no_dict={}

            for highlight in highlights:
                if len(highlight[2]):
                    if highlight[0] not in final_highlighted_text_dict:
                        final_highlighted_text_dict[highlight[0]]=highlight[2]
                        final_page_no_dict[highlight[0]]=highlight[1]
                    else:
                        final_highlighted_text_dict[highlight[0]]+= ' ... ' + highlight[2]
        for result in results:
            result.page_no=final_page_no_dict.get(result.doc_id,result.page_no)
            result.highlighted_text=final_highlighted_text_dict.get(result.doc_id, result.highlighted_text)
            result.search_type='keyword'
            result.score = 1-result.score
        results.sort(key=lambda x:x.score, reverse=True)
        return results
    
    def adv_search_split_page_wise(self, merged_chunks: List[Any]):
        _page_wise_split_chunk_list = []
        for merged_chunk in merged_chunks:
            page_breaks = merged_chunk.jmeta.get('page_breaks_char_offset',[])
            doc_summary=merged_chunk.doc_summary
            if not len(page_breaks):
                page_breaks = [len(merged_chunk.chunk_text)]
            if page_breaks[-1] != len(merged_chunk.chunk_text):
                page_breaks += [len(merged_chunk.chunk_text)]
            page_starting_offset_in_chars = 0
            for idx, pg_break_end_offset in enumerate(page_breaks):
                chunk_text= merged_chunk.chunk_text[page_starting_offset_in_chars: pg_break_end_offset]
                jmeta = merged_chunk.jmeta.copy()
                page_no= str(int(merged_chunk.page_no) + idx)
                doc_id= merged_chunk.doc_id
                lang= merged_chunk.lang
                score= merged_chunk.score
                encrypted= merged_chunk.encrypted
                _page_wise_split_chunk_list.append(
                    self.AdvSearchReadSchema(
                        chunk_text=chunk_text, 
                        doc_summary=doc_summary,
                        jmeta=jmeta, 
                        page_no=page_no,
                        doc_id=doc_id,
                        lang=lang,
                        score=score,
                        encrypted=encrypted
                        )
                    )
                page_starting_offset_in_chars = pg_break_end_offset
        return _page_wise_split_chunk_list
        

    def adv_search_highlight_text(self, document_tuples: List[Tuple], query:str, lang):
        StartSel='<span>'
        ModifiedStartSel='<span style="background-color: #7BF3B9;">'
        StopSel='</span>'
        
                
        with scoped_session(self._session_factory)() as session:
            data_to_highlight = session.query(sqlalchemy.values(
                column("document", sqlalchemy.String),
                column("document_lang", sqlalchemy.String),
                column("doc_id", sqlalchemy.Integer),
                column("page_no", sqlalchemy.String),
                column("query", sqlalchemy.String),
                column("lang", sqlalchemy.String),
                name="data_to_highlight",
            ).data(
                [(text, text_lang, doc_id, page_no, query, lang) for text, text_lang, doc_id, page_no in document_tuples]
            )).subquery()
            highlighted_text = session.query(data_to_highlight)\
            .with_entities(
                        data_to_highlight.c.doc_id,
                        data_to_highlight.c.page_no,
                        ts_headline(
                            func.cast(
                                data_to_highlight.c.document_lang,REGCONFIG
                                ),
                            data_to_highlight.c.document,
                            func.cast(
                                func.regexp_replace(
                                    func.cast(
                                        func.plainto_tsquery(
                                            func.cast(
                                                data_to_highlight.c.document_lang,REGCONFIG
                                                ),query
                                            ),TEXT
                                        ),'&','|','g'
                                    ),TSQUERY
                        )
                    ,text(f'\'MinWords=1,MaxFragments=3,StartSel={StartSel},StopSel={StopSel},HighlightAll=true\'')).label('highlight')
            )\
            .subquery()
            final_highlights=session.query(
                highlighted_text
            ).with_entities(
                highlighted_text.c.doc_id,
                highlighted_text.c.page_no,
                case(
                    
                        (
                            highlighted_text.c.highlight.like('%'+StopSel+'%'),
                            func.regexp_replace(highlighted_text.c.highlight,StartSel,ModifiedStartSel,'g')
                        )
                    ,
                    else_=text("''")
                ).label('final_highlight')
            ).all()
        return final_highlights


    def advance_search(self, user_query, tenant_id, lang, filter, k, **kwargs):
        docs = self.adv_search_query_collection(user_query,tenant_id, lang, filter, k)
        return docs

    def advance_search_with_threshold(self, user_query, tenant_id, lang, filter, k, score_threshold_keyword, **kwargs):
        docs = self.advance_search(user_query,tenant_id, lang, filter, k)
        docs = [result for result in docs if result.score>score_threshold_keyword]
        return docs
    

class DocumentPGKeywordRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    store: DocumentPGKeyword
    """VectorStore to use for retrieval."""
    # name: str = "keyword"
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
            score_threshold = values.get("search_kwargs", {}).get("score_threshold_keyword")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold_keyword` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        return values

    def _get_relevant_documents(
        self, inputs: dict, *, run_manager: CallbackManagerForRetrieverRun, **kwargs 
    ) -> List[Document]:
        lang=inputs['lang'] 
        tenant_id=inputs['tenant_id']
        query=' '.join(inputs['query'].split(' ')[:40]+inputs['query'].split(' ')[-30:])
        # lang=kwargs['lang'] 
        # tenant_id=kwargs['tenant_id']
        search_kwargs = self.search_kwargs.copy()
        search_kwargs.update({'filter': inputs['filter'],
                         'k' : inputs['k']})


        if self.search_type == "similarity":
            docs = self.store.keyword_search(query, lang, tenant_id, **search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.store.keyword_search_with_relevance_scores(
                    query,lang,tenant_id, **search_kwargs
                )
            )
            docs = docs_and_similarities
        elif self.search_type == "advanced_search":
            # filter = kwargs['filter']
            # k = kwargs['k']
            docs_and_similarities = (
                self.store.advance_search(
                    query, tenant_id, lang, **search_kwargs
                )
            )
            docs = docs_and_similarities
        elif self.search_type == "advanced_search_with_threshold":
            docs_and_similarities = (
                self.store.advance_search_with_threshold(
                    query, tenant_id, lang, **search_kwargs,
                )
            )
            docs = docs_and_similarities
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.store.akeyword_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.store.akeyword_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
