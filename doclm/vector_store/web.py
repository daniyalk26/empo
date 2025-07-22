from __future__ import annotations

import asyncio
import os
import logging
import uuid
import contextlib
from datetime import datetime
from pydantic import Field
from typing import (
    # TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    # Optional,
    Tuple,
    Type,
)
from typing import Optional, Tuple

import html2text
import httpx

import sqlalchemy
from bs4 import BeautifulSoup
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter,  MarkdownTextSplitter
from sqlalchemy import delete, func, or_, text
from sqlalchemy.sql import column
from sqlalchemy.orm import Session, scoped_session, aliased
from sqlalchemy.dialects.postgresql import to_tsvector, TSQUERY, REGCONFIG, TEXT
from sqlalchemy.sql.functions import GenericFunction

from langchain_core.embeddings import Embeddings

from ..config import web_supported_languages
# pylint: disable=W0622,invalid-name


from .base import Base as cBase
from .base import WebStore

SERPER_API_KEY = os.environ['SERPER_API_KEY']
doc_logger = logging.getLogger("doclogger")


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


_DEFAULT_COLLECTION_NAME = "langchain"


class TSRank(GenericFunction):
    package = 'full_text'
    name = 'ts_rank'


class TSCDRank(GenericFunction):
    package = 'full_text'
    name = 'ts_rank_cd'

class RegexReplace(GenericFunction):
    package = "full_text"
    name = "regexp_replace"

# VectorStore
class WebPGStore(cBase, VectorStore):
    def __init__(self, connection_string: str, collection_name: str = _DEFAULT_COLLECTION_NAME,
                 collection_metadata: Optional[dict] = None, pre_delete_collection: bool = False,
                 logger: Optional[logging.Logger] = None, **db_kwargs) -> None:

        super().__init__(connection_string, logger, db_kwargs=db_kwargs)
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self.pre_delete_collection = pre_delete_collection
        self.Store = WebStore

    def create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            from ._pgvector_data_models import DecBase
            DecBase.metadata.create_all(self._conn)

    def drop_tables(self) -> None:
        with self._conn.begin():
            from ._pgvector_data_models import DecBase
            DecBase.metadata.drop_all(self._conn)

    def create_collection(self, tenant_id, application_id) -> None:
        if self.pre_delete_collection:
            self.delete_collection()

        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            self.CollectionStore.get_or_create(
                session, tenant_id, application_id,self.collection_name, cmetadata=self.collection_metadata
            )
        self.logger.info('db connection closed')

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()
        self.logger.info('db connection closed')

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    def delete(
            self,
            ids: Optional[List[int]] = None,
            **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
        """
        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )
                stmt = delete(self.EmbeddingStore).where(
                    self.EmbeddingStore.doc_id.in_(ids)
                )
                self.logger.debug('deleting file ')
                session.execute(stmt)
            session.commit()
        self.logger.info('db connection closed')

    def get_collection(self, session: Session) -> Optional["CollectionStore"]:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            collection_name: str = _DEFAULT_COLLECTION_NAME,
            # distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            connection_string: Optional[str] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> WebPGStore:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        if connection_string is None:
            connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            # distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def delete_file(self, files: List[Dict]) -> None:
        self.logger.info("Deleting files %s", files)
        # raise ValueError("Collection not found")
        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            file_ids = [item[self.id_tag] for item in files]

            try:
                stmt = delete(self.Store).where(
                    self.Store.doc_id.in_(file_ids)
                )
                session.execute(stmt)
                session.commit()

            except Exception as e_x:
                session.rollback()
                raise Exception(e_x) from e_x

        self.logger.info('db connection closed')
        self.logger.info("files %s Deleted", files)

    def add_embeddings(
            self,
            texts: Iterable[str],
            # embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of lists of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            # raise ValueError('ids are not given ')
            ids = [str(uuid.uuid1()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for doc, metadata, _id in zip(texts, metadatas, ids):
                doc_url = metadata.pop('link').replace('\x00','')
                date = metadata.pop('date')
                chunk_no=metadata.pop('chunk_no')
                title = metadata.get('title',"").replace('\x00','')
                description = metadata.get('description',"").replace('\x00','')
                embedding_store = self.Store(
                    document=doc.replace('\x00',''),
                    doc_url=doc_url,
                    cmetadata=metadata,
                    chunk_no=chunk_no,
                    date=date,
                    collection_id=collection.collection_id,
                    uuid=_id,
                    ts_col=to_tsvector(metadata["lang"], title + ' ' + description + ' ' + doc.replace('\x00','')),
                )
                session.add(embedding_store)
            session.commit()
        self.logger.info('db connection closed')
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        if len(texts) == 0:
            raise ValueError("No texts received")


        return self.add_embeddings(
            texts=texts,
            # embeddings=embeddings,
            metadatas=metadatas, ids=ids,
            **kwargs
        )

    # pylint: disable=R0914

    def similarity_search(
            self,
            query: str,
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
        # embedding_data = self._get_embeddings_crosslingual(query)
        return self.similarity_search_by_vector(
            query,       # embedding=embedding_data,
            k=k, filter=filter, **kwargs
        )

    def similarity_search_with_score(
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
        # embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(query,
                                                           k=k, filter=filter
                                                           )
        return docs

    def similarity_search_with_score_by_vector(
            self, query,
            # embedding: dict,
            k: int = 20,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to duplication remover algorithm.
            Defaults to 20.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.

        """
        self.logger.info('printing embeddings for query ')
        results = self.__query_collection(query, k=k, filter=filter)
        # embedding_list = [result.WebStore.embedding for result in results]
        results = self._results_to_docs_and_scores(results)
        # results = self._remove_duplicate_docs(results)
        # results = self._diverse_docs_filter(results, embedding_list)
        # results = self._rescale_scores(results)
        # results = self._filter_results(results, k)

        return results

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        url_dict = {}
        count = 1
        docs = []
        for result in results:
            url = result.WebStore.doc_url
            if url not in url_dict:
                url_dict[url] = count
                count+=1

            _id = url_dict[url]
            docs.append(
                ( self.prepare_document(result, _id),  1 - result.score )
            )
        return docs

    def __query_collection(
            self, query,
            k: int = 20,
            filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        lang = filter['lang']
        #todo add urls to search from
        self.logger.info('Requesting a connection from db pool')
        next_chunk=aliased(self.Store)
        previous_chunk=aliased(self.Store)
        try:
            with scoped_session(self._session_factory)() as session:
                collection = self.get_collection(session)
                if not collection:
                    raise ValueError("Collection not found")
                filter_by = self.Store.collection_id == collection.collection_id
                simple_ts_query = func.cast(
                    func.full_text.regexp_replace(
                        func.cast(
                            func.plainto_tsquery(
                                'simple', query
                            ), TEXT
                        ), '&', '|', 'g'
                    ), TSQUERY
                )
                lang_ts_query = func.cast(
                    func.full_text.regexp_replace(
                        func.cast(
                            func.plainto_tsquery(
                                func.cast(
                                    lang, REGCONFIG
                                ), query
                            ), TEXT
                        ), '&', '|', 'g'
                    ), TSQUERY
                )
                filter_clauses = [
                    or_(
                        self.Store.ts_col.op("@@")(simple_ts_query),
                        self.Store.ts_col.op("@@")(lang_ts_query)
                    ),
                    (self.Store.date == datetime.now().date())
                ]
                if filter:
                    if "doc_url" in filter:
                        filter_clauses.append(self.Store.doc_url.in_(filter["doc_url"]))

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)
                results: List[Any] = session.query(self.Store,
                    next_chunk,
                    previous_chunk)\
                    .join(next_chunk, sqlalchemy.and_(self.Store.date==next_chunk.date, self.Store.doc_url==next_chunk.doc_url,self.Store.chunk_no==next_chunk.chunk_no-1), isouter=True)\
                    .join(previous_chunk, sqlalchemy.and_(self.Store.date==previous_chunk.date, self.Store.doc_url==previous_chunk.doc_url,self.Store.chunk_no==previous_chunk.chunk_no+1), isouter=True)\
                    .with_entities(
                        self.Store.document,
                        self.Store.cmetadata,
                        self.Store.doc_url,
                        self.Store.date,
                        self.Store.chunk_no,
                        (1 - (
                            func.full_text.ts_rank(
                                self.Store.ts_col, simple_ts_query, 32
                            ) +
                            func.full_text.ts_rank(
                                self.Store.ts_col, lang_ts_query, 32
                            )
                        ) / 2.0
                            ).label("distance"),
                        next_chunk.document,
                        next_chunk.cmetadata,
                        previous_chunk.document,
                        previous_chunk.cmetadata) \
                    .filter(filter_by) \
                    .order_by(text("distance")) \
                    .limit(k) \
                    .all()
                results = [self.Store.apply_schema(*r) for r in results]
        except Exception as e:
            self.logger.error(e, exc_info=True, stack_info=True)
            raise e
        self.logger.info('db connection closed')
        return results

    def similarity_search_by_vector(
            self,
            query,
            # embedding: dict,
            k: int = 20,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            query: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(query,
            # embedding=embedding,
            k=k, filter=filter,
        )
        # return _results_to_docs(docs_and_scores)
        return docs_and_scores

    def prepare_document(self, result: WebStore, _id: int):
        page_content = result.WebStore.document
        metadata = result.WebStore.cmetadata
        metadata[self.source_tag] = result.WebStore.doc_url
        metadata[self.summary_tag]=metadata.get('title',"")
        metadata[self.chunk_num_tag] = result.WebStore.chunk_no
        metadata[self.id_tag] = _id
        metadata[self.name_tag] = result.WebStore.cmetadata.get('title')
        metadata[self.page_break_tag]=[]
        metadata[self.next_chunk_tag]=result.next_chunk_text
        metadata[self.previous_chunk_tag]=result.previous_chunk_text
        metadata[self.next_chunk_page_breaks_tag]=[]
        metadata[self.previous_chunk_page_breaks_tag]=[]
        metadata[self.page_tag] = 1


        return Document(
            page_content=page_content,
            metadata=metadata,
        )

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    @classmethod
    def from_texts(
            cls: Type[WebPGStore],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            collection_name: str = _DEFAULT_COLLECTION_NAME,
            # distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> WebPGStore:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres' connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            # distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )


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
    def from_documents(
            cls: Type[WebPGStore],
            documents: List[Document],
            embedding: Embeddings,
            collection_name: str = _DEFAULT_COLLECTION_NAME,
            # distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> WebPGStore:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            # distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            **kwargs,
        )

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


    def url_database(self, urls: List[Tuple]):
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection(session)

            t2 = sqlalchemy.values(
                column("doc_url", sqlalchemy.String),
                column("date", sqlalchemy.Date),
                name="myvalues",
            ).data(
                urls
            )
            # q1 = session.query(self.Store.doc_url).filter(self.Store.collection_id == collection.uuid).distinct()
            q1 = session.query(self.Store.doc_url, self.Store.date).filter(self.Store.collection_id == collection.collection_id).distinct()
            final_results = [r for (r,_) in session.query(t2).except_(q1).all()]
        return final_results

    def if_exists(self, files):
        file_ids = [item[self.id_tag] for item in files]
        with scoped_session(self._session_factory)() as session:
            # with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            exists = session.query(self.Store) \
                         .where(self.Store.doc_url == collection.collection_id) \
                         .filter(self.Store.doc_url.in_(file_ids)).first() is not None

        return exists


# TODO: language detection mechanism needs to be unified, hint: search for lang_detector.predict to see the issue.
async def download_webpages(webpages: List[Dict[Any, Any]], lang_detector):

    httpx_client = httpx.AsyncClient()
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    async def load(client, webpage: Dict[Any, Any]):
        metadata =  {key: webpage.get(key) for key in ["link", "title", "favicon", "date"]}
        # process_data
        content = webpage.get("snippet", "")
        try:
            r = await client.get(webpage["link"])
            if r.status_code == 200:
                if webpage["link"].lower().endswith(".pdf"):
                    raise ValueError('PDF not supported')
                elif webpage["link"].endswith(".xml"):
                    parser = "xml"
                else:
                    parser = "html.parser"

                web_text = await r.aread()
                if web_text:
                    content = h.handle(web_text.decode())

                soup = BeautifulSoup(web_text, parser)

                if title := soup.find("title"):
                    metadata["title"] = title.get_text()
                if description := soup.find("meta", attrs={"name": "description"}):
                    metadata["description"] = description.get("content", "No description found.")
                if html := soup.find("html"):
                    if lang := html.get("lang"):
                        metadata["lang"] = lang.split('-')[0]
            if "lang" not in metadata:
                metadata["lang"] = lang_detector.predict(content[:50])
            if metadata["lang"] not in web_supported_languages:
                content = ""
                raise ValueError(f"wrong language document {webpage.get('link')}")

            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            doc_logger.error(e, exc_info=True)
            if content:
                metadata["lang"] = lang_detector.predict(content)
            return Document(page_content=content, metadata=metadata)

    try:
        tasks = [load(httpx_client, w) for w in webpages]
        results = await asyncio.gather(*tasks)
        return results
    finally:
        await httpx_client.aclose()


class WebResearchRetriever(BaseRetriever):
    """`Google Search API` retriever."""
    lang_detector: Any

    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    search: SerpAPIWrapper = Field(..., description="Google Search API Wrapper")
    text_splitter: TextSplitter = Field(MarkdownTextSplitter(),
        description="Text splitter for splitting web pages into chunks",
    )
    search_kwargs: dict = Field(default_factory=dict)

    @staticmethod
    def clean_search_query(query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1:]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean)
        if 'error' in result:
            raise ValueError(result['error'])
        try:
            return result['organic_results']
        except KeyError:
            doc_logger.error("unable to search %s", result)
            return [{}]


    def _get_relevant_documents(
                self,
                inputs: dict,
                *,
                run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:

        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
            :type filter: object
            :param config:
            :param input:
        """
        # search_kwargs = {'filter': filter or {}}
        # if not filter:
        #     filter = {}
        filter=inputs.get('filter',{})
        if 'lang' not in filter:
            if 'lang' in inputs:
                filter['lang']=inputs['lang']
        # Get urls
        query=inputs['query']
        urls_to_look = []
        urls_meta_from_serp = {}

        doc_logger.info("%s: searching query `%s`: ", datetime.now(), query)

        search_results = self.search_tool(query)
        doc_logger.info(f"Search engine results: {search_results}")

        for res in search_results:
            if link := res.get("link"):
                urls_to_look.append(link)
                res['date'] = datetime.now().date()
                # try:
                #     if date := res.get('date'):
                #         res['date'] = datetime.strptime(date, '%b %d, %Y').date()
                #     else:
                #         res['date'] = datetime.now().date()
                #
                # except ValueError:
                #     res['date'] = datetime.now().date()

                urls_meta_from_serp[link] = res

        # Relevant urls
        urls_date = set([(k, val['date']) for k, val in urls_meta_from_serp.items()])
        # urls_date = list(set([val['date'] for k, val in urls_meta_from_serp.items()]))
        # Check for any new urls that we have not processed
        new_urls=[]
        if len(urls_date)>0:
            new_urls = self.vectorstore.url_database(urls_date)

        doc_logger.info(f"New URLs to load: {new_urls}")
        # Load, split, and add new urls to vectorstore
        if new_urls:
            try:
                loop = asyncio.get_event_loop()
                docs = loop.run_until_complete(asyncio.gather(
                    download_webpages(
                        [urls_meta_from_serp[url] for url in new_urls],
                        self.lang_detector
                    )))[0]
                doc_logger.debug("downloaded urls")
            except RuntimeError:
                doc_logger.debug("urls download failed, falling back")
                docs = asyncio.run(download_webpages(
                        [urls_meta_from_serp[url] for url in new_urls],
                    self.lang_detector
                    ))
            docs_splitted=[]
            for doc in docs:
                doc_splitted = self.text_splitter.split_documents([doc])
                for chunk_num, split in enumerate(doc_splitted, start=1):
                    split.metadata['chunk_no']=chunk_num
                    docs_splitted.append(split)

            # docs = self.text_splitter.split_documents([x for x in docs if x.page_content])
            if docs_splitted:
                self.vectorstore.add_documents(docs_splitted)

        # Search for relevant splits
        # TODO: make this async
        doc_logger.info("Grabbing most relevant splits from urls...")

        filter["doc_url"] = urls_meta_from_serp.keys()

        docs_and_scores = self.vectorstore.similarity_search(query, filter=filter, **self.search_kwargs)

        # Get unique docs
        # unique_documents_dict = {
        #     (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        # }
        # unique_documents = list(unique_documents_dict.values())
        return docs_and_scores

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError


def web_retriever(vectorstore, search_region,
                  lang_detector,
                  number_of_searches=5,
                  language="en", **kwargs):

    params = {
        "num":number_of_searches,
        "gl": search_region,
        "hl": language,
    }

    search = SerpAPIWrapper(params=params, serpapi_api_key=SERPER_API_KEY, )
    return WebResearchRetriever(
        vectorstore=vectorstore,
        search=search,
        lang_detector=lang_detector,
        text_splitter = MarkdownTextSplitter(),
        **kwargs)
