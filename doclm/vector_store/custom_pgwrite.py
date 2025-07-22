from __future__ import annotations

import asyncio
from asyncio import current_task

import re
import os
import logging
import uuid
from typing import (
    # TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Union,
    Type,
)
from itertools import chain
from typing import Optional, Tuple

from tenacity import retry, wait_random_exponential, stop_after_attempt

from sqlalchemy import delete, func, and_, cast
from sqlalchemy.orm import  scoped_session
from sqlalchemy.dialects.postgresql import to_tsvector, REGCONFIG
from sqlalchemy.ext.asyncio import  async_scoped_session
from sqlalchemy.types import UserDefinedType

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import DocumentPGBase
from .base import async_concurrent_requests

# pylint: disable=W0622,invalid-name
CONCUR_EMBEDDING = int(os.getenv("ASYNC_EMBEDDING_BATCH", '2'))
SUMMARY_WORD_LIMIT = 60


class SingleByteChar(UserDefinedType):
    """1 byte "char" type.

    https://www.postgresql.org/docs/current/datatype-character.html
    #DATATYPE-CHARACTER-SPECIAL-TABLE
    """

    def get_col_spec(self, **kwargs: Any) -> str:
        return '"char"'

    def bind_processor(
        self, dialect
    ) -> Callable[[str], Union[str, bytes]]:

        def process(value: str) -> Union[str, bytes]:
            if dialect.driver == 'asyncpg':
                return value.encode()
            return value

        return process


# VectorStore
class DocumentPGWrite(DocumentPGBase):
    def __init__(self, connection_string: str, embedding_function: Embeddings, pre_delete_collection: bool = False,
                 logger: Optional[logging.Logger] = None, **db_kwargs) -> None:

        super().__init__(connection_string=connection_string,
                         embedding_function=embedding_function,
                         logger=logger,
                         db_kwargs=db_kwargs)
        self.pre_delete_collection = pre_delete_collection


    def create_collection(self, tenant_id:int, application_id:int, collection_name: str, collection_metadata: Optional[dict]) -> None:
        if self.pre_delete_collection:
            self.delete_collection(tenant_id)

        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            self.CollectionStore.get_or_create(
                session, tenant_id, application_id,collection_name, cmetadata=collection_metadata
            )
        self.logger.info('db connection closed')


    def delete_collection(self, tenant_id: int) -> None:
        self.logger.debug("Trying to delete collection")
        self.logger.info('Requesting a connection from db pool')
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection_by_tenant_id(session, tenant_id)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()
        self.logger.info('db connection closed')


    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            connection_string=None,
            **kwargs: Any,
    ) -> DocumentPGWrite:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        if connection_string is None:
            connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def delete_file(self, files: List[Dict]) -> None:
        self.logger.info("Deleting files %s", files)
        self.logger.info('Requesting a connection from db pool')
        tenant_ids= [item[self.tenant_id_tag] for item in files]
        file_ids = [item[self.id_tag] for item in files]

        with scoped_session(self._session_factory)() as session:
            try:
                for file_id, tenant_id in zip(file_ids,tenant_ids):
                    collection = self.get_collection_by_tenant_id(session, tenant_id)
                    if not collection:
                        raise ValueError("Collection not found")
                    
                    # session.delete
                    stmt = delete(self.DocumentStore).where(and_(self.DocumentStore.collection_id == collection.collection_id,
                        self.DocumentStore.id==file_id)
                    )
                session.execute(stmt)
                session.commit()
            except Exception as e_x:
                raise Exception(e_x) from e_x
        self.logger.info('db connection closed')
        self.logger.info("files %s Deleted", files)

    def _pre_process(self, texts, chunck_meta, summary):
        # ids = []
        texts_copy = []
        for text, metadata in zip(list(texts), chunck_meta):
            heading = metadata.get("heading", "")
            texts_copy.append(summary + "\n" + heading + "\n" + text)

        return texts_copy

    def process_file_name_for_robustness(self, file_name: str):
        result_list = []
        result_list.append(file_name)

        extension_removing_regex = r'\.[A-Za-z]{3,5}$'
        extension_free_filename = re.sub(extension_removing_regex, r'',file_name)
        result_list.append(extension_free_filename)

        split_by_non_alphanumerics_regex = r'([A-Za-z\d]+)?'
        split_list = re.findall(split_by_non_alphanumerics_regex, extension_free_filename)
        result_list.extend(split_list)

        alphabetic_regex=r'([A-Za-z]+)'
        numeric_regex=r'(\d+)'
        for literal in split_list:
            result_list.extend(re.findall(alphabetic_regex, literal))      
            result_list.extend(re.findall(numeric_regex, literal))  
           
        result_list = [result for result in list(set(result_list)) if len(result)>0]
        return ' '.join(list(set(' '.join(result_list).split(' '))))

    def add_embeddings(
            self,
            texts: Iterable[str],
            embeddings: List[List[float]],
            tenant_id: int,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        self.logger.info('Requesting a connection from db pool')
        self.logger.info(" Storing data in dataBase")

        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection_by_tenant_id(session, tenant_id)
            if not collection:
                raise ValueError("Collection not found")

            doc = self.DocumentStore(**kwargs['document_data'], collection_id=collection.collection_id)
            doc.document_summary = self.encrypt_text(doc.document_summary)
            doc.encrypted = True if self.encryption_enable else False
            session.add(doc)
            document_name = kwargs['document_data']['name']
            document_name_processed = self.process_file_name_for_robustness(document_name)
            for text, metadata, embedding, U_id in zip(texts, metadatas, embeddings, ids):
                jmeta = {k: metadata.pop(k) for k in
                         list(set(metadata.keys()) - set(self.EmbeddingStore.__table__.columns.keys()))}
                embedding_store = self.EmbeddingStore(
                    doc_id=doc.id,
                    embedding=embedding,
                    document=self.encrypt_text(text),
                    uuid=U_id,
                    jmeta=jmeta,
                    **metadata,
                    keywords_ts_vector= 
                        func.setweight(to_tsvector(func.cast(metadata['lang'], REGCONFIG), text),'A').concat(
                        func.setweight(to_tsvector('usimple', text),'A')).concat(
                        func.setweight(to_tsvector('usimple', document_name_processed),'A')).concat(
                        func.setweight(to_tsvector('usimple', "page " + str(metadata['page'])),'A')).concat(
                        func.setweight(to_tsvector(func.cast(metadata['lang'], REGCONFIG), kwargs['document_data']['document_summary']),'B')).concat(
                        func.setweight(to_tsvector('usimple', kwargs['document_data']['document_summary']),'B')).concat(
                        func.setweight(to_tsvector('usimple', kwargs['document_data'].get('title','')),'B')).concat(
                        func.setweight(to_tsvector(func.cast(metadata['lang'], REGCONFIG), kwargs['document_data'].get('title','')),'B')),
                    adv_search_ts_vector= 
                        func.setweight(to_tsvector('usimple', text),'A').concat(
                            func.setweight(to_tsvector('usimple', kwargs['document_data']['document_summary']),'A')
                        ).concat(
                            func.setweight(to_tsvector('usimple', document_name_processed),'A')
                        ).concat(
                            func.setweight(to_tsvector('usimple', kwargs['document_data'].get('title','')),'B')
                        ).concat(
                            func.setweight(to_tsvector('usimple', kwargs['document_data'].get('author','')),'A')
                        ).concat(
                            func.setweight(to_tsvector('usimple', ' '.join(kwargs['document_data'].get('keyword',[]))),'A')
                        )
                        ,
                    parent_document=doc
                    )
                session.add(embedding_store)

            session.commit()
        self.logger.info('db connection closed')
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            tenant_id: int,
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
        format = kwargs.get('extracted_meta').get('format')
        if len(texts) == 0 and format!='html':
            raise ValueError("No texts received")

        document_meta = kwargs['extracted_meta']
        callbacks = kwargs.get("callbacks", [])

        texts_copy = self._pre_process(
            texts,
            chunck_meta=metadatas,
            summary=' '.join(document_meta[self.summary_tag].split(' ')[:SUMMARY_WORD_LIMIT]),
        )

        # @retry(stop=stop_after_attempt(7), wait=wait_random_exponential(multiplier=0.5, min=10, max=180))
        # def embedding_with_retry(text_for_embedding):
        # #TODO: token should be calculated/get from rest call so we do not returne none
        embeddings = self.embedding_function.embed_documents(texts_copy,)
        _, tokens, ind = self.embedding_function._tokenize(texts_copy, self.embedding_function.chunk_size)
        try:
            for cb in callbacks:
                if hasattr(cb, "on_embedding"):
                    cb.on_embedding(self.embedding_function.model, tokens)
        except Exception:
            self.logger.error('could not add embedding tokens', exc_info=True)
        # return embed
        self.logger.info("generating embedding for %s text docs", len(texts))

        # for i in range(0, len(texts_copy), 16):
        #     self.logger.debug("Generating embedding %s", i)
        #     embeddings += embedding_with_retry(
        #         texts_copy[i: i + 16]
        #     )

        # texts, metadatas = self.encrypt_data(texts, metadatas)
        # _, _ = self.encrypt_data(['_'], [document_meta])
        # document_meta = document_meta_encrypted[0]

        jmeta = {k: document_meta.pop(k) for k in
                 list(set(document_meta.keys()) - set(self.DocumentStore.__table__.columns.keys()))}
        document_meta['jmeta'] = jmeta

        return self.add_embeddings(
            texts=texts, embeddings=embeddings,
            metadatas=metadatas, ids=ids,
            tenant_id=tenant_id,
            document_data=document_meta,
            **kwargs
        )

    # pylint: disable=R0914
    async def aadd_texts(
            self,
            texts: Iterable[str],
            tenant_id: int,
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

        document_meta = kwargs['extracted_meta']
        callbacks = kwargs.get("callbacks", [])

        texts_copy = self._pre_process(
            texts,
            chunck_meta=metadatas,
            summary=' '.join(document_meta[self.summary_tag].split(' ')[:SUMMARY_WORD_LIMIT]),
        )

        self.logger.info("generating embedding for %s text docs", len(texts))

        # @async_concurrent_requests(CONCUR_EMBEDDING)
        # @retry(stop=stop_after_attempt(7), wait=wait_random_exponential(multiplier=0.5, min=10, max=180))
        # async def embedding_with_retry(batch, text_for_embedding):
            #TODO: token should be calculated/get from rest call so we do not return none
            # self.logger.debug("Generating embedding %s", batch)

        embeddings = await self.embedding_function.aembed_documents(texts_copy)
        _, tokens, ind = self.embedding_function._tokenize(texts_copy, self.embedding_function.chunk_size)
        try:
            for cb in callbacks:
                if hasattr(cb, "on_embedding"):
                    cb.on_embedding(self.embedding_function.model, tokens)
        except Exception:
            self.logger.error('could not add embedding tokens', exc_info=True)
            # return embed

        # embeddings = await asyncio.gather(*[
        #     embedding_with_retry(i, texts_copy[i: i + 16])
        #     for i in range(0, len(texts_copy), 16)], return_exceptions=True)


        texts, metadatas = self.encrypt_data(texts, metadatas)
        _, _ = self.encrypt_data(['_'], [document_meta])
        # document_meta = document_meta_encrypted[0]

        jmeta = {k: document_meta.pop(k) for k in
                 list(set(document_meta.keys()) - set(self.DocumentStore.__table__.columns.keys()))}
        document_meta['jmeta'] = jmeta

        return await self.async_add_embeddings(
            texts=texts, embeddings=embeddings,
            metadatas=metadatas, ids=ids,
            tenant_id=tenant_id,
            document_data=document_meta,
            **kwargs
        )

    async def async_add_embeddings(self,
            texts: Iterable[str],
            embeddings: List[List[float]],
            tenant_id: int,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

            Args:
                texts: Iterable of strings to add to the vectorstore.
                embeddings: list of embedding vectors.
                metadatas: List of metadata associated with the texts.
                kwargs: vectorstore specific parameters
            """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        self.logger.info('Requesting a connection from db pool')
        async with async_scoped_session(self._async_session_factory, scopefunc=current_task)() as session:
            # conn.dialect.use_setinputsizes = False
            collection = await self.async_get_collection_by_tenant_id(session, tenant_id)
            if not collection:
                raise ValueError("Collection not found")

            doc = self.DocumentStore(**kwargs['document_data'], collection_id=collection.collection_id)
            # embeddings = chain(*embeddings)  # to join the result

            session.add(doc)

            for text, metadata, embedding, U_id in zip(texts, metadatas, embeddings, ids):
                jmeta = {k: metadata.pop(k) for k in
                         list(set(metadata.keys()) - set(self.EmbeddingStore.__table__.columns.keys()))}

                embedding_store = self.EmbeddingStore(
                    doc_id=doc.id,
                    embedding=embedding,
                    # embedding_hnsw=embedding,
                    document=text,
                    uuid=U_id,
                    jmeta=jmeta,
                    **metadata,
                    keywords_ts_vector=
                    func.setweight(to_tsvector(cast(metadata['lang'], REGCONFIG), text),
                                   cast('A',SingleByteChar)
                                   ).concat(
                        func.setweight(to_tsvector('usimple', text),
                                      cast('A',SingleByteChar)
                        )
                    ).concat(
                        func.setweight(to_tsvector('usimple', kwargs['document_data']['name']),
                                       cast('A',SingleByteChar)
                        )
                    ).concat(
                        func.setweight(to_tsvector('usimple', "page " + str(metadata['page'])),
                                       cast('A',SingleByteChar)
                        )
                    ).concat(
                        func.setweight(to_tsvector(cast(metadata['lang'], REGCONFIG),
                                                   kwargs['document_data']['document_summary']),
                                       cast('B',SingleByteChar))
                    ).concat(
                        func.setweight(to_tsvector('usimple',
                                                   kwargs['document_data']['document_summary']),
                                       cast('B',SingleByteChar)
                        )
                    ).concat(
                        func.setweight(to_tsvector('usimple', kwargs['document_data'].get('title', '')),
                                       cast('B',SingleByteChar)
                        )
                    ).concat(
                        func.setweight(to_tsvector(cast(metadata['lang'], REGCONFIG),
                                                   kwargs['document_data'].get('title', '')),
                                       cast('B',SingleByteChar))
                    ),
                    adv_search_ts_vector=
                    func.setweight(to_tsvector(cast(metadata['lang'], REGCONFIG), text),
                                   cast('A',SingleByteChar)
                                   ),
                    parent_document=doc
                )
                session.add(embedding_store)
            await session.commit()
        self.logger.info('db connection closed')
        return ids


    @classmethod
    def from_texts(
            cls: Type[DocumentPGWrite],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> DocumentPGWrite:
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
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
            cls,
            text_embeddings: List[Tuple[str, List[float]]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> DocumentPGWrite:
        """Construct PGVector wrapper from raw documents and pre-generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres' connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from langchain.vectorstores import PGVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = PGVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
            cls: Type[DocumentPGWrite],
            embedding: Embeddings,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> DocumentPGWrite:
        """
        Get intsance of an existing PGVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
        )

        return store

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
            cls: Type[DocumentPGWrite],
            documents: List[Document],
            embedding: Embeddings,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> DocumentPGWrite:
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
            metadatas=metadatas,
            ids=ids,
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

    
    def update_file_name(self, tenant_id, new_name, file_id, old_name):
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection_by_tenant_id(session, tenant_id)
            if not collection:
                raise ValueError("Collection not found")

            try:
                file_to_rename = session.query(self.DocumentStore). \
                    where(self.DocumentStore.collection_id == collection.collection_id) \
                    .filter(self.DocumentStore.id.in_({file_id}))
                # if file_to_rename.count() == 0:
                #     self.logger.error('could not rename for files %s', file_to_rename.count())
                # raise LookupError(f'Can not rename file not found {old_name}')
                file_to_rename.update({self.DocumentStore.name: new_name},
                                      synchronize_session=False)
            except Exception as e_x:
                session.rollback()
                self.logger.error(e_x, exc_info=True)
                return False
            else:
                session.commit()
                return True

    def close_connection(self):
        try:
            self._conn.close()
        except Exception as e:
            self.logger.error(e, exc_info=True, stack_info=True)

    def if_not_exists(self, files: List[dict]):
        file_ids = [item[self.id_tag] for item in files]
        tenant_ids= [item[self.tenant_id_tag] for item in files]
        files_dont_exist=[]
        for idx,(file_id, tenant_id) in enumerate(zip(file_ids,tenant_ids)):
            with scoped_session(self._session_factory)() as session:
            # with Session(self._conn) as session:
                collection = self.get_collection_by_tenant_id(session, tenant_id)
                if not collection:
                    raise ValueError("Collection not found")

                file_dont_exist = session.query(self.DocumentStore) \
                            .where(self.DocumentStore.collection_id == collection.collection_id) \
                            .filter(self.DocumentStore.id.in_(file_id)).first() is None
                if file_dont_exist:
                    files_dont_exist.append(files[idx])

        return files_dont_exist

    def if_exists(self, file: dict):
        file_id = file[self.id_tag]
        tenant_id= file[self.tenant_id_tag]
        with scoped_session(self._session_factory)() as session:
            collection = self.get_collection_by_tenant_id(session, tenant_id)
            if not collection:
                raise ValueError("Collection not found")

            file_exist = session.query(self.DocumentStore) \
                        .where(self.DocumentStore.collection_id == collection.collection_id) \
                        .filter(self.DocumentStore.id.in_([file_id])).first() is not None

        return file_exist
