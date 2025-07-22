"""
 Implements abstract method for vector stores
"""
import logging

import functools
from typing import Dict, Iterable, List, Tuple, Callable, Generator, Optional
import contextlib
from asyncio import BoundedSemaphore

import sqlalchemy
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ._pgvector_data_models import DocumentStore, CollectionStore, EmbeddingStore, WebStore, AdvSearchReadSchema

from ..schema import Schema

log = logging.getLogger("doclogger")
log.disabled = False

__all__ = ['Base', 'async_concurrent_requests', 'DocumentPGBase', 'WebStore']
async_mode = False

def async_concurrent_requests(requests=1):
    semaphore = BoundedSemaphore(requests)

    def inner(func):

        @functools.wraps(func)
        async def wrap(*args, **kwargs):
            await semaphore.acquire()
            try:
                val = await func(*args, **kwargs)
                return val

            except Exception as e:
                log.error(e, exc_info=True)
                raise e
            finally:
                semaphore.release()

        return wrap

    return inner


class Base(Schema):

    def __init__(self, connection_string: str,
                 logger: Optional[logging.Logger] = None,
                 **db_kwargs):
        self.connection_string = connection_string
        self.logger = logger or logging.getLogger("doclogger")
        self.CollectionStore = CollectionStore
        self.__post_init__(**db_kwargs)

    def __post_init__(
            self, db_kwargs
    ) -> None:
        """
        Initialize the store.
        """
        self._conn = self.connect(**db_kwargs)
        self._session_factory = self.create_session_factory()
        if async_mode:
            self._async_session_factory = self.create_async_session()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    def connect(self, **db_kwargs) -> sqlalchemy.engine:
        engine = sqlalchemy.create_engine(self.connection_string, **db_kwargs)
        return engine

    def create_session_factory(self):
        return sessionmaker(self._conn)

    def create_async_session(self, **db_kwargs):
        async_session_factory = create_async_engine(self.connection_string, **db_kwargs)
        return async_sessionmaker(async_session_factory, expire_on_commit=False)

    def if_exists(self, file: list):
        raise NotImplementedError('not implemented')

    def delete_file(self, files: List[Dict]) -> None:
        raise NotImplementedError("delete_file method must be implemented by subclass.")

    def encrypt_data(self, texts: Iterable[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict]]:
        log.info('Encrypting data')

        if self.encryption_enable.lower() == 'true':
            texts = [Schema.encrypt(text) for text in texts]
            for metadata in metadatas:
                for key in self.encrypt_meta_keys:
                    key_data = metadata.get(key, None)
                    if not key_data:
                        log.warning('%s not found in metadata to encrypt', key)
                        continue
                    metadata[key] = Schema.encrypt(key_data)
                metadata[Schema.encrypt_tag] = True
        return texts, metadatas

    def encrypt_text(self, text: str) -> str:
        log.info('Encrypting text')
        return Schema.encrypt(text)

    def decrypt_data(self, text: str, metadata: Dict):
        log.info('Decrypting data')

        if self.encryption_enable.lower() == 'true':
            text = Schema.decrypt(text)

            for key in self.encrypt_meta_keys:
                key_data = metadata.get(key)
                if not key_data:
                    log.warning('%s not present in metadata to decrypt', key)
                    continue
                metadata[key] = Schema.decrypt(key_data)

        return text, metadata
    
    def decrypt_text(self, text: str) -> str:
        log.info('Decrypting text')
        return Schema.decrypt(text)


class DocumentPGBase(Base):

    def __init__(self, connection_string: str, logger, db_kwargs, embedding_function: Embeddings= None):

        super().__init__(connection_string=connection_string, logger=logger, db_kwargs=db_kwargs)

        self.DocumentStore = DocumentStore
        self.EmbeddingStore = EmbeddingStore
        self.embedding_function = embedding_function
        self.AdvSearchReadSchema = AdvSearchReadSchema

    def get_embedding_model_name(self):
        if self.embedding_function:
            return self.embedding_function.model
        raise ValueError('no embedding model found')

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def get_collection_by_name(self, session: Session, collection_name: str) -> Optional["CollectionStore"]:
        return self.CollectionStore.get_by_name(session, collection_name)

    def get_collection_by_tenant_id(self, session: Session, tenant_id: int) -> Optional["CollectionStore"]:
        return self.CollectionStore.get_by_tenant_id(session, tenant_id)

    async def async_get_collection_by_tenant_id(self, session: Session, tenant_id: int) -> Optional[
        "CollectionStore"]:
        return await self.CollectionStore.async_get_by_tenant_id(session, tenant_id)

    def _prepare_document(self, result):
        # TODO: Need to make this generic so that we do not have to update here every time me make an update to the schema.
        if result.format.lower() in ['xlsx', 'csv', 'xls']:
            result.next_chunk_jmeta = {}
            result.previous_chunk_jmeta = {}
            result.previous_chunk_text = ""
            result.next_chunk_text = ""

        page_content = result.EmbeddingStore.document
        metadata: dict = result.EmbeddingStore.jmeta

        prev_chunk_heading = result.previous_chunk_jmeta.get(self.chunk_heading_tag, '')
        current_chunk_heading = metadata.get(self.chunk_heading_tag, '')
        next_chunk_heading = result.next_chunk_jmeta.get(self.chunk_heading_tag, '')

        metadata[self.page_break_tag] = metadata.get(self.page_break_tag, [])
        metadata[self.name_tag] = result.document_name
        metadata[self.id_tag] = result.EmbeddingStore.doc_id
        metadata[self.lang_param] = result.EmbeddingStore.lang
        metadata[self.chunk_num_tag] = result.EmbeddingStore.chunk_num
        metadata[self.chunk_len_in_chars_tag] = result.EmbeddingStore.chunk_len_in_chars
        metadata[self.page_tag] = result.EmbeddingStore.page
        metadata[self.summary_tag] = result.document_summary
        metadata[self.author_tag] = result.author

        if prev_chunk_heading == current_chunk_heading:
            metadata[self.previous_chunk_tag] = result.previous_chunk_text
            metadata[self.previous_chunk_page_breaks_tag] = result.previous_chunk_jmeta.get(self.page_break_tag, [])
        else:
            metadata[self.previous_chunk_tag] = ''
            metadata[self.previous_chunk_page_breaks_tag] = []

        if next_chunk_heading == current_chunk_heading:
            metadata[self.next_chunk_tag] = result.next_chunk_text
            metadata[self.next_chunk_page_breaks_tag] = result.next_chunk_jmeta.get(self.page_break_tag, [])
        else:
            metadata[self.next_chunk_tag] = ''
            metadata[self.next_chunk_page_breaks_tag] = []
        metadata[self.current_chunk_uuid] = result.EmbeddingStore.uuid
        metadata[self.original_format_tag] = result.original_format
        metadata[self.format_tag] = result.format

        log.debug("preparing document with source %s", metadata[self.name_tag])

        if result.encrypted.lower() == "true":
            try:
                page_content, metadata = self.decrypt_data(page_content, metadata)
            except Exception as e_x:
                log.error("unable to decrypt doc %s", metadata)
                page_content = ""
                for key in self.encrypt_meta_keys:
                    metadata[key] = ""

        # metadata = self._remove_chunk_text_overlap_from_doc_metadata(page_content,metadata)

        return Document(
            page_content=page_content,
            metadata=metadata,
        )