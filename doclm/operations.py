
import os
import logging
from logging.config import dictConfig
from typing import  Optional
from threading import Lock

from .external_endpoints.embedding import get_embeddings
from .logger import configuration
from .vector_store import DocumentPGWrite

dictConfig(configuration)

log = logging.getLogger("doclogger")
log.disabled = False
posgres_conn = os.getenv("POSTGRES_CONN_STRING")
thread_pool_size = int(os.getenv("MY_APP_THREADS",10))

# pylint: disable=W0718,C0103,R0914


class RecordOperation:
    def __init__(self):
        embedding_object = get_embeddings(os.getenv("OPENAI_API_TYPE", "openai"))
        self.store_db_writer = self._create_file_store_writer(embedding_object)

    def create_tenant(self, tenant_id: int, application_id: int, collection_name: str,
                      collection_metadata: Optional[dict] = None):
        collection_name = collection_name or 'langchain'
        self.store_db_writer.create_collection(tenant_id, application_id, collection_name, collection_metadata)

    def delete_tenant_by_tenant_id(self, tenant_id: int):
        self.store_db_writer.delete_collection(tenant_id)

    def rename_file(self, tenant_id, file_id, new_name, old_name, file):
        return self.store_db_writer.update_file_name(tenant_id, new_name, file_id, old_name)

    def delete_document(self, files):
        self.store_db_writer.delete_file(files)

    @staticmethod
    def _create_file_store_writer(embeddings_fn):
        """
        build connection to vector store
        :param persist_directory: str directory path
        :return:
        """
        embeddings = embeddings_fn

        log.info("Connecting PgVector dB")

        db_store = DocumentPGWrite(
            posgres_conn,
            embedding_function=embeddings,
            pool_pre_ping=True,
            pool_size=thread_pool_size,
            pool_recycle=3600,  # this line might not be needed
            connect_args={
                "keepalives": 1,
                # "keepalives_idle": 30,
                # "keepalives_interval": 10,
                # "keepalives_count": 5,
            }
        )
        return db_store

rec_obj = None
lock = Lock()

def get_record_obj():
    global rec_obj
    with lock:
        if rec_obj is None:
            rec_obj = RecordOperation()
    return rec_obj
