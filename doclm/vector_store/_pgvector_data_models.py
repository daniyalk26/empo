import uuid
from pydantic import BaseModel as schemaBase
from pydantic import Field, ConfigDict

from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import sqlalchemy
from sqlalchemy import select, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import JSON, ARRAY, UUID, TSVECTOR, INTEGER
from pgvector.sqlalchemy import Vector
from pgvector.utils import from_db
from sqlalchemy.orm import Session, relationship, declarative_base, Mapped, mapped_column

DecBase = declarative_base()  # type: Any


# class BaseModel(DecBase):
#     """Base model for the SQL stores."""

#     __abstract__ = True
#     uuid: Mapped[UUID] = mapped_column(type_=UUID, primary_key=True, default=uuid.uuid4)


class CollectionStore(DecBase):
    """Collection store."""

    __tablename__ = "langchain_pg_collection"
    collection_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    cmetadata: Mapped[Optional[JSON]] = mapped_column(type_=JSON)
    tenant_id: Mapped[int]
    application_id: Mapped[int]

    doc_ids: Mapped[List["DocumentStore"]] = relationship(
        back_populates="collection",
        passive_deletes=True,
    )

    webdata: Mapped[List["WebStore"]] = relationship(
        back_populates="collection",
        passive_deletes=True,
    )

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.name == name).first()  # type: ignore
    
    @classmethod
    def get_by_id(cls, session: Session, id: uuid) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def get_by_tenant_id(cls, session: Session, tenant_id: int) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.tenant_id == tenant_id).first()

    @classmethod
    async def async_get_by_tenant_id(cls, session: Session, tenant_id: int) -> Optional["CollectionStore"]:
        query = select(cls).filter(cls.tenant_id == tenant_id)
        result = await session.execute(query)
        return result.scalar()

    @classmethod
    def get_by_application_id(cls, session: Session, application_id: int) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.application_id == application_id).first()
    
    @classmethod
    def get_or_create(
            cls,
            session: Session,
            tenant_id: int,
            application_id: int,
            name: str,
            cmetadata: Optional[dict] = None,
    ) -> Tuple["CollectionStore", bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = cls.get_by_tenant_id(session, tenant_id)
        if collection:
            return collection, created

        collection = cls(name=name,tenant_id=tenant_id,application_id=application_id, cmetadata=cmetadata)
        session.add(collection)
        session.commit()
        created = True
        return collection, created


class DocumentStore(DecBase):
    """document store."""

    __tablename__ = "langchain_pg_document"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.collection_id",
            ondelete="CASCADE",
        ),
    )
    collection: Mapped["CollectionStore"] = relationship(back_populates="doc_ids")

    document_chunks: Mapped[List["EmbeddingStore"]] = relationship(
        back_populates="parent_document",
        passive_deletes=True,
    )

    jmeta: Mapped[Optional[JSON]] = mapped_column(type_=JSON)
    document_summary: Mapped[str]
    lang: Mapped[str]
    author: Mapped[Optional[str]]
    keyword: Mapped[Optional[List[str]]] = mapped_column(type_=ARRAY(sqlalchemy.String))
    title: Mapped[Optional[str]]
    edition: Mapped[Optional[str]]
    year: Mapped[Optional[str]]
    name: Mapped[str]
    format: Mapped[str]
    original_format: Mapped[str]
    processing_time: Mapped[str]
    encrypted: Mapped[Optional[str]]


class EmbeddingStore(DecBase):
    __tablename__ = "langchain_pg_embedding"
    uuid: Mapped[UUID] = mapped_column(type_=UUID, primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[int] = mapped_column(INTEGER)
    collection_id: Mapped[int] = mapped_column(INTEGER)
    lang: Mapped[str]
    embedding: Mapped[Vector] = mapped_column(Vector(1536))
    document: Mapped[str]
    page: Mapped[Optional[str]]
    chunk_num: Mapped[int]
    chunk_len_in_chars: Mapped[Optional[int]]
    jmeta: Mapped[Optional[JSON]] = mapped_column(type_=JSON)
    keywords_ts_vector: Mapped[Optional[TSVECTOR]]=mapped_column(type_=TSVECTOR)
    adv_search_ts_vector: Mapped[Optional[TSVECTOR]]=mapped_column(type_=TSVECTOR)
    
    parent_document: Mapped["DocumentStore"] = relationship(back_populates="document_chunks")

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    __table_args__ = (
        sqlalchemy.Index('idx_composite_docid_lang', "doc_id", "lang", postgresql_using="btree"), 
        # sqlalchemy.Index('idx_hnsw', "embedding_hnsw", postgresql_ops={'embedding_hnsw': 'vector_cosine_ops'}, postgresql_with={'m': 16, 'ef_construction': 128},postgresql_using="hnsw"), 
        sqlalchemy.Index('idx_hnsw_partial', "embedding", postgresql_ops={'embedding': 'vector_cosine_ops'}, postgresql_with={'m': 16, 'ef_construction': 128}, postgresql_where={'use_hnsw':1},postgresql_using="hnsw"), 
        sqlalchemy.Index('idx_adv_srch_kwvector_fts', "adv_search_ts_vector", postgresql_using="gin"), 
        sqlalchemy.Index('idx_chat_kwvector_fts', "keywords_ts_vector", postgresql_using="gin"), 
        ForeignKeyConstraint([collection_id, doc_id],
                                           [DocumentStore.collection_id, DocumentStore.id], ondelete="CASCADE")
    )

    @classmethod
    def apply_schema(
        cls,
        embedding: Vector,
        document,
        jmeta,
        page,
        _uuid,
        doc_id,
        lang,
        chunk_num,
        chunk_len_in_chars,
        document_summary,
        document_name,
        score,
        encrypted,
        original_format,
        format,
        previous_chunk_text,
        previous_chunk_jmeta,
        next_chunk_text,
        next_chunk_jmeta,
        author
        ):
        record = cls(
            embedding=from_db(embedding),
            # embedding_hnsw=embedding,
            document=document,
            jmeta=jmeta,
            page=page,
            uuid=_uuid,
            doc_id=doc_id,
            lang=lang,
            chunk_num=chunk_num,
            chunk_len_in_chars=chunk_len_in_chars)

        return DocumentReadSchema(
            EmbeddingStore=record, document_name=document_name, distance=score,
            document_summary=document_summary,
            previous_chunk_text=previous_chunk_text or '',
            previous_chunk_jmeta=previous_chunk_jmeta or {},
            next_chunk_text=next_chunk_text or '',
            next_chunk_jmeta=next_chunk_jmeta or {},
            encrypted=encrypted or 'false',
            original_format=original_format,
            format=format,
            author=author
            )


class DocumentReadSchema(schemaBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    EmbeddingStore: Any
    document_name: str
    document_summary: str
    distance: float
    previous_chunk_text: str
    previous_chunk_jmeta: dict
    next_chunk_text: str
    next_chunk_jmeta: dict
    encrypted: str
    original_format: str
    format: str
    author: str
    
class AdvSearchReadSchema(schemaBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_text: str
    doc_summary: str
    jmeta: dict
    page_no: str
    doc_id: int
    lang: str
    score: float
    encrypted: str
    highlighted_text: Optional[str]=None
    search_type: Optional[str]=None

    @classmethod
    def apply_schema(cls,chunk_text,doc_summary,jmeta,page_no,doc_id,lang,score,encrypted):
        return cls(chunk_text = chunk_text,
                   doc_summary =doc_summary,
                   jmeta=jmeta,
                   page_no=page_no,
                   doc_id=doc_id,
                   lang=lang,
                   score=score,
                   encrypted=encrypted or 'false'
                   )



class WebStore(DecBase):
    """Embedding store."""

    __tablename__ = "langchain_pg_web"
    uuid: Mapped[UUID] = mapped_column(type_=UUID, primary_key=True, default=uuid.uuid4)
    collection_id = mapped_column(
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.collection_id",
            ondelete="CASCADE",
        ),
    )
    collection: Mapped["CollectionStore"] = relationship(back_populates="webdata")
    doc_url :Mapped[str] = mapped_column(sqlalchemy.String, nullable=False)
    # lang = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    # embedding: Vector = sqlalchemy.Column(Vector(1536))
    date = sqlalchemy.Column(sqlalchemy.DATE, nullable=False)
    # chat_id = sqlalchemy.Column(sqlalchemy.int, nullable=False)
    # embedding_hnsw: Vector = sqlalchemy.Column(Vector(1536))
    document :Mapped[str] = mapped_column(sqlalchemy.String, nullable=False)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)
    ts_col = sqlalchemy.Column(TSVECTOR)
    chunk_no = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    # custom_id : any user defined id
    # custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    @classmethod
    def apply_schema(cls,
                     document,
                     jmeta,
                     doc_url,
                     date,
                     chunk_num,
                     score,
                     next_chunk_text,
                     next_chunk_jmeta,
                     previous_chunk_text,
                     previous_chunk_jmeta
                     ):
        record = cls(
                     document=document,
                     cmetadata=jmeta,
                     doc_url=doc_url,
                     date=date,
                     chunk_no=chunk_num,
                     )
        return WebReadSchema(WebStore=record, 
                    score=score,
                    next_chunk_text=next_chunk_text or '',
                    next_chunk_jmeta=next_chunk_jmeta or {},
                    previous_chunk_text=previous_chunk_text or '',
                    previous_chunk_jmeta=previous_chunk_jmeta or {}
                    )


class WebReadSchema(schemaBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    WebStore: Any
    score: float
    next_chunk_text: str
    next_chunk_jmeta: dict
    previous_chunk_text: str
    previous_chunk_jmeta: dict