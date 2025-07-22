"""release 3.4 seq 2 create partial index for hnsw

Revision ID: 6031d261f854
Revises: 0d47b9945549
Create Date: 2025-05-02 12:21:56.962623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6031d261f854'
down_revision: Union[str, None] = '0d47b9945549'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""ALTER TABLE IF EXISTS public.langchain_pg_embedding ADD COLUMN IF NOT EXISTS use_hnsw integer DEFAULT 1;""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_hnsw_partial 
ON public.langchain_pg_embedding USING hnsw 
(embedding vector_cosine_ops) WHERE (use_hnsw = 1);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare
    begin
    raise notice 'Inside approximate adv search nolang';

    RETURN QUERY SELECT INNE.* FROM (SELECT distinct on (D.ID)
        E.DOCUMENT AS LANGCHAIN_PG_EMBEDDING_DOCUMENT,
        E.JMETA AS LANGCHAIN_PG_EMBEDDING_CMETADATA,
        E.PAGE AS LANGCHAIN_PG_EMBEDDING_PAGE,
        D.ID AS LANGCHAIN_PG_EMBEDDING_DOC_ID,
        E.LANG AS LANGCHAIN_PG_EMBEDDING_LANG,
        E.EMBEDDING <=> query_language_vector as distance,
        D.ENCRYPTED
        FROM
        LANGCHAIN_PG_EMBEDDING E
        JOIN LANGCHAIN_PG_DOCUMENT D ON E.DOC_ID = D.ID
        AND D.ID = ANY (query_files)
        AND D.COLLECTION_ID = collectin_id 
	WHERE E.USE_HNSW=1
    ORDER BY
    D.ID, E.EMBEDDING <=> query_language_vector) INNE
	ORDER BY INNE.DISTANCE
    LIMIT num_chunks_to_return;
    end first_block 
    
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_singlelang(
	search_language character varying,
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search singlelang';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and e.lang = search_language
    and d.collection_id = collectin_id
	and use_hnsw=1
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 
                                
                
    
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_otherlangs(
	exclude_language character varying,
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search otherlangs';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and e.lang <> exclude_language
    and d.collection_id = collectin_id
	and use_hnsw=1
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 
                                
                
    
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search nolang';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and d.collection_id = collectin_id
	and use_hnsw=1
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 

                    
    
$BODY$;""")
    op.execute("""DROP INDEX IF EXISTS public.idx_hnsw;""")
    op.execute("""ALTER TABLE IF EXISTS public.langchain_pg_embedding DROP COLUMN IF EXISTS embedding_hnsw;""")
def downgrade() -> None:
    op.execute("""ALTER TABLE IF EXISTS public.langchain_pg_embedding
    ADD COLUMN embedding_hnsw vector(1536) GENERATED ALWAYS AS (embedding) stored;""");
    op.execute("""CREATE INDEX IF NOT EXISTS idx_hnsw
    ON public.langchain_pg_embedding USING hnsw
    (embedding_hnsw vector_cosine_ops)
    TABLESPACE pg_default;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search nolang';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding_hnsw <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and d.collection_id = collectin_id
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 

                    
    
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_otherlangs(
	exclude_language character varying,
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search otherlangs';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding_hnsw <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and e.lang <> exclude_language
    and d.collection_id = collectin_id
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 
                                
                
    
$BODY$;
""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_singlelang(
	search_language character varying,
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare

    begin
    raise notice 'Inside approximate search singlelang';
    RETURN QUERY SELECT
    e.embedding AS langchain_pg_embedding_embedding,
    e.document AS langchain_pg_embedding_document,
    e.jmeta AS langchain_pg_embedding_cmetadata,
    e.page AS langchain_pg_embedding_page,
    e.uuid AS langchain_pg_embedding_uuid,
    e.doc_id AS langchain_pg_embedding_doc_id,
    e.lang AS langchain_pg_embedding_lang,
    e.chunk_num AS chunk_num,
    e.chunk_len_in_chars AS chunk_len_in_chars,
    
    d.document_summary AS document_summary,
    d.name as document_name,
    e.embedding_hnsw <=> query_language_vector as distance,
    d.encrypted
    
    FROM
    langchain_pg_embedding e
    join langchain_pg_document d on d.id = e.doc_id
    WHERE
    e.doc_id = ANY (query_files)
    and e.lang = search_language
    and d.collection_id = collectin_id
    ORDER BY
    distance ASC
    LIMIT num_chunks_to_return
    ;
    end first_block 
                                
                
    
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare
    begin
    raise notice 'Inside approximate adv search nolang';

    RETURN QUERY SELECT INNE.* FROM (SELECT distinct on (D.ID)
        E.DOCUMENT AS LANGCHAIN_PG_EMBEDDING_DOCUMENT,
        E.JMETA AS LANGCHAIN_PG_EMBEDDING_CMETADATA,
        E.PAGE AS LANGCHAIN_PG_EMBEDDING_PAGE,
        D.ID AS LANGCHAIN_PG_EMBEDDING_DOC_ID,
        E.LANG AS LANGCHAIN_PG_EMBEDDING_LANG,
        E.EMBEDDING_HNSW <=> query_language_vector as distance,
        D.ENCRYPTED
        FROM
        LANGCHAIN_PG_EMBEDDING E
        JOIN LANGCHAIN_PG_DOCUMENT D ON E.DOC_ID = D.ID
        AND D.ID = ANY (query_files)
        AND D.COLLECTION_ID = collectin_id 
    ORDER BY
    D.ID, E.EMBEDDING_HNSW <=> query_language_vector) INNE
	ORDER BY INNE.DISTANCE
    LIMIT num_chunks_to_return;
    end first_block 
    
$BODY$;""")
    op.execute("""DROP INDEX IF EXISTS public.idx_hnsw_partial;""")
    op.execute("""ALTER TABLE IF EXISTS public.langchain_pg_embedding DROP COLUMN IF EXISTS use_hnsw;""")
    
