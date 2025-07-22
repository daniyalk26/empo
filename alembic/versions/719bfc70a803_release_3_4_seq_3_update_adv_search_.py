"""release 3.4 seq 3 update adv search semantic functions

Revision ID: 719bfc70a803
Revises: 6031d261f854
Create Date: 2025-05-08 09:34:45.232827

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '719bfc70a803'
down_revision: Union[str, None] = '6031d261f854'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nolang(vector, integer, integer, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_document character varying, langchain_pg_document_summary character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare
    doc_count integer := 0;
    hnsw_threshold constant integer := 70000;
    begin
    raise notice 'Inside adv search nolang';

    select count(d.id) 
    into doc_count
    from public.langchain_pg_embedding e
    join public.langchain_pg_document d
    on e.doc_id = d.id
    and d.id = ANY (query_files)
    and d.collection_id = collectin_id
    ;
    
    raise notice 'The number of document is %', doc_count;

    if doc_count<hnsw_threshold then
        RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_document_summary,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
    else
        RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_document_summary,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_hnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
    end if; 
    end first_block 
    
$BODY$;""")
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nohnsw_nolang(vector, integer, integer, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nohnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_document character varying, langchain_pg_document_summary character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision, encrypted character varying) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
                                                                                                    
    <<first_block>>
    declare
    begin
    raise notice 'Inside exact adv search nolang';

    RETURN QUERY SELECT INNE.* FROM (SELECT distinct on (D.ID)
        E.DOCUMENT AS LANGCHAIN_PG_EMBEDDING_DOCUMENT,
		D.DOCUMENT_SUMMARY AS langchain_pg_document_summary,
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
    ORDER BY
    D.ID, E.EMBEDDING <=> query_language_vector) INNE
	ORDER BY INNE.DISTANCE
    LIMIT num_chunks_to_return;
    end first_block 
    
$BODY$;""")
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_hnsw_nolang(vector, integer, integer, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id integer,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_document character varying, langchain_pg_document_summary character varying,langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision, encrypted character varying) 
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
		D.DOCUMENT_SUMMARY AS langchain_pg_document_summary,
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
    
$BODY$;
""")

def downgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_hnsw_nolang(vector, integer, integer, integer[]);""")
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
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nohnsw_nolang(vector, integer, integer, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nohnsw_nolang(
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
    raise notice 'Inside exact adv search nolang';

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
    ORDER BY
    D.ID, E.EMBEDDING <=> query_language_vector) INNE
	ORDER BY INNE.DISTANCE
    LIMIT num_chunks_to_return;
    end first_block 
    
$BODY$;""")
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nolang(vector, integer, integer, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nolang(
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
    doc_count integer := 0;
    hnsw_threshold constant integer := 70000;
    begin
    raise notice 'Inside adv search nolang';

    select count(d.id) 
    into doc_count
    from public.langchain_pg_embedding e
    join public.langchain_pg_document d
    on e.doc_id = d.id
    and d.id = ANY (query_files)
    and d.collection_id = collectin_id
    ;
    
    raise notice 'The number of document is %', doc_count;

    if doc_count<hnsw_threshold then
        RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
    else
        RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_hnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
    end if; 
    end first_block 
    
$BODY$;""")
    
