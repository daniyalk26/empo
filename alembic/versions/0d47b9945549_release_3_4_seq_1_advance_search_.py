"""release 3.4 seq 1 advance search functions update

Revision ID: 0d47b9945549
Revises: 621d58e4a981
Create Date: 2025-05-02 10:34:10.942586

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d47b9945549'
down_revision: Union[str, None] = '621d58e4a981'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_hnsw_nolang(vector, integer, integer, integer[]);""")
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nohnsw_nolang(vector, integer, integer, integer[]);""")
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

def downgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_nohnsw_nolang(vector, integer, integer, integer[]);""")
    op.execute("""DROP FUNCTION IF EXISTS public.adv_search_semantic_hnsw_nolang(vector, integer, integer, integer[]);""")
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

    RETURN QUERY SELECT distinct on (D.ID)
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
    D.ID, E.EMBEDDING <=> query_language_vector
    LIMIT num_chunks_to_return;
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

    RETURN QUERY SELECT distinct on (D.ID)
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
    D.ID, E.EMBEDDING_HNSW <=> query_language_vector
    LIMIT num_chunks_to_return;
    end first_block 
    
$BODY$;""")
