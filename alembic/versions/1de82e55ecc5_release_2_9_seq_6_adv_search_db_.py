"""release 2.9 seq 6 adv search db functions

Revision ID: 1de82e55ecc5
Revises: fecf2e29f617
Create Date: 2024-11-28 17:29:33.066041

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1de82e55ecc5'
down_revision: Union[str, None] = 'fecf2e29f617'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
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

	select count(*) 
	into doc_count
	from public.langchain_pg_embedding e
	join public.langchain_pg_document d
	on e.doc_id = d.id
	where d.collection_id = collectin_id
	and e.doc_id = ANY (query_files);
	
	raise notice 'The number of document is %', doc_count;

	if doc_count<hnsw_threshold then
		RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
	else
		RETURN QUERY SELECT fn.langchain_pg_embedding_document,fn.langchain_pg_embedding_cmetadata,fn.langchain_pg_embedding_page,fn.langchain_pg_embedding_doc_id,fn.langchain_pg_embedding_lang,fn.distance,fn.encrypted FROM adv_search_semantic_hnsw_nolang(query_language_vector, num_chunks_to_return, collectin_id, variadic query_files) as fn;
	end if;	
end first_block 
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_nohnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
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

	RETURN QUERY SELECT
	OUTT2.LANGCHAIN_PG_EMBEDDING_DOCUMENT,
	OUTT2.LANGCHAIN_PG_EMBEDDING_CMETADATA,
	OUTT2.LANGCHAIN_PG_EMBEDDING_PAGE,
	OUTT2.LANGCHAIN_PG_EMBEDDING_DOC_ID,
	OUTT2.LANGCHAIN_PG_EMBEDDING_LANG,
	OUTT2.DISTANCE,
	OUTT2.ENCRYPTED
from (SELECT
	OUTT.LANGCHAIN_PG_EMBEDDING_DOCUMENT,
	OUTT.LANGCHAIN_PG_EMBEDDING_CMETADATA,
	OUTT.LANGCHAIN_PG_EMBEDDING_PAGE,
	OUTT.LANGCHAIN_PG_EMBEDDING_DOC_ID,
	OUTT.LANGCHAIN_PG_EMBEDDING_LANG,
	OUTT.DISTANCE,
	OUTT.ENCRYPTED,
	ROW_NUMBER() OVER (
		PARTITION BY
			OUTT.LANGCHAIN_PG_EMBEDDING_DOC_ID
		ORDER BY
			OUTT.DISTANCE
	) AS CHUNK_RANK_PER_DOC
FROM
	(
		SELECT
			E.DOCUMENT AS LANGCHAIN_PG_EMBEDDING_DOCUMENT,
			E.JMETA AS LANGCHAIN_PG_EMBEDDING_CMETADATA,
			E.PAGE AS LANGCHAIN_PG_EMBEDDING_PAGE,
			E.DOC_ID AS LANGCHAIN_PG_EMBEDDING_DOC_ID,
			E.LANG AS LANGCHAIN_PG_EMBEDDING_LANG,
			E.EMBEDDING <=> query_language_vector as distance,
			D.ENCRYPTED
		FROM
			LANGCHAIN_PG_EMBEDDING E
			JOIN LANGCHAIN_PG_DOCUMENT D ON E.DOC_ID = D.ID
		WHERE
			D.COLLECTION_ID = collectin_id 
			AND E.DOC_ID = ANY (query_files)
	) AS OUTT) AS OUTT2
WHERE
	OUTT2.CHUNK_RANK_PER_DOC = 1
ORDER BY
	OUTT2.DISTANCE
LIMIT num_chunks_to_return;
end first_block 
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.adv_search_semantic_hnsw_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
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

	RETURN QUERY SELECT
	OUTT2.LANGCHAIN_PG_EMBEDDING_DOCUMENT,
	OUTT2.LANGCHAIN_PG_EMBEDDING_CMETADATA,
	OUTT2.LANGCHAIN_PG_EMBEDDING_PAGE,
	OUTT2.LANGCHAIN_PG_EMBEDDING_DOC_ID,
	OUTT2.LANGCHAIN_PG_EMBEDDING_LANG,
	OUTT2.DISTANCE,
	OUTT2.ENCRYPTED
from (SELECT
	OUTT.LANGCHAIN_PG_EMBEDDING_DOCUMENT,
	OUTT.LANGCHAIN_PG_EMBEDDING_CMETADATA,
	OUTT.LANGCHAIN_PG_EMBEDDING_PAGE,
	OUTT.LANGCHAIN_PG_EMBEDDING_DOC_ID,
	OUTT.LANGCHAIN_PG_EMBEDDING_LANG,
	OUTT.DISTANCE,
	OUTT.ENCRYPTED,
	ROW_NUMBER() OVER (
		PARTITION BY
			OUTT.LANGCHAIN_PG_EMBEDDING_DOC_ID
		ORDER BY
			OUTT.DISTANCE
	) AS CHUNK_RANK_PER_DOC
FROM
	(
		SELECT
			E.DOCUMENT AS LANGCHAIN_PG_EMBEDDING_DOCUMENT,
			E.JMETA AS LANGCHAIN_PG_EMBEDDING_CMETADATA,
			E.PAGE AS LANGCHAIN_PG_EMBEDDING_PAGE,
			E.DOC_ID AS LANGCHAIN_PG_EMBEDDING_DOC_ID,
			E.LANG AS LANGCHAIN_PG_EMBEDDING_LANG,
			E.EMBEDDING_HNSW <=> query_language_vector as distance,
			D.ENCRYPTED
		FROM
			LANGCHAIN_PG_EMBEDDING E
			JOIN LANGCHAIN_PG_DOCUMENT D ON E.DOC_ID = D.ID
		WHERE
			D.COLLECTION_ID = collectin_id 
			AND E.DOC_ID = ANY (query_files)
	) AS OUTT) AS OUTT2
WHERE
	OUTT2.CHUNK_RANK_PER_DOC = 1
ORDER BY
	OUTT2.DISTANCE
LIMIT num_chunks_to_return;
end first_block 
$BODY$;""")


def downgrade() -> None:
    op.execute('DROP FUNCTION IF EXISTS public.adv_search_semantic_hnsw_nolang(vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.adv_search_semantic_nohnsw_nolang(vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.adv_search_semantic_nolang(vector, integer, uuid, integer[]);')
    
