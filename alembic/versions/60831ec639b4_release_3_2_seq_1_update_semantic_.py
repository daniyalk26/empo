"""release 3.2 seq 1 update semantic functions to get author as well

Revision ID: 60831ec639b4
Revises: 1e6c25edb135
Create Date: 2025-02-11 10:43:15.677801

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '60831ec639b4'
down_revision: Union[str, None] = '1e6c25edb135'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.semantic_crosslingual(character varying, vector, vector, integer, uuid, integer[]);""")
    op.execute("""DROP FUNCTION IF EXISTS public.semantic_nolang(vector, integer, uuid, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, original_format character varying, format character varying, previous_chunk character varying, previous_chunk_jmeta json, next_chunk character varying, next_chunk_jmeta json, author character varying) 
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
                                raise notice 'Inside nolang';
                                
                                select count(*) 
                                into doc_count
                                from public.langchain_pg_embedding e
                                join public.langchain_pg_document d
                                on e.doc_id = d.id
                                where d.collection_id = collectin_id
                                and e.doc_id = ANY (query_files);
                                
                                raise notice 'The number of document is %', doc_count;
                                
                                if doc_count<hnsw_threshold then
                                    RETURN QUERY 
                                SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_json, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_json, COALESCE(DOC_TBL.author,'Unknown') as author FROM 
                                (SELECT * from semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
								INNER JOIN
								langchain_pg_document as DOC_TBL
								ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id
                                ;
                                else
                                    RETURN QUERY 
                                SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_json, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_json, COALESCE(DOC_TBL.author,'Unknown') as author FROM 
                                (SELECT * from semantic_hnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
								 INNER JOIN
								 langchain_pg_document as DOC_TBL
								 ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id;
                                end if;
                            end first_block 
                            
            
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_crosslingual(
	query_language character varying,
	query_language_vector vector,
	en_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, original_format character varying, format character varying, previous_chunk character varying, previous_chunk_json json, next_chunk character varying, next_chunk_json json, author character varying) 
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
                                raise notice 'Inside cross lingual semantic function';  
                                RETURN QUERY 
                            SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_json, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_json,COALESCE(DOC_TBL.author,'Unknown') FROM 
                                (
                                SELECT INN.* FROM (SELECT * from semantic_otherlangs(query_language, en_language_vector, num_chunks_to_return, collectin_id, variadic query_files)
                                                UNION ALL
                                                SELECT * from semantic_singlelang(query_language, query_language_vector, num_chunks_to_return, collectin_id, variadic query_files)
                                                            ) INN
                                                        ORDER BY INN.DISTANCE ASC
                                                        LIMIT num_chunks_to_return
                                ) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
                                INNER JOIN
								 langchain_pg_document as DOC_TBL
								 ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id;
                            end first_block 
                            
            
$BODY$;""")
def downgrade() -> None:
    op.execute("""DROP FUNCTION IF EXISTS public.semantic_nolang(vector, integer, uuid, integer[];""")
    op.execute("""DROP FUNCTION IF EXISTS public.semantic_crosslingual(character varying, vector, vector, integer, uuid, integer[]);""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nolang(
	query_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, original_format character varying, format character varying, previous_chunk character varying, previous_chunk_jmeta json, next_chunk character varying, next_chunk_jmeta json) 
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
                                raise notice 'Inside nolang';
                                
                                select count(*) 
                                into doc_count
                                from public.langchain_pg_embedding e
                                join public.langchain_pg_document d
                                on e.doc_id = d.id
                                where d.collection_id = collectin_id
                                and e.doc_id = ANY (query_files);
                                
                                raise notice 'The number of document is %', doc_count;
                                
                                if doc_count<hnsw_threshold then
                                    RETURN QUERY 
                                SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_page_breaks, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_page_breaks FROM 
                                (SELECT * from semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
								INNER JOIN
								langchain_pg_document as DOC_TBL
								ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id
                                ;
                                else
                                    RETURN QUERY 
                                SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_json, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_json FROM 
                                (SELECT * from semantic_hnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
								 INNER JOIN
								 langchain_pg_document as DOC_TBL
								 ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id;
                                end if;
                            end first_block 
                            
            
$BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_crosslingual(
	query_language character varying,
	query_language_vector vector,
	en_language_vector vector,
	num_chunks_to_return integer,
	collectin_id uuid,
	VARIADIC query_files integer[])
    RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, original_format character varying, format character varying, previous_chunk character varying, previous_chunk_json json, next_chunk character varying, next_chunk_json json) 
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
                                raise notice 'Inside cross lingual semantic function';  
                                RETURN QUERY 
                            SELECT MAIN.*,DOC_TBL.original_format, DOC_TBL.format, COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(PREVIOUS_CHUNK_TBL.jmeta, '{}')::json as previous_chunk_json, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk, COALESCE(NEXT_CHUNK_TBL.jmeta,'{}')::json as next_chunk_json FROM 
                                (
                                SELECT INN.* FROM (SELECT * from semantic_otherlangs(query_language, en_language_vector, num_chunks_to_return, collectin_id, variadic query_files)
                                                UNION ALL
                                                SELECT * from semantic_singlelang(query_language, query_language_vector, num_chunks_to_return, collectin_id, variadic query_files)
                                                            ) INN
                                                        ORDER BY INN.DISTANCE ASC
                                                        LIMIT num_chunks_to_return
                                ) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
                                INNER JOIN
								 langchain_pg_document as DOC_TBL
								 ON MAIN.langchain_pg_embedding_doc_id = DOC_TBL.id;
                            end first_block 
                            
            
$BODY$;""")