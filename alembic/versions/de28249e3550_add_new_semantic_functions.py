"""add new semantic functions

Revision ID: de28249e3550
Revises: 56b58ac0709a
Create Date: 2024-07-23 13:57:59.430660

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de28249e3550'
down_revision: Union[str, None] = '56b58ac0709a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_singlelang(
                search_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
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
                                raise notice 'Inside singlelang';
                                select count(*) 
                                into doc_count
                                from public.langchain_pg_embedding e
                                join public.langchain_pg_document d
                                on e.doc_id = d.id
                                where d.collection_id = collectin_id
                                and e.doc_id = ANY (query_files)
                                and e.lang = search_language;
                                raise notice 'The number of document is %', doc_count;
                                
                                if doc_count<hnsw_threshold then
                                    RETURN QUERY SELECT * from semantic_nohnsw_singlelang(search_language,query_language_vector, num_chunks_to_return, collectin_id, variadic query_files);
                                else
                                    RETURN QUERY SELECT * from semantic_hnsw_singlelang(search_language, query_language_vector, num_chunks_to_return, collectin_id, variadic query_files);
                                end if;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_otherlangs(
                exclude_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying) 
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
                                raise notice 'Inside otherlangs';
                                select count(*) 
                                into doc_count
                                from public.langchain_pg_embedding e
                                join public.langchain_pg_document d
                                on e.doc_id = d.id
                                where d.collection_id = collectin_id
                                and e.doc_id = ANY (query_files)
                                and e.lang <> exclude_language;
                                
                                raise notice 'The number of document is %', doc_count;
                                
                                if doc_count<hnsw_threshold then
                                    RETURN QUERY SELECT * from semantic_nohnsw_otherlangs(exclude_language,query_language_vector, num_chunks_to_return,collectin_id, variadic query_files);
                                else
                                    RETURN QUERY SELECT * from semantic_hnsw_otherlangs(exclude_language, query_language_vector, num_chunks_to_return,collectin_id, variadic query_files);
                                end if;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, previous_chunk character varying, next_chunk character varying) 
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
                                SELECT MAIN.*,COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk FROM 
                                (SELECT * from semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num
                                ;
                                else
                                    RETURN QUERY 
                                SELECT MAIN.*,COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk FROM 
                                (SELECT * from semantic_hnsw_nolang(query_language_vector, num_chunks_to_return,collectin_id, variadic query_files)) as MAIN
                                LEFT OUTER JOIN
                                langchain_pg_embedding as PREVIOUS_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = PREVIOUS_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num-1 = PREVIOUS_CHUNK_TBL.chunk_num
                                LEFT OUTER JOIN
                                langchain_pg_embedding as NEXT_CHUNK_TBL
                                ON MAIN.langchain_pg_embedding_doc_id = NEXT_CHUNK_TBL.doc_id
                                AND MAIN.chunk_num+1 = NEXT_CHUNK_TBL.chunk_num;
                                end if;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nohnsw_singlelang(
                search_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
                                    raise notice 'Inside exact search singlelang';
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
                                    ORDER BY
                                    distance ASC
                                    LIMIT num_chunks_to_return
                                    ;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nohnsw_otherlangs(
                exclude_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
                                    raise notice 'Inside exact search otherlangs';
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
                                    ORDER BY
                                    distance ASC
                                    LIMIT num_chunks_to_return
                                    ;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nohnsw_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
                                    raise notice 'Inside exact search nolang';
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
                                    ORDER BY
                                    distance ASC
                                    LIMIT num_chunks_to_return
                                    ;
                            end first_block 
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_singlelang(
                search_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_otherlangs(
                exclude_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
                            
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
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
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_crosslingual(
                query_language character varying,
                query_language_vector vector,
                en_language_vector vector,
                num_chunks_to_return integer,
                collectin_id uuid,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_page character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, chunk_num integer, chunk_len_in_chars integer, document_summary character varying, document_name character varying, distance double precision, encrypted character varying, previous_chunk character varying, next_chunk character varying) 
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
                            SELECT MAIN.*,COALESCE(PREVIOUS_CHUNK_TBL.document,'') as previous_chunk, COALESCE(NEXT_CHUNK_TBL.document,'') as next_chunk FROM 
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
                                ;
                            end first_block 
                            
            $BODY$;""")


def downgrade() -> None:
    op.execute('DROP FUNCTION IF EXISTS public.semantic_singlelang(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_otherlangs(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nolang(vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_singlelang(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_otherlangs(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_nolang(vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_singlelang(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_otherlangs(character varying, vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_nolang(vector, integer, uuid, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_crosslingual(character varying, vector, vector, integer, uuid, integer[]);')
