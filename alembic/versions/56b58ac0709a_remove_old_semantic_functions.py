"""remove old semantic functions

Revision ID: 56b58ac0709a
Revises: 6fa2a8a1d0e7
Create Date: 2024-07-23 13:43:07.310610

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56b58ac0709a'
down_revision: Union[str, None] = '6fa2a8a1d0e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('DROP FUNCTION IF EXISTS public.semantic_singlelang(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_otherlangs(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nolang(vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_singlelang(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_otherlangs(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_nohnsw_nolang(vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_singlelang(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_otherlangs(character varying, vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_hnsw_nolang(vector, integer, integer[]);')
    op.execute('DROP FUNCTION IF EXISTS public.semantic_crosslingual(character varying, vector, vector, integer, integer[]);')


def downgrade() -> None:
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_singlelang(
                search_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                from public.langchain_pg_embedding
                where doc_id = ANY (query_files)
                and lang = search_language;
                raise notice 'The number of document is %', doc_count;
                
                if doc_count<hnsw_threshold then
                    RETURN QUERY SELECT * from semantic_nohnsw_singlelang(search_language,query_language_vector, num_chunks_to_return, variadic query_files);
                else
                    RETURN QUERY SELECT * from semantic_hnsw_singlelang(search_language, query_language_vector, num_chunks_to_return, variadic query_files);
                end if;
            end first_block 
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_otherlangs(
                exclude_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                from public.langchain_pg_embedding
                where doc_id = ANY (query_files)
                and lang <> exclude_language;
                raise notice 'The number of document is %', doc_count;
                
                if doc_count<hnsw_threshold then
                    RETURN QUERY SELECT * from semantic_nohnsw_otherlangs(exclude_language,query_language_vector, num_chunks_to_return, variadic query_files);
                else
                    RETURN QUERY SELECT * from semantic_hnsw_otherlangs(exclude_language, query_language_vector, num_chunks_to_return, variadic query_files);
                end if;
            end first_block 
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                from public.langchain_pg_embedding
                where doc_id = ANY (query_files);
                raise notice 'The number of document is %', doc_count;
                
                if doc_count<hnsw_threshold then
                    RETURN QUERY SELECT * from semantic_nohnsw_nolang(query_language_vector, num_chunks_to_return, variadic query_files);
                else
                    RETURN QUERY SELECT * from semantic_hnsw_nolang(query_language_vector, num_chunks_to_return, variadic query_files);
                end if;
            end first_block 
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nohnsw_singlelang(
                search_language character varying,
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
                    and lang = search_language
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
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
                    and lang <> exclude_language
                    ORDER BY
                    distance ASC
                    LIMIT num_chunks_to_return
                    ;
            end first_block 
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_nohnsw_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
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
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding_hnsw <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
                    and lang = search_language
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
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding_hnsw <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
                    and lang <> exclude_language
                    ORDER BY
                    distance ASC
                    LIMIT num_chunks_to_return
                    ;
            end first_block 
            $BODY$;""")
    op.execute("""CREATE OR REPLACE FUNCTION public.semantic_hnsw_nolang(
                query_language_vector vector,
                num_chunks_to_return integer,
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                    langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
                    langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
                    langchain_pg_embedding.document AS langchain_pg_embedding_document,
                    langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
                    langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
                    langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid,
                    langchain_pg_embedding.doc_id AS langchain_pg_embedding_doc_id,
                    langchain_pg_embedding.lang AS langchain_pg_embedding_lang,
                    langchain_pg_embedding.embedding_hnsw <=> query_language_vector as distance
                    
                    FROM
                    langchain_pg_embedding
                    WHERE
                    (
                    langchain_pg_embedding.doc_id = ANY (query_files)
                    )
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
                VARIADIC query_files integer[])
                RETURNS TABLE(langchain_pg_embedding_collection_id uuid, langchain_pg_embedding_embedding vector, langchain_pg_embedding_document character varying, langchain_pg_embedding_cmetadata json, langchain_pg_embedding_custom_id character varying, langchain_pg_embedding_uuid uuid, langchain_pg_embedding_doc_id integer, langchain_pg_embedding_lang character varying, distance double precision) 
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
                RETURN QUERY SELECT * FROM (SELECT * from semantic_otherlangs(query_language, en_language_vector, num_chunks_to_return, variadic query_files)
                                UNION ALL
                                SELECT * from semantic_singlelang(query_language, query_language_vector, num_chunks_to_return, variadic query_files)
                                            ) INN
                                        ORDER BY DISTANCE ASC
                                        LIMIT num_chunks_to_return
                ;
            end first_block 
            $BODY$;""")
