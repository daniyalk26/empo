"""add data to release2.7 tables

Revision ID: c1ec05911d14
Revises: d60276da18fb
Create Date: 2024-07-22 14:44:41.746926

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1ec05911d14'
down_revision: Union[str, None] = 'd60276da18fb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""UPDATE langchain_pg_embedding SET page = cmetadata->>'page', \
                chunk_num = (cmetadata->>'chunk_num')::integer, \
                chunk_len_in_chars = (cmetadata->>'chunk_len_in_chars')::integer, \
                jmeta = '{}'::jsonb ;""")
    op.execute("""insert into langchain_pg_document(
                    id, collection_id, jmeta, document_summary, lang, author, keyword, title, edition, year, name, format, original_format, processing_time, encrypted)
                        select distinct doc_id as id, 
                            collection_id,
                            '{}'::jsonb as jmeta,
                            first_value(cmetadata->>'document_summary') over (partition by doc_id) as document_summary,
                            cmetadata->>'lang' as lang,
                            NULL as author,
                            NULL::varchar[] as keyword,
                            NULL as title,
                            NULL as edition,
                            NULL as year,
                            cmetadata->>'name' as name,
                            cmetadata->>'format' as format,
                            cmetadata->>'original_format' as original_format,
                            cmetadata->>'processing_time' as processing_time,
                            cmetadata->>'encrypted' as encrypted
                        from langchain_pg_embedding
                ;""")


def downgrade() -> None:
    pass
