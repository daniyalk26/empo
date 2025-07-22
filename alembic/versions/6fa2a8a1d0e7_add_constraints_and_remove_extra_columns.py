"""add constraints and remove extra columns

Revision ID: 6fa2a8a1d0e7
Revises: c1ec05911d14
Create Date: 2024-07-23 13:40:41.534471

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql



# revision identifiers, used by Alembic.
revision: str = '6fa2a8a1d0e7'
down_revision: Union[str, None] = 'c1ec05911d14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint('langchain_pg_embedding_collection_id_fkey', 'langchain_pg_embedding', type_='foreignkey')
    op.create_foreign_key('langchain_pg_embedding_doc_id_fkey', 'langchain_pg_embedding', 'langchain_pg_document', ['doc_id'], ['id'], ondelete='CASCADE')
    op.drop_column('langchain_pg_embedding', 'cmetadata')
    op.drop_column('langchain_pg_embedding', 'collection_id')


def downgrade() -> None:
    op.add_column('langchain_pg_embedding', sa.Column('collection_id', sa.UUID(), autoincrement=False, nullable=True))
    op.add_column('langchain_pg_embedding', sa.Column('cmetadata', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.drop_constraint('langchain_pg_embedding_doc_id_fkey', 'langchain_pg_embedding', type_='foreignkey')
    op.create_foreign_key('langchain_pg_embedding_collection_id_fkey', 'langchain_pg_embedding', 'langchain_pg_collection', ['collection_id'], ['uuid'], ondelete='CASCADE')
