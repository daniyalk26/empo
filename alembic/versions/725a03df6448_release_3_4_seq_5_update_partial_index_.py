"""release 3.4 seq 5 update partial index for hnsw

Revision ID: 725a03df6448
Revises: 533c8d8699d6
Create Date: 2025-05-28 13:40:57.010119

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '725a03df6448'
down_revision: Union[str, None] = '533c8d8699d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""DROP INDEX IF EXISTS public.idx_hnsw_partial;""")
    op.execute("""CREATE INDEX idx_hnsw_partial ON public.langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128) WHERE use_hnsw = 1;""")


def downgrade() -> None:
    op.execute("""DROP INDEX IF EXISTS public.idx_hnsw_partial;""")
    op.execute("""CREATE INDEX idx_hnsw_partial ON public.langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) WHERE use_hnsw = 1;""")

