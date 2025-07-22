"""release/3.1 seq 1 adding indexes for fts

Revision ID: 38d4c37e7e17
Revises: 1de82e55ecc5
Create Date: 2025-01-07 16:47:51.110824

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '38d4c37e7e17'
down_revision: Union[str, None] = '1de82e55ecc5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""CREATE INDEX IF NOT EXISTS idx_chat_kwvector_fts
        ON public.langchain_pg_embedding USING gin
        (keywords_ts_vector)""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_adv_srch_kwvector_fts
        ON public.langchain_pg_embedding USING gin
        (adv_search_ts_vector)""")


def downgrade() -> None:
    op.execute("""DROP INDEX IF EXISTS public.idx_adv_srch_kwvector_fts;""")
    op.execute("""DROP INDEX IF EXISTS public.idx_chat_kwvector_fts;""")
