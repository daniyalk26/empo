"""release_2_8_seq_3_truncate ai db

Revision ID: 22821225bcad
Revises: f8ed5d755dee
Create Date: 2024-09-18 17:03:46.453450

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '22821225bcad'
down_revision: Union[str, None] = 'f8ed5d755dee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # op.execute("""TRUNCATE langchain_pg_document cascade;""")
    pass


def downgrade() -> None:
    pass
