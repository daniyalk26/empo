"""release_2_8_seq_2_adding unaccent support

Revision ID: f8ed5d755dee
Revises: cbf7b386206b
Create Date: 2024-09-06 16:37:41.050297

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8ed5d755dee'
down_revision: Union[str, None] = 'cbf7b386206b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""create extension if not exists unaccent;""")
    op.execute("""CREATE TEXT SEARCH CONFIGURATION fr ( COPY = french );
        ALTER TEXT SEARCH CONFIGURATION fr ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, french_stem;

        CREATE TEXT SEARCH CONFIGURATION en ( COPY = english );
        ALTER TEXT SEARCH CONFIGURATION en ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, english_stem;

        CREATE TEXT SEARCH CONFIGURATION de ( COPY = german );
        ALTER TEXT SEARCH CONFIGURATION de ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, german_stem;

        CREATE TEXT SEARCH CONFIGURATION nl ( COPY = dutch );
        ALTER TEXT SEARCH CONFIGURATION nl ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, dutch_stem;

        CREATE TEXT SEARCH CONFIGURATION da ( COPY = danish );
        ALTER TEXT SEARCH CONFIGURATION da ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, danish_stem;

        CREATE TEXT SEARCH CONFIGURATION fi ( COPY = finnish );
        ALTER TEXT SEARCH CONFIGURATION fi ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, finnish_stem;

        CREATE TEXT SEARCH CONFIGURATION hu ( COPY = hungarian );
        ALTER TEXT SEARCH CONFIGURATION hu ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, hungarian_stem;

        CREATE TEXT SEARCH CONFIGURATION it ( COPY = italian );
        ALTER TEXT SEARCH CONFIGURATION it ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, italian_stem;

        CREATE TEXT SEARCH CONFIGURATION no ( COPY = norwegian );
        ALTER TEXT SEARCH CONFIGURATION no ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, norwegian_stem;

        CREATE TEXT SEARCH CONFIGURATION pt ( COPY = portuguese );
        ALTER TEXT SEARCH CONFIGURATION pt ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, portuguese_stem;

        CREATE TEXT SEARCH CONFIGURATION ro ( COPY = romanian );
        ALTER TEXT SEARCH CONFIGURATION ro ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, romanian_stem;

        CREATE TEXT SEARCH CONFIGURATION ru ( COPY = russian );
        ALTER TEXT SEARCH CONFIGURATION ru ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, russian_stem;

        CREATE TEXT SEARCH CONFIGURATION es ( COPY = spanish );
        ALTER TEXT SEARCH CONFIGURATION es ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, spanish_stem;

        CREATE TEXT SEARCH CONFIGURATION sv ( COPY = swedish );
        ALTER TEXT SEARCH CONFIGURATION sv ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, swedish_stem;

        CREATE TEXT SEARCH CONFIGURATION tr ( COPY = turkish );
        ALTER TEXT SEARCH CONFIGURATION tr ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, turkish_stem;

        CREATE TEXT SEARCH CONFIGURATION usimple ( COPY = simple );
        ALTER TEXT SEARCH CONFIGURATION usimple ALTER MAPPING
        FOR hword, hword_part, word WITH unaccent, simple;""")


def downgrade() -> None:
    op.execute("""DROP TEXT SEARCH CONFIGURATION fr;
        DROP TEXT SEARCH CONFIGURATION en;
        DROP TEXT SEARCH CONFIGURATION de;
        DROP TEXT SEARCH CONFIGURATION nl;
        DROP TEXT SEARCH CONFIGURATION da;
        DROP TEXT SEARCH CONFIGURATION fi;
        DROP TEXT SEARCH CONFIGURATION hu;
        DROP TEXT SEARCH CONFIGURATION it;
        DROP TEXT SEARCH CONFIGURATION no;
        DROP TEXT SEARCH CONFIGURATION pt;
        DROP TEXT SEARCH CONFIGURATION ro;
        DROP TEXT SEARCH CONFIGURATION ru;
        DROP TEXT SEARCH CONFIGURATION es;
        DROP TEXT SEARCH CONFIGURATION sv;
        DROP TEXT SEARCH CONFIGURATION tr;
        DROP TEXT SEARCH CONFIGURATION usimple;""")
    op.execute("""drop extension if exists unaccent;""")