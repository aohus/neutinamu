"""add url cols to photo

Revision ID: 1029384756ab
Revises: 39d5e3b5d03e
Create Date: 2025-12-04 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1029384756ab'
down_revision: Union[str, None] = '39d5e3b5d03e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('photos', sa.Column('url', sa.String(), nullable=True))
    op.add_column('photos', sa.Column('thumbnail_url', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('photos', 'thumbnail_url')
    op.drop_column('photos', 'url')
