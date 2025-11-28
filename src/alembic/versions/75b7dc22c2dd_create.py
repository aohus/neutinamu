"""create

Revision ID: 75b7dc22c2dd
Revises: 48d8a35087c5
Create Date: 2025-11-27 04:26:37.564499

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '75b7dc22c2dd'
down_revision: Union[str, None] = '48d8a35087c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
