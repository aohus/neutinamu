"""create

Revision ID: 90dec3e4d0f4
Revises: 75b7dc22c2dd
Create Date: 2025-11-27 04:26:44.072712

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '90dec3e4d0f4'
down_revision: Union[str, None] = '75b7dc22c2dd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
