"""create photo_details table

Revision ID: 9028347102ab
Revises: 5346d5dd0ebe
Create Date: 2025-12-23 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9028347102ab'
down_revision: Union[str, None] = '5346d5dd0ebe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'photo_details',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('photo_id', sa.String(), nullable=False),
        sa.Column('device', sa.String(), nullable=True),
        sa.Column('focal_length', sa.Float(), nullable=True),
        sa.Column('exposure_time', sa.Float(), nullable=True),
        sa.Column('iso', sa.Integer(), nullable=True),
        sa.Column('flash', sa.Integer(), nullable=True),
        sa.Column('orientation', sa.Integer(), nullable=True),
        sa.Column('gps_img_direction', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('photo_id')
    )


def downgrade() -> None:
    op.drop_table('photo_details')
