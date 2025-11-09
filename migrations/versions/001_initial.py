"""Initial migration

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create training_runs table
    op.create_table(
        'training_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('epochs', sa.Integer(), nullable=False),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        sa.Column('learning_rate', sa.Float(), nullable=False),
        sa.Column('final_loss', sa.Float(), nullable=True),
        sa.Column('final_accuracy', sa.Float(), nullable=True),
        sa.Column('training_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_training_runs_id'), 'training_runs', ['id'], unique=False)

    # Create epoch_results table
    op.create_table(
        'epoch_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('training_run_id', sa.Integer(), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('train_loss', sa.Float(), nullable=False),
        sa.Column('train_accuracy', sa.Float(), nullable=False),
        sa.Column('val_loss', sa.Float(), nullable=True),
        sa.Column('val_accuracy', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_epoch_results_id'), 'epoch_results', ['id'], unique=False)
    op.create_index(op.f('ix_epoch_results_training_run_id'), 'epoch_results', ['training_run_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_epoch_results_training_run_id'), table_name='epoch_results')
    op.drop_index(op.f('ix_epoch_results_id'), table_name='epoch_results')
    op.drop_table('epoch_results')
    op.drop_index(op.f('ix_training_runs_id'), table_name='training_runs')
    op.drop_table('training_runs')
