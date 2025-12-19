from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.user import User

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self.db.add(user)
        # We don't commit here because AuthService uses UnitOfWork to commit transaction explicitly
        # But wait, other repos in this project seem to commit inside methods (e.g. JobRepository.create_job).
        # However, strictly speaking, Repository should not commit if we use UoW.
        # But for consistency with existing Refactoring (e.g. JobRepo commits), I should follow the pattern OR clean it up.
        # In `AuthService`, I used `self.uow.commit()`. So here I should NOT commit.
        # The existing `JobRepository` committing inside might be an issue for atomicity if UoW is used for multiple actions.
        # For now, `UserRepository` will just add to session.
        return user
