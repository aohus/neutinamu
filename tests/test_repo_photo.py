import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from app.repository.photo import PhotoRepository
from app.models.photo import Photo

@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    return session

@pytest.mark.asyncio
async def test_get_by_id(mock_db_session):
    repo = PhotoRepository(mock_db_session)
    
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = Photo(id="p1")
    mock_db_session.execute.return_value = mock_result
    
    photo = await repo.get_by_id("p1")
    assert photo.id == "p1"

@pytest.mark.asyncio
async def test_get_by_job_id(mock_db_session):
    repo = PhotoRepository(mock_db_session)
    
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [Photo(id="p1")]
    mock_db_session.execute.return_value = mock_result
    
    photos = await repo.get_by_job_id("j1")
    assert len(photos) == 1

@pytest.mark.asyncio
async def test_save_photo(mock_db_session):
    repo = PhotoRepository(mock_db_session)
    photo = Photo(id="p1")
    
    await repo.save(photo)
    
    mock_db_session.add.assert_called_once_with(photo)
    mock_db_session.flush.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(photo)
