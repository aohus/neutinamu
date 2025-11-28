from abc import ABC, abstractmethod
from typing import BinaryIO, Optional

class StorageService(ABC):
    """Abstract base class for storage services."""

    @abstractmethod
    async def save_file(self, file: BinaryIO, path: str) -> str:
        """
        Save a file to the storage.
        
        Args:
            file: The file-like object to save.
            path: The destination path/key in the storage.
            
        Returns:
            The path/url where the file was saved.
        """
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from the storage.
        
        Args:
            path: The path/key of the file to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def move_file(self, source_path: str, dest_path: str) -> str:
        """
        Move a file within the storage.
        
        Args:
            source_path: The source path/key.
            dest_path: The destination path/key.
            
        Returns:
            The new path/url of the file.
        """
        pass
        
    @abstractmethod
    def get_url(self, path: str) -> str:
        """
        Get the accessible URL for a file.
        
        Args:
            path: The storage path/key.
            
        Returns:
            The public or internal URL.
        """
        pass