from abc import ABC, abstractmethod
from typing import List
from app.photometa import PhotoMeta

class ClusteringStrategy(ABC):
    """Abstract base class for a clustering strategy."""

    @abstractmethod
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Applies a clustering strategy to a list of photos.

        Args:
            photos: A list of PhotoMeta objects to cluster.

        Returns:
            A list of clusters, where each cluster is a list of PhotoMeta objects.
        """
        pass
