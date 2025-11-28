from typing import List

from app.core.config import SessionConfig
from app.models.photometa import PhotoMeta
from app.services.clusterers.base_clusterer import Clusterer


class TimeSplitClusterer(Clusterer):
    def __init__(self, config: SessionConfig):
        self.config = config

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """Splits a list of photos into groups based on time gaps."""
        if not photos:
            return []

        # Sort photos by timestamp. Photos without a timestamp are handled gracefully.
        sorted_photos = sorted(
            photos,
            key=lambda p: (p.timestamp is None, p.timestamp if p.timestamp is not None else 0.0, p.path)
        )

        if not sorted_photos:
            return []

        groups: List[List[PhotoMeta]] = []
        current_group: List[PhotoMeta] = [sorted_photos[0]]

        for prev_photo, current_photo in zip(sorted_photos, sorted_photos[1:]):
            # If either photo is missing a timestamp, start a new group
            if prev_photo.timestamp is None or current_photo.timestamp is None:
                groups.append(current_group)
                current_group = [current_photo]
                continue

            # If the time gap is within the threshold, add to the current group
            if (current_photo.timestamp - prev_photo.timestamp) <= self.config.MAX_TIME_GAP_SEC:
                current_group.append(current_photo)
            # Otherwise, finalize the current group and start a new one
            else:
                groups.append(current_group)
                current_group = [current_photo]

        # Add the last group
        if current_group:
            groups.append(current_group)

        return groups
