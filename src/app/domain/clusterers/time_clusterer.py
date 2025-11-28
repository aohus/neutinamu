import logging
from typing import List

from app.core.config import SessionConfig
from app.domain.clusterers.base_clusterer import Clusterer
from app.models.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class TimeSplitClusterer(Clusterer):
    def __init__(self, config: SessionConfig):
        self.config = config
        logger.debug(f"TimeSplitClusterer initialized with MAX_TIME_GAP_SEC: {config.MAX_TIME_GAP_SEC}")

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """Splits a list of photos into groups based on time gaps."""
        if not photos:
            return []

        logger.info(f"Starting time-based splitting for {len(photos)} photos.")
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
                logger.debug("Found photo with no timestamp, starting new group.")
                groups.append(current_group)
                current_group = [current_photo]
                continue

            time_gap = current_photo.timestamp - prev_photo.timestamp
            # If the time gap is within the threshold, add to the current group
            if time_gap <= self.config.MAX_TIME_GAP_SEC:
                current_group.append(current_photo)
            # Otherwise, finalize the current group and start a new one
            else:
                logger.debug(f"Time gap of {time_gap:.2f}s exceeded threshold. Starting new group.")
                groups.append(current_group)
                current_group = [current_photo]

        # Add the last group
        if current_group:
            groups.append(current_group)

        logger.info(f"Time-based splitting resulted in {len(groups)} groups.")
        return groups
