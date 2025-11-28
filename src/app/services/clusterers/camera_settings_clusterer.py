from typing import List

from app.models.photometa import PhotoMeta
from app.services.clusterers.base_clusterer import Clusterer


class CameraSettingsClusterer(Clusterer):
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Splits a group of photos based on camera settings like orientation.
        """
        if not photos:
            return []

        subgroups: List[List[PhotoMeta]] = []
        for photo in photos:
            assigned = False
            for subgroup in subgroups:
                reference_photo = subgroup[0]

                # Group photos with similar orientation.
                # Photos with no orientation info are grouped together.
                orientation_ok = (
                    photo.orientation == reference_photo.orientation or
                    photo.orientation is None or
                    reference_photo.orientation is None
                )
                
                if orientation_ok:
                    subgroup.append(photo)
                    assigned = True
                    break

            if not assigned:
                subgroups.append([photo])
                
        return subgroups
