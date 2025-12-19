import io
import logging
from PIL import Image, ImageFile
import piexif

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_thumbnail(image_data: bytes, is_full_image: bool = False) -> bytes | None:
    """
    image_data: Bytes data of the full or partial file.
    Returns: Thumbnail JPEG bytes or None.
    """
    try:
        # 1. Try to extract embedded thumbnail via piexif
        exif_dict = piexif.load(image_data)
        if exif_dict and exif_dict.get("thumbnail"):
            logger.debug("Extracted embedded thumbnail via piexif.")
            return exif_dict["thumbnail"]
        
        # 2. If full image is provided, generate thumbnail using PIL
        if is_full_image:
            image_data_io = io.BytesIO(image_data)
            with Image.open(image_data_io) as img:
                # Convert to RGB to handle RGBA/P images
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                    
                img.thumbnail((1024, 768))
                thumb_io = io.BytesIO()
                img.save(thumb_io, format="JPEG", quality=85, optimize=True)
                thumb_bytes = thumb_io.getvalue()
                return thumb_bytes
    except Exception as e:
        logger.debug(f"Thumbnail generation failed: {e}")
        pass
    return None
