import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple, Union

import httpx
import piexif
from app.models.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class MetadataExtractor:
    async def extract(self, image_path: str) -> PhotoMeta:
        """Asynchronously extracts metadata from an image file or URL."""
        exif = None
        try:
            if image_path.startswith("http"):
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_path)
                    resp.raise_for_status()
                    content = resp.content
                # Run the synchronous piexif.load with bytes
                exif = self._export_exif_sync(content)
            else:
                # Run the synchronous piexif.load in a thread pool (implicitly via sync call here for now, or wrap if needed)
                exif = self._export_exif_sync(image_path)
        except Exception as e:
            logger.error(f"Failed to run exif extraction for {image_path}: {e}")
            exif = None
        
        return self._build_photometa(image_path, exif)

    def extract_from_bytes(self, content: bytes, file_name: str) -> PhotoMeta:
        """Extracts metadata directly from image bytes."""
        exif = self._export_exif_sync(content)
        return self._build_photometa(file_name, exif)

    def _build_photometa(self, path: str, exif: Optional[dict]) -> PhotoMeta:
        lat, lon, alt, direction = self._get_gps_from_exif(exif)
        timestamp = self._parse_datetime_from_exif(exif)

        focal_35mm = None
        orientation = None
        digital_zoom = None
        scene_capture_type = None
        white_balance = None
        exposure_mode = None
        flash = None

        if exif is not None:
            exif_0th = exif.get("0th", {})
            exif_exif = exif.get("Exif", {})

            orientation = exif_0th.get(piexif.ImageIFD.Orientation)
            focal_35mm = exif_exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
            if isinstance(focal_35mm, (tuple, list)):
                focal_35mm = self._rational_to_float(focal_35mm)

            dz = exif_exif.get(piexif.ExifIFD.DigitalZoomRatio)
            digital_zoom = self._rational_to_float(dz)

            scene_capture_type = exif_exif.get(piexif.ExifIFD.SceneCaptureType)
            white_balance = exif_exif.get(piexif.ExifIFD.WhiteBalance)
            exposure_mode = exif_exif.get(piexif.ExifIFD.ExposureMode)
            flash = exif_exif.get(piexif.ExifIFD.Flash)

        return PhotoMeta(
            path=path,
            original_name=path.split("/")[-1],
            lat=lat,
            lon=lon,
            alt=alt,
            timestamp=timestamp,
            focal_35mm=float(focal_35mm) if focal_35mm is not None else None,
            orientation=orientation,
            digital_zoom=digital_zoom,
            scene_capture_type=scene_capture_type,
            white_balance=white_balance,
            exposure_mode=exposure_mode,
            flash=flash,
            gps_img_direction=direction,
        )

    async def _load_cache(self):
        pass

    def _export_exif_sync(self, img_input: Union[str, bytes]) -> Optional[dict]:
        """Synchronous helper for loading EXIF data from file path or bytes."""
        try:
            return piexif.load(img_input)
        except Exception as e:
            logger.warning(f"Failed to load EXIF from input: {e}")
            return None

    def _rational_to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                if den == 0: return None
                return float(num) / float(den)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                v0 = value[0]
                if isinstance(v0, tuple) and len(v0) == 2:
                    num, den = v0
                    if den == 0: return None
                    return float(num) / float(den)
        except (ValueError, TypeError, ZeroDivisionError):
            return None
        return None

    def _parse_datetime_from_exif(self, exif: Optional[dict]) -> Optional[float]:
        if exif is None: return None
        exif_exif = exif.get("Exif", {})
        dt_bytes = exif_exif.get(piexif.ExifIFD.DateTimeOriginal)
        if not dt_bytes: return None

        try:
            dt_str = dt_bytes.decode()
            base_dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except (UnicodeDecodeError, ValueError):
            return None

        # Microseconds
        subsec_bytes = exif_exif.get(piexif.ExifIFD.SubSecTimeOriginal)
        microseconds = 0
        if subsec_bytes:
            try:
                subsec_str = subsec_bytes.decode().ljust(6, '0')
                microseconds = int(subsec_str[:6])
            except (UnicodeDecodeError, ValueError):
                microseconds = 0
        
        # Timezone
        # Default to KST (Korea Standard Time) if no offset is provided
        offset_bytes = exif_exif.get(piexif.ExifIFD.OffsetTimeOriginal)
        tz = timezone(timedelta(hours=9))
        
        if offset_bytes:
            try:
                offset_str = offset_bytes.decode()
                sign = -1 if offset_str[0] == "-" else 1
                h = int(offset_str[1:3])
                m = int(offset_str[4:6])
                tz = timezone(sign * timedelta(hours=h, minutes=m))
            except (UnicodeDecodeError, ValueError):
                pass 

        dt = base_dt.replace(microsecond=microseconds, tzinfo=tz)
        return dt.timestamp()

    def _get_gps_from_exif(self, exif: Optional[dict]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if exif is None or "GPS" not in exif:
            return None, None, None, None

        gps = exif["GPS"]
        def convert_coord(coord, ref):
            try:
                degrees, minutes, seconds = [x[0] / x[1] for x in coord]
                result = degrees + (minutes / 60.0) + (seconds / 3600.0)
                
                # Handle reference (N/S/E/W)
                if ref:
                    if isinstance(ref, bytes):
                        try:
                            ref = ref.decode().upper()
                        except:
                            pass # Keep as bytes if decode fails
                    if isinstance(ref, str):
                        ref = ref.upper()
                
                if ref in [b"N", "N", b"E", "E"]:
                    return result
                elif ref in [b"S", "S", b"W", "W"]:
                    return -result
                else:
                    return result if ref in [b"N", "N", b"E", "E"] else -result

            except (TypeError, ValueError, ZeroDivisionError, IndexError):
                return None

        lat = convert_coord(gps.get(piexif.GPSIFD.GPSLatitude), gps.get(piexif.GPSIFD.GPSLatitudeRef))
        lon = convert_coord(gps.get(piexif.GPSIFD.GPSLongitude), gps.get(piexif.GPSIFD.GPSLongitudeRef))
        
        # Filter out invalid (0, 0) coordinates which often indicate GPS init failure
        if lat == 0.0 and lon == 0.0:
            return None, None, None, None

        alt = None
        if piexif.GPSIFD.GPSAltitude in gps:
            alt_val = gps[piexif.GPSIFD.GPSAltitude]
            if alt_val[1] != 0:
                alt = alt_val[0] / alt_val[1]
                
                # Handle Altitude Reference
                # 0 = Above Sea Level, 1 = Below Sea Level
                alt_ref = gps.get(piexif.GPSIFD.GPSAltitudeRef)
                if alt_ref == 1 or alt_ref == b'\x01':
                    alt = -alt

        direction = self._rational_to_float(gps.get(piexif.GPSIFD.GPSImgDirection))
        
        # Correct direction if it refers to Magnetic North
        # Korean Average Magnetic Declination is approx 8.5 degrees West (as of 2020s)
        # True North = Magnetic North - 8.5
        if direction is not None:
            dir_ref = gps.get(piexif.GPSIFD.GPSImgDirectionRef)
            if dir_ref:
                if isinstance(dir_ref, bytes):
                    try:
                        dir_ref = dir_ref.decode().upper()
                    except:
                        pass
                elif isinstance(dir_ref, str):
                    dir_ref = dir_ref.upper()

                # 'M' stands for Magnetic North
                if dir_ref == 'M':
                    direction -= 8.5
                    if direction < 0:
                        direction += 360.0

        return lat, lon, alt, direction
