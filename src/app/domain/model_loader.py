import logging

from app.domain.clusterers.hybrid import CosPlaceExtractor
from app.domain.clusterers.hybrid_legacy import CosPlaceExtractorLegacy

logger = logging.getLogger(__name__)

# Global instances for model extractors
_cos_place_extractor: CosPlaceExtractor | None = None
_cos_place_extractor_legacy: CosPlaceExtractorLegacy | None = None


def get_cos_place_extractor() -> CosPlaceExtractor:
    """
    Returns a singleton instance of CosPlaceExtractor.
    The model will be loaded on the first call to this function.
    """
    global _cos_place_extractor
    if _cos_place_extractor is None:
        logger.info("Initializing CosPlaceExtractor (ONNX/Torch) for the first time.")
        _cos_place_extractor = CosPlaceExtractor()
    return _cos_place_extractor


def get_cos_place_extractor_legacy() -> CosPlaceExtractorLegacy:
    """
    Returns a singleton instance of CosPlaceExtractorLegacy.
    The model will be loaded on the first call to this function.
    """
    global _cos_place_extractor_legacy
    if _cos_place_extractor_legacy is None:
        logger.info("Initializing CosPlaceExtractorLegacy (Torch) for the first time.")
        _cos_place_extractor_legacy = CosPlaceExtractorLegacy()
    return _cos_place_extractor_legacy


async def initialize_all_models():
    """
    Explicitly initializes all model extractors. Call this during application startup.
    """
    logger.info("Pre-initializing all CosPlace model extractors...")
    get_cos_place_extractor()
    # get_cos_place_extractor_legacy()
    logger.info("CosPlace model extractors pre-initialized.")
