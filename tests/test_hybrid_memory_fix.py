import asyncio
import unittest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from PIL import Image

# Mock dependencies before importing HybridCluster if needed, 
# or use patch.
from app.domain.clusterers.hybrid import HybridCluster
from app.models.photometa import PhotoMeta

class TestHybridClusterMemoryFix(unittest.IsolatedAsyncioTestCase):
    
    @patch("app.domain.clusterers.hybrid.GCSStorageService")
    @patch("app.domain.clusterers.hybrid.CosPlaceExtractor")
    @patch("app.domain.clusterers.hybrid.Image.open")
    @patch("app.domain.clusterers.hybrid.os.path.exists")
    async def test_extract_features_resizes_images_early(self, mock_exists, mock_open, MockExtractor, MockStorage):
        # Setup
        mock_exists.return_value = True # Simulate local file exists
        
        # Mock Image
        mock_img_context = MagicMock()
        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img
        
        # Mock resize chain: img.convert("RGB").resize(...)
        mock_converted = MagicMock()
        mock_img.convert.return_value = mock_converted
        
        mock_resized = MagicMock()
        mock_converted.resize.return_value = mock_resized
        
        # Setup Extractor
        mock_extractor_instance = MockExtractor.return_value
        # extract_batch returns a list of vectors (numpy arrays)
        # BATCH_SIZE is 16. Let's send 2 photos.
        mock_extractor_instance.extract_batch.return_value = [np.zeros(512), np.zeros(512)]
        
        # Instantiate Clusterer
        clusterer = HybridCluster()
        
        # Create dummy photos
        photos = [
            PhotoMeta(path="/tmp/fake1.jpg", thumbnail_path=None),
            PhotoMeta(path="/tmp/fake2.jpg", thumbnail_path=None)
        ]
        
        # Run extraction
        # We need to ensure _download_safe doesn't fail or does nothing.
        # Since we mock os.path.exists = True, it treats them as local files and skips download?
        # In code: 
        # if not target_path: ... 
        # ...
        # if idx in local_files_map and local_files_map[idx].exists(): ...
        
        # We need to make sure the file paths are treated as existing local files or temporary files.
        # The code uses tempfile.TemporaryDirectory().
        
        # Let's mock _download_safe to do nothing
        clusterer._download_safe = MagicMock()
        clusterer._download_safe.return_value = asyncio.Future()
        clusterer._download_safe.return_value.set_result(None)

        # We also need to mock Path.exists() if it uses Path objects.
        # The code uses: if idx in local_files_map and local_files_map[idx].exists():
        # local_files_map[idx] is a Path object.
        
        # It's easier to mock `pathlib.Path.exists` but that's global.
        # Let's rely on the fact that `_extract_features_optimized` does:
        # dest_path = temp_path / f"{idx}{ext}"
        # local_files_map[idx] = dest_path
        # ...
        # if idx in local_files_map and local_files_map[idx].exists():
        
        # So we need the temp file to "exist".
        
        with patch("pathlib.Path.exists", return_value=True):
             features = await clusterer._extract_features_optimized(photos)

        # Assertions
        
        # 1. Verify Image.open was called
        assert mock_open.call_count == 2
        
        # 2. Verify resize was called on the image object (convert result)
        # Expectation: img.convert("RGB").resize((640, 480))
        mock_img.convert.assert_called_with("RGB")
        mock_converted.resize.assert_called_with((640, 480))
        
        # 3. Verify extract_batch was called with the RESIZED images
        # The list passed to extract_batch should contain `mock_resized` objects
        args, _ = mock_extractor_instance.extract_batch.call_args
        passed_images = args[0]
        assert len(passed_images) == 2
        assert passed_images[0] == mock_resized
        assert passed_images[1] == mock_resized
        
        print("\nTest passed: Images were resized before batch extraction.")

if __name__ == "__main__":
    unittest.main()
