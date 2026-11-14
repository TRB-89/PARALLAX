import os
import rasterio
from starlette.testclient import TestClient
from app.main import app
from app.services.search_service import search_service

client = TestClient(app)

def test_chennai_opera_discovery():
    """
    Fulfills Phase E:
    Tests discovery for Chennai, Tamil Nadu [80.0, 12.7, 80.4, 13.3]
    between 2024-11-25 and 2024-12-10 for OPERA_L3_DSWX-S1_V1.
    """
    res = search_service.search_granules(
        dataset="OPERA_L3_DSWX-S1_V1",
        bbox=[80.0, 12.7, 80.4, 13.3],
        start_date="2024-11-25",
        end_date="2024-12-10",
        limit=5
    )
    assert res.granules_found > 0
    granule = res.granules[0]
    assert "OPERA_L3_DSWx" in granule.granule_ur
    assert len(granule.data_links) > 0
    assert granule.bounding_box is not None

def test_verified_geotiff_raster_validation():
    """Validates real downloaded NASA GeoTIFF assets using Rasterio."""
    rtc_file = os.path.abspath("data/cases/demo_flood_case/post_rtc_vv.tif")
    assert os.path.exists(rtc_file)

    with rasterio.open(rtc_file) as src:
        assert src.count == 1
        assert src.crs is not None
        assert src.bounds.left < src.bounds.right
        assert src.bounds.bottom < src.bounds.top
        data = src.read(1)
        assert data.size > 10000
