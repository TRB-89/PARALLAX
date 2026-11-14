import os
import sys
import pytest
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.auth.earthdata import auth_manager
from app.services.geospatial import geospatial_processor
from app.services.search_service import search_service
from app.services.detection_engine import detection_engine
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_earthdata_auth_secure_handling():
    """Validates that Earthdata authentication safely handles invalid or missing credentials."""
    status = auth_manager.authenticate()
    assert "authenticated" in status
    assert "status" in status
    assert "error_message" in status
    # Ensure sensitive credentials like raw passwords are not exposed
    assert "Tarunbanu2007@" not in str(status)
    if status.get("username_masked"):
        assert "ta***r" in status["username_masked"] or "***" in status["username_masked"]

def test_cmr_dataset_search():
    """Validates NASA CMR satellite discovery for OPERA datasets."""
    res = search_service.search_granules(
        dataset="OPERA_L2_RTC-S1_V1",
        bbox=[-5.263, 48.839, -3.985, 49.246],
        start_date="2024-01-01",
        end_date="2024-01-30",
        limit=2
    )
    assert res.dataset == "OPERA_L2_RTC-S1_V1"
    assert res.granules_found > 0
    assert len(res.granules) > 0
    first_granule = res.granules[0]
    assert "OPERA_L2_RTC" in first_granule.granule_ur

def test_geospatial_raster_conversion_and_bounds():
    """Validates GeoTIFF reading, linear to dB conversion, and CRS validation."""
    sample_rtc = os.path.abspath("data/cases/demo_flood_case/post_rtc_vv.tif")
    assert os.path.exists(sample_rtc), "Sample RTC GeoTIFF must exist"
    
    data, profile = geospatial_processor.read_geotiff(sample_rtc)
    assert data.shape[0] > 0
    assert profile["crs"] is not None

    # Test linear to dB
    db_data = geospatial_processor.linear_to_db(data)
    valid_mask = ~np.isnan(db_data)
    assert np.any(valid_mask)
    # Typical radar backscatter dB ranges between -35 dB and +15 dB
    assert np.nanmean(db_data[valid_mask]) < 10.0
    assert np.nanmean(db_data[valid_mask]) > -30.0

def test_detection_engine_pipeline():
    """Validates the PARALLAX flood & inundated vegetation detection engine on real NASA rasters."""
    case_dir = os.path.abspath("data/cases/demo_flood_case")
    pre_vv = os.path.join(case_dir, "pre_rtc_vv.tif")
    post_vv = os.path.join(case_dir, "post_rtc_vv.tif")
    dswx = os.path.join(case_dir, "opera_dswx_aligned.tif")

    results = detection_engine.analyze_scene(
        pre_vv_path=pre_vv,
        post_vv_path=post_vv,
        dswx_path=dswx,
        case_id="demo_flood_case"
    )

    assert results["total_aoi_km2"] > 500.0
    assert "cross_sensor_breakdown" in results
    assert "candidate_hidden_water_km2" in results["cross_sensor_breakdown"]
    assert "radar_statistics" in results
    assert "confidence_summary" in results

def test_api_endpoints_end_to_end():
    """Validates all required backend endpoints."""
    # 1. GET /api/nasa/status
    r_status = client.get("/api/nasa/status")
    assert r_status.status_code == 200

    # 2. GET /api/nasa/datasets
    r_datasets = client.get("/api/nasa/datasets")
    assert r_datasets.status_code == 200
    assert len(r_datasets.json()) == 2

    # 3. POST /api/nasa/process
    r_process = client.post("/api/nasa/process", json={"case_id": "demo_flood_case"})
    assert r_process.status_code == 200
    data = r_process.json()
    assert data["case_id"] == "demo_flood_case"
    assert data["total_aoi_km2"] > 0
    assert "cross_sensor_breakdown" in data

    # 4. POST /api/analysis/compare
    r_compare = client.post("/api/analysis/compare", json={"case_id": "demo_flood_case"})
    assert r_compare.status_code == 200
    assert "scientific_interpretation" in r_compare.json()

    # 5. GET /api/analysis/{id}/metrics
    r_metrics = client.get(f"/api/analysis/{data['analysis_id']}/metrics")
    assert r_metrics.status_code == 200
    assert r_metrics.json()["analysis_id"] == data["analysis_id"]
