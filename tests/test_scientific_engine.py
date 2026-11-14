import os
import pytest
from app.services.detection_engine import detection_engine

def test_scientific_detection_engine_execution():
    """
    Fulfills Phase G & Phase J:
    Validates scientific radar backscatter change detection,
    inundated vegetation double-bounce, specular open-water attenuation,
    and cross-sensor divergence metrics.
    """
    case_dir = os.path.abspath("data/cases/demo_flood_case")
    pre_vv = os.path.join(case_dir, "pre_rtc_vv.tif")
    post_vv = os.path.join(case_dir, "post_rtc_vv.tif")
    pre_vh = os.path.join(case_dir, "pre_rtc_vh.tif")
    post_vh = os.path.join(case_dir, "post_rtc_vh.tif")
    dswx = os.path.join(case_dir, "opera_dswx_aligned.tif")

    res = detection_engine.analyze_scene(
        pre_vv_path=pre_vv,
        post_vv_path=post_vv,
        pre_vh_path=pre_vh,
        post_vh_path=post_vh,
        dswx_path=dswx,
        darkening_thresh_db=-3.0,
        brightening_thresh_db=2.5,
        case_id="demo_flood_case"
    )

    # Scientific constraints validation (Phase J)
    assert res["total_aoi_km2"] > 0
    assert res["open_water_km2"] >= 0
    assert res["candidate_inundated_veg_km2"] >= 0
    assert res["opera_dswx_water_km2"] >= 0

    breakdown = res["cross_sensor_breakdown"]
    assert breakdown["agreement_open_water_km2"] >= 0
    assert breakdown["candidate_hidden_water_km2"] >= 0
    assert breakdown["opera_only_water_km2"] >= 0

    # Overlap cannot exceed either mask
    assert breakdown["agreement_open_water_km2"] <= res["open_water_km2"] + 0.1
    assert breakdown["agreement_open_water_km2"] <= res["opera_dswx_water_km2"] + 0.1

    # Confidence sum bounds
    conf = res["confidence_summary"]
    total_pct = conf["high_pct"] + conf["medium_pct"] + conf["low_pct"]
    assert 99.0 <= total_pct <= 101.0

    # Radar statistics
    stats = res["radar_statistics"]
    assert "mean_db_change_open_water" in stats
    assert "mean_db_change_vegetation" in stats
    # Water should show negative change or zero
    if res["open_water_km2"] > 0:
        assert stats["mean_db_change_open_water"] <= -3.0
    # Veg double bounce should show positive change
    if res["candidate_inundated_veg_km2"] > 0:
        assert stats["mean_db_change_vegetation"] >= 2.5
