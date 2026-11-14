import os
import json
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.schemas.analysis import (
    CompareRequest,
    CompareResponse,
    AnalysisMetrics,
    ProcessRequest,
)
from app.routers.nasa import process_satellite_imagery

logger = logging.getLogger("parallax.api.analysis")
router = APIRouter(prefix="/api/analysis", tags=["PARALLAX Analysis"])

@router.get("/cases")
def list_available_cases():
    """Lists locally available verified demonstration cases."""
    cases = []
    for case_dir in settings.CASES_DIR.iterdir():
        if case_dir.is_dir():
            manifest_file = case_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                        cases.append(data)
                except Exception as e:
                    logger.warning("Error reading manifest for %s: %s", case_dir.name, e)
    return cases

@router.post("/compare", response_model=CompareResponse)
def compare_cross_sensor(payload: CompareRequest):
    """
    Executes cross-sensor comparison between Sentinel-1 L2 RTC-derived inundation
    and NASA OPERA Level-3 DSWx water extent.
    """
    case_id = payload.case_id or "demo_flood_case"
    # Run or retrieve processing
    metrics = process_satellite_imagery(ProcessRequest(case_id=case_id))

    breakdown = metrics.cross_sensor_breakdown
    summary = (
        f"Cross-sensor evaluation identified {breakdown.agreement_open_water_km2} km² of confirmed "
        f"open-water agreement and {breakdown.candidate_hidden_water_km2} km² of candidate inundated "
        f"vegetation obscured in optical/surface water products. {breakdown.opera_only_water_km2} km² "
        f"were observed as surface water by OPERA without significant radar backscatter change."
    )

    interpretation = (
        "NISAR / Sentinel-1 radar change detection identifies backscatter enhancement (+2.5 dB) "
        "indicative of dihedral double-bounce reflections between standing water and woody plant stalks. "
        "These regions constitute candidate hidden water gaps. "
        "Per scientific protocol: Disagreement with OPERA DSWx highlights detection complementarity, "
        "not definitive superiority."
    )

    return CompareResponse(
        analysis_id=metrics.analysis_id,
        case_id=case_id,
        metrics=metrics,
        comparison_summary=summary,
        scientific_interpretation=interpretation
    )

@router.get("/{analysis_id}/metrics", response_model=AnalysisMetrics)
def get_analysis_metrics(analysis_id: str):
    """Retrieves full scientific metrics and layer metadata for an analysis session."""
    # Maps analysis_id (e.g. ana_demo_flood_case) to case
    case_id = analysis_id.replace("ana_", "") if analysis_id.startswith("ana_") else "demo_flood_case"
    return process_satellite_imagery(ProcessRequest(case_id=case_id))
