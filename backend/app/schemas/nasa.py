from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class EarthdataStatusResponse(BaseModel):
    authenticated: bool
    configured: bool
    username_masked: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    demo_fallback_active: bool
    daac_status: Dict[str, str]

class GranuleMetadata(BaseModel):
    granule_ur: str
    title: str
    dataset: str
    acquisition_time: str
    cloud_cover: Optional[float] = None
    size_mb: float
    data_links: List[str]
    bounding_box: List[float] = Field(description="[min_lon, min_lat, max_lon, max_lat]")
    polarizations: List[str] = []
    day_night_flag: Optional[str] = None

class SearchRequest(BaseModel):
    dataset: str = Field(..., description="e.g. OPERA_L3_DSWX-S1_V1 or OPERA_L2_RTC-S1_V1")
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    polarization: Optional[str] = Field(None, description="e.g. VV, VH, or VV+VH")
    limit: int = Field(10, ge=1, le=50)

class SearchResponse(BaseModel):
    dataset: str
    granules_found: int
    granules: List[GranuleMetadata]
    provenance: str = "NASA Earthdata CMR (Live Search)"

class DownloadRequest(BaseModel):
    dataset: str
    granule_urs: List[str]
    assets: List[str] = Field(default=["VV", "VH", "WTR"], description="Assets to extract")

class DownloadResponse(BaseModel):
    job_id: str
    status: str
    message: str
    granule_count: int

class DatasetInfo(BaseModel):
    id: str
    short_name: str
    title: str
    level: str
    sensor: str
    spatial_resolution_m: int
    provider: str
    doi: str
    description: str
    bands_or_polarizations: List[str]
    citation: str
