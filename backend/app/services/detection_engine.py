import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import ndimage
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from app.config import settings
from app.services.geospatial import geospatial_processor

logger = logging.getLogger("parallax.detection")

class DetectionEngine:
    """
    PARALLAX Dual-Signature Radar Flood & Inundated Vegetation Engine.
    Detects both specular open-water darkening and canopy double-bounce brightening.
    Performs cross-sensor divergence analysis against NASA OPERA DSWx.
    """

    def analyze_scene(
        self,
        pre_vv_path: str,
        post_vv_path: str,
        pre_vh_path: Optional[str] = None,
        post_vh_path: Optional[str] = None,
        dswx_path: Optional[str] = None,
        darkening_thresh_db: float = -3.0,
        brightening_thresh_db: float = 2.5,
        min_cluster_pixels: int = 5,
        case_id: str = "custom_analysis"
    ) -> Dict[str, Any]:
        """
        Executes scientific radar backscatter change analysis and cross-sensor comparison.
        """
        logger.info(f"Starting PARALLAX detection on case {case_id}")

        # 1. Ingest reference post-VV
        post_vv_raw, profile = geospatial_processor.read_geotiff(post_vv_path)
        pre_vv_raw, _ = geospatial_processor.read_geotiff(pre_vv_path)

        # 2. Convert to decibels (dB)
        post_vv_db = geospatial_processor.linear_to_db(post_vv_raw)
        pre_vv_db = geospatial_processor.linear_to_db(pre_vv_raw)

        # Compute valid mask
        valid_mask = ~np.isnan(post_vv_db) & ~np.isnan(pre_vv_db)
        pixel_size_m = abs(profile["transform"].a)
        pixel_area_km2 = (pixel_size_m * pixel_size_m) / 1e6

        # 3. Compute Delta dB (Change detection)
        delta_vv = np.full_like(post_vv_db, np.nan)
        delta_vv[valid_mask] = post_vv_db[valid_mask] - pre_vv_db[valid_mask]

        # Ingest VH polarization if available for dual-pol coherence
        delta_vh = None
        if post_vh_path and pre_vh_path and os.path.exists(post_vh_path) and os.path.exists(pre_vh_path):
            post_vh_raw, _ = geospatial_processor.read_geotiff(post_vh_path)
            pre_vh_raw, _ = geospatial_processor.read_geotiff(pre_vh_path)
            post_vh_db = geospatial_processor.linear_to_db(post_vh_raw)
            pre_vh_db = geospatial_processor.linear_to_db(pre_vh_raw)
            vh_valid = ~np.isnan(post_vh_db) & ~np.isnan(pre_vh_db)
            delta_vh = np.full_like(post_vh_db, np.nan)
            delta_vh[vh_valid] = post_vh_db[vh_valid] - pre_vh_db[vh_valid]

        # 4. Classification
        # Class 1: Specular Open Water Darkening (delta < darkening_thresh_db AND post_vv < -16.0 dB)
        open_water_mask = valid_mask & (delta_vv < darkening_thresh_db) & (post_vv_db < -16.0)

        # Class 2: Inundated Vegetation Double-Bounce Brightening (delta > brightening_thresh_db)
        inundated_veg_mask = valid_mask & (delta_vv > brightening_thresh_db)
        if delta_vh is not None:
            # Canopy double-bounce is especially prominent when cross-pol VH also enhances
            inundated_veg_mask = inundated_veg_mask & (delta_vh > 1.5)

        # 5. Morphological Spatial Filtering (remove isolated single-pixel speckle)
        structure = ndimage.generate_binary_structure(2, 2)
        open_water_clean = ndimage.binary_opening(open_water_mask, structure=structure)
        inundated_veg_clean = ndimage.binary_opening(inundated_veg_mask, structure=structure)

        # Cluster size filter
        open_water_clean = self._filter_min_size(open_water_clean, min_cluster_pixels)
        inundated_veg_clean = self._filter_min_size(inundated_veg_clean, min_cluster_pixels)

        # 6. Confidence Estimation (0.0 to 1.0)
        confidence_map = np.zeros_like(post_vv_db, dtype=np.float32)
        # Open water confidence scales with magnitude of drop below threshold
        confidence_map[open_water_clean] = np.clip(
            np.abs(delta_vv[open_water_clean]) / 8.0, 0.4, 0.98
        )
        # Inundated veg confidence scales with brightening amplitude
        confidence_map[inundated_veg_clean] = np.clip(
            (delta_vv[inundated_veg_clean] - brightening_thresh_db) / 6.0 + 0.5, 0.45, 0.95
        )

        # 7. Comparison against NASA OPERA DSWx
        opera_water_mask = np.zeros_like(open_water_clean, dtype=bool)
        if dswx_path and os.path.exists(dswx_path):
            dswx_raw, _ = geospatial_processor.read_geotiff(dswx_path)
            # OPERA DSWx: Class 1 = Open water, Class 2 = Partial surface water
            opera_water_mask = (dswx_raw == 1) | (dswx_raw == 2)

        # Cross-sensor categories
        # Category A: Agreement on surface water
        agreement_mask = open_water_clean & opera_water_mask
        # Category B: Candidate Inundated Vegetation (Potential canopy penetration / gap)
        candidate_hidden_water_mask = inundated_veg_clean & (~opera_water_mask)
        # Category C: OPERA only (detected by optical/DSWx, no radar delta detected)
        opera_only_mask = opera_water_mask & (~open_water_clean) & (~inundated_veg_clean)

        # 8. Compute Scientific Metrics
        total_valid_pixels = int(np.count_nonzero(valid_mask))
        total_aoi_km2 = round(total_valid_pixels * pixel_area_km2, 2)
        
        open_water_km2 = round(int(np.count_nonzero(open_water_clean)) * pixel_area_km2, 2)
        inundated_veg_km2 = round(int(np.count_nonzero(inundated_veg_clean)) * pixel_area_km2, 2)
        opera_water_km2 = round(int(np.count_nonzero(opera_water_mask)) * pixel_area_km2, 2)

        agreement_km2 = round(int(np.count_nonzero(agreement_mask)) * pixel_area_km2, 2)
        candidate_hidden_km2 = round(int(np.count_nonzero(candidate_hidden_water_mask)) * pixel_area_km2, 2)
        opera_only_km2 = round(int(np.count_nonzero(opera_only_mask)) * pixel_area_km2, 2)
        unflooded_km2 = round(total_aoi_km2 - (open_water_km2 + inundated_veg_km2), 2)

        # dB statistics
        mean_delta_water = round(float(np.nanmean(delta_vv[open_water_clean])) if np.any(open_water_clean) else 0.0, 2)
        mean_delta_veg = round(float(np.nanmean(delta_vv[inundated_veg_clean])) if np.any(inundated_veg_clean) else 0.0, 2)
        baseline_mean_vv = round(float(np.nanmean(pre_vv_db[valid_mask])), 2)
        event_mean_vv = round(float(np.nanmean(post_vv_db[valid_mask])), 2)

        # Confidence counts
        high_conf = np.count_nonzero(confidence_map >= 0.8)
        med_conf = np.count_nonzero((confidence_map >= 0.5) & (confidence_map < 0.8))
        low_conf = np.count_nonzero((confidence_map > 0.0) & (confidence_map < 0.5))
        total_detected = high_conf + med_conf + low_conf or 1

        # 9. Save Product Rasters & GeoJSON Polygons
        output_dir = settings.PROCESSED_DIR / case_id
        output_dir.mkdir(parents=True, exist_ok=True)

        classification_raster_path = str(output_dir / "classification.tif")
        # Raster classes: 0=Dry, 1=Open Water, 2=Inundated Veg, 3=OPERA Only, 255=NoData
        class_arr = np.full(post_vv_db.shape, 255, dtype=np.uint8)
        class_arr[valid_mask] = 0
        class_arr[open_water_clean] = 1
        class_arr[inundated_veg_clean] = 2
        class_arr[opera_only_mask] = 3

        class_profile = profile.copy()
        class_profile.update(dtype=rasterio.uint8, count=1, nodata=255)
        with rasterio.open(classification_raster_path, "w", **class_profile) as dst:
            dst.write(class_arr, 1)

        # Export candidate polygons GeoJSON
        geojson_features = self._vectorize_candidates(
            class_arr, confidence_map, delta_vv, profile, pixel_area_km2
        )
        geojson_path = str(output_dir / "candidates.geojson")
        with open(geojson_path, "w") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": geojson_features
            }, f)

        bounds_wgs84 = geospatial_processor.get_wgs84_bounds(profile)

        return {
            "analysis_id": f"ana_{case_id}",
            "case_id": case_id,
            "bounds_wgs84": bounds_wgs84,
            "center": [(bounds_wgs84[0] + bounds_wgs84[2])/2, (bounds_wgs84[1] + bounds_wgs84[3])/2],
            "total_aoi_km2": total_aoi_km2,
            "open_water_km2": open_water_km2,
            "candidate_inundated_veg_km2": inundated_veg_km2,
            "opera_dswx_water_km2": opera_water_km2,
            "cross_sensor_breakdown": {
                "agreement_open_water_km2": agreement_km2,
                "candidate_hidden_water_km2": candidate_hidden_km2,
                "opera_only_water_km2": opera_only_km2,
                "unflooded_land_km2": max(0.0, unflooded_km2)
            },
            "radar_statistics": {
                "baseline_vv_mean_db": baseline_mean_vv,
                "event_vv_mean_db": event_mean_vv,
                "mean_db_change_open_water": mean_delta_water,
                "mean_db_change_vegetation": mean_delta_veg
            },
            "confidence_summary": {
                "high_pct": round((high_conf / total_detected) * 100, 1),
                "medium_pct": round((med_conf / total_detected) * 100, 1),
                "low_pct": round((low_conf / total_detected) * 100, 1)
            },
            "layer_urls": {
                "geojson": f"/data/processed/{case_id}/candidates.geojson",
                "classification_tif": f"/data/processed/{case_id}/classification.tif"
            }
        }

    def _filter_min_size(self, binary_mask: np.ndarray, min_size: int) -> np.ndarray:
        labeled, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return binary_mask
        counts = np.bincount(labeled.ravel())
        remove_mask = counts < min_size
        remove_mask[0] = False
        remove_pixels = remove_mask[labeled]
        filtered = binary_mask.copy()
        filtered[remove_pixels] = False
        return filtered

    def _vectorize_candidates(
        self,
        class_arr: np.ndarray,
        confidence_map: np.ndarray,
        delta_vv: np.ndarray,
        profile: Dict[str, Any],
        pixel_area_km2: float
    ) -> list:
        """Converts flood and inundated vegetation candidate clusters into GeoJSON polygons."""
        candidate_mask = (class_arr == 1) | (class_arr == 2)
        if not np.any(candidate_mask):
            return []

        features = []
        for geom_dict, val in shapes(class_arr.astype(np.int32), mask=candidate_mask, transform=profile["transform"]):
            poly = shape(geom_dict)
            if poly.area < (pixel_area_km2 * 1e6 * 3): # Skip tiny speckles
                continue

            cls_name = "Open Water / Specular Darkening" if val == 1 else "Candidate Inundated Vegetation (Double-Bounce)"
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class_id": int(val),
                    "class_name": cls_name,
                    "confidence_tier": "High" if val == 1 else "Medium-High",
                    "detection_mechanism": "Surface specular reflection attenuation" if val == 1 else "Canopy-water dihedral double-bounce enhancement",
                    "area_ha": round((poly.area / 10000), 2),
                    "scientific_label": "Candidate Inundation (Unverified ground-truth)"
                }
            })

            if len(features) >= 150: # Cap for map performance
                break

        return features

detection_engine = DetectionEngine()
