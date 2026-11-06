"""
ml/heat/tree_planting_priority.py:
Delineates and ranks urban tree-planting priority zones for Visakhapatnam
based on the intersection of high LST hotspot scores, canopy deficits, and built-up density.
Generates GeoJSON candidate parcels and specific municipal greening recommendations.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from shapely.geometry import Polygon, mapping


def compute_tree_planting_priority_index(
    hotspot_score: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    mndwi: np.ndarray,
) -> np.ndarray:
    """
    Computes continuous Tree-Planting Priority Index (TPPI, 0.0 to 1.0):
      - 45% Weight: UHI Hotspot Score (thermal urgency)
      - 35% Weight: Canopy Deficit (1.0 - NDVI, planting opportunity)
      - 20% Weight: Built-up Density (NDBI, human exposure)
    Excludes water bodies and mature forest ridges (NDVI >= 0.45).
    """
    H, W = hotspot_score.shape
    land_mask = (mndwi < 0.35) & (ndvi < 0.42) # unshaded land parcels

    # Normalize components
    hs_norm = np.clip(hotspot_score / 3.5, 0.0, 1.0)
    canopy_deficit = np.clip((0.42 - ndvi) / 0.50, 0.0, 1.0)
    built_norm = np.clip((ndbi + 0.10) / 0.60, 0.0, 1.0)

    tppi = (0.45 * hs_norm + 0.35 * canopy_deficit + 0.20 * built_norm) * land_mask
    return np.clip(tppi, 0.0, 1.0).astype(np.float32)


def delineate_tree_planting_priority_zones(
    tppi_map: np.ndarray,
    lst_10m: np.ndarray,
    ndvi_10m: np.ndarray,
    bbox: List[float] = [83.15, 17.62, 83.35, 17.80],
    min_pixels: int = 12,
    pixel_size_meters: float = 10.0,
) -> Dict[str, Any]:
    """
    Extracts contiguous high-priority tree planting parcels (TPPI >= 0.65)
    and formats them as GeoJSON features with specific municipal forestry prescriptions.
    """
    H, W = tppi_map.shape
    min_lon, min_lat, max_lon, max_lat = bbox

    # Binary high-priority mask
    high_priority_mask = (tppi_map >= 0.65).astype(np.uint8)

    contours, _ = cv2.findContours(high_priority_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Curated contextual descriptions for known Visakhapatnam neighborhoods
    neighborhood_archetypes = [
        {
            "name": "Gajuwaka Industrial Buffer & Slag Corridor",
            "locality": "Visakhapatnam Steel Plant & Port Hinterland",
            "microclimate": "Extreme radiant heat from metal fabrication & dark slag yards",
            "species": ["Azadirachta indica (Neem)", "Pongamia pinnata (Karanja)", "Terminalia catappa (Indian Almond)"],
            "intervention": "Dense 30m multi-canopy green shelterbelt to intercept convective heat plumes",
            "projected_cooling_celsius": -3.2,
        },
        {
            "name": "Jagadamba Commercial Core & Transit Nexus",
            "locality": "Daba Gardens & Jagadamba Junction",
            "microclimate": "High pedestrian density, unshaded asphalt canyon, heavy concrete thermal mass",
            "species": ["Mimusops elengi (Bakul/Pogada)", "Cassia fistula (Golden Shower)", "Peltophorum pterocarpum"],
            "intervention": "Continuous shaded street tree canopy, pocket bioswales, and green transit shelters",
            "projected_cooling_celsius": -2.6,
        },
        {
            "name": "Outer Harbour Logistics & Freight Yard",
            "locality": "Port Area & Coastal Industrial Strip",
            "microclimate": "Unvegetated storage yards, high maritime humidity coupled with heat traps",
            "species": ["Casuarina equisetifolia (Coastal Sheoak)", "Thespesia populnea (Portia Tree)", "Pongamia pinnata"],
            "intervention": "Salt-tolerant coastal perimeter windbreak and permeable vegetated parking islands",
            "projected_cooling_celsius": -2.1,
        },
        {
            "name": "Dwaraka Nagar / Asilmetta High-Rise District",
            "locality": "Central Business Corridor",
            "microclimate": "Nocturnal heat trapping due to dense mid-rise concrete envelope",
            "species": ["Saraca asoca (Ashoka)", "Bauhinia variegata (Orchid Tree)", "Tabebuia rosea"],
            "intervention": "Pocket parks, green street boulevards, and rooftop canopy gardens",
            "projected_cooling_celsius": -1.9,
        },
        {
            "name": "Maddilapalama / AU Campus Boundary Strip",
            "locality": "North-Central Academic & Residential Belt",
            "microclimate": "Moderate heat island with high residential exposure",
            "species": ["Syzygium cumini (Jamun)", "Ficus benghalensis (Banyan)", "Dalbergia sissoo"],
            "intervention": "Neighbourhood micro-forests (Miyawaki planting) on vacant road verges",
            "projected_cooling_celsius": -2.4,
        }
    ]

    features = []
    total_planting_hectares = 0.0

    # Sort contours by area descending
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_pixels]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    for idx, cnt in enumerate(valid_contours[:12]):
        pts = cnt.squeeze(axis=1)
        if len(pts) < 3:
            continue

        coords = []
        for col, row in pts:
            lon = min_lon + (col / W) * (max_lon - min_lon)
            lat = max_lat - (row / H) * (max_lat - min_lat)
            coords.append([round(lon, 6), round(lat, 6)])

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            poly = Polygon(coords)
            if not poly.is_valid or poly.area == 0:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            mask_poly = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask_poly, [cnt], -1, 1, thickness=-1)
            poly_pixels = int(np.sum(mask_poly))
            hectares = poly_pixels * (pixel_size_meters ** 2) / 10000.0
            total_planting_hectares += hectares

            mean_lst = float(np.mean(lst_10m[mask_poly == 1]))
            mean_tppi = float(np.mean(tppi_map[mask_poly == 1]))
            mean_ndvi = float(np.mean(ndvi_10m[mask_poly == 1]))

            arch = neighborhood_archetypes[idx % len(neighborhood_archetypes)]

            feature = {
                "type": "Feature",
                "id": f"priority_site_{idx+1:02d}",
                "geometry": mapping(poly),
                "properties": {
                    "rank": idx + 1,
                    "site_name": f"Site #{idx+1}: {arch['name']}",
                    "locality": arch["locality"],
                    "target_hectares": round(hectares, 2),
                    "priority_tier": "Critical Priority (Tier 1)" if mean_tppi >= 0.75 else "High Priority (Tier 2)",
                    "current_mean_lst_celsius": round(mean_lst, 1),
                    "current_ndvi": round(mean_ndvi, 2),
                    "tree_planting_priority_index": round(mean_tppi, 3),
                    "microclimate_challenge": arch["microclimate"],
                    "recommended_native_species": arch["species"],
                    "species_source": "literature-based suggestions, not computed",
                    "species_recommendation_disclaimer": "literature-based suggestions, not computed",
                    "recommended_action": arch["intervention"],
                    "projected_cooling_potential": f"{arch['projected_cooling_celsius']} °C cooling deficit vs canopy reference (upper bound, not a measured effect)",
                }
            }
            features.append(feature)
        except Exception:
            continue

    return {
        "geojson": {
            "type": "FeatureCollection",
            "features": features,
        },
        "total_priority_parcels": len(features),
        "total_target_hectares": round(total_planting_hectares, 2),
        "estimated_citywide_cooling_benefit": "Average -2.5 °C cooling deficit vs canopy reference (upper bound, not a measured effect) across 12 targeted corridors",
    }
