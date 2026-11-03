import math
import re
import httpx
from typing import Dict, Any, List, Optional, Tuple

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "CropHealthMonitoringApp/2.0 (contact: support@crophealth.ai)"

# Comprehensive built-in agricultural gazetteer for instant, reliable lookups
# Covers major agricultural regions in Andhra Pradesh, India, and global reference sites
AGRICULTURAL_GAZETTEER = {
    # Andhra Pradesh & Telangana Agricultural Centers
    "tadepalligudem": {
        "display_name": "Tadepalligudem, West Godavari District, Andhra Pradesh, India",
        "latitude": 16.810244,
        "longitude": 81.526592,
        "type": "city",
        "address": {"city": "Tadepalligudem", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "bhimavaram": {
        "display_name": "Bhimavaram, West Godavari District, Andhra Pradesh, India",
        "latitude": 16.544893,
        "longitude": 81.521241,
        "type": "city",
        "address": {"city": "Bhimavaram", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "vijayawada": {
        "display_name": "Vijayawada, Krishna District, Andhra Pradesh, India",
        "latitude": 16.506174,
        "longitude": 80.648015,
        "type": "city",
        "address": {"city": "Vijayawada", "district": "Krishna", "state": "Andhra Pradesh", "country": "India"}
    },
    "guntur": {
        "display_name": "Guntur, Guntur District, Andhra Pradesh, India",
        "latitude": 16.306652,
        "longitude": 80.436540,
        "type": "city",
        "address": {"city": "Guntur", "district": "Guntur", "state": "Andhra Pradesh", "country": "India"}
    },
    "maruteru": {
        "display_name": "Maruteru Agricultural Research Station, West Godavari, Andhra Pradesh, India",
        "latitude": 16.634100,
        "longitude": 81.745500,
        "type": "village",
        "address": {"village": "Maruteru", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "rajahmundry": {
        "display_name": "Rajahmundry, East Godavari District, Andhra Pradesh, India",
        "latitude": 17.000465,
        "longitude": 81.804034,
        "type": "city",
        "address": {"city": "Rajahmundry", "district": "East Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "eluru": {
        "display_name": "Eluru, Eluru District, Andhra Pradesh, India",
        "latitude": 16.710661,
        "longitude": 81.095243,
        "type": "city",
        "address": {"city": "Eluru", "district": "Eluru", "state": "Andhra Pradesh", "country": "India"}
    },
    "tanuku": {
        "display_name": "Tanuku, West Godavari District, Andhra Pradesh, India",
        "latitude": 16.757022,
        "longitude": 81.681816,
        "type": "city",
        "address": {"city": "Tanuku", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "kurnool": {
        "display_name": "Kurnool, Kurnool District, Andhra Pradesh, India",
        "latitude": 15.828126,
        "longitude": 78.037273,
        "type": "city",
        "address": {"city": "Kurnool", "district": "Kurnool", "state": "Andhra Pradesh", "country": "India"}
    },
    "anantapur": {
        "display_name": "Anantapur, Anantapur District, Andhra Pradesh, India",
        "latitude": 14.681888,
        "longitude": 77.600591,
        "type": "city",
        "address": {"city": "Anantapur", "district": "Anantapur", "state": "Andhra Pradesh", "country": "India"}
    },
    "amaravati": {
        "display_name": "Amaravati, Guntur District, Andhra Pradesh, India",
        "latitude": 16.518600,
        "longitude": 80.620000,
        "type": "city",
        "address": {"city": "Amaravati", "district": "Guntur", "state": "Andhra Pradesh", "country": "India"}
    },
    "kakinada": {
        "display_name": "Kakinada, Kakinada District, Andhra Pradesh, India",
        "latitude": 16.989065,
        "longitude": 82.247465,
        "type": "city",
        "address": {"city": "Kakinada", "district": "Kakinada", "state": "Andhra Pradesh", "country": "India"}
    },
    "visakhapatnam": {
        "display_name": "Visakhapatnam, Andhra Pradesh, India",
        "latitude": 17.686816,
        "longitude": 83.218482,
        "type": "city",
        "address": {"city": "Visakhapatnam", "state": "Andhra Pradesh", "country": "India"}
    },
    "tenali": {
        "display_name": "Tenali, Guntur District, Andhra Pradesh, India",
        "latitude": 16.243000,
        "longitude": 80.640000,
        "type": "city",
        "address": {"city": "Tenali", "district": "Guntur", "state": "Andhra Pradesh", "country": "India"}
    },
    "machilipatnam": {
        "display_name": "Machilipatnam, Krishna District, Andhra Pradesh, India",
        "latitude": 16.180000,
        "longitude": 81.130000,
        "type": "city",
        "address": {"city": "Machilipatnam", "district": "Krishna", "state": "Andhra Pradesh", "country": "India"}
    },
    "palakollu": {
        "display_name": "Palakollu, West Godavari District, Andhra Pradesh, India",
        "latitude": 16.520000,
        "longitude": 81.730000,
        "type": "city",
        "address": {"city": "Palakollu", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "narsapur": {
        "display_name": "Narsapur, West Godavari District, Andhra Pradesh, India",
        "latitude": 16.430000,
        "longitude": 81.700000,
        "type": "city",
        "address": {"city": "Narsapur", "district": "West Godavari", "state": "Andhra Pradesh", "country": "India"}
    },
    "narasaraopet": {
        "display_name": "Narasaraopet, Palnadu District, Andhra Pradesh, India",
        "latitude": 16.230000,
        "longitude": 80.050000,
        "type": "city",
        "address": {"city": "Narasaraopet", "district": "Palnadu", "state": "Andhra Pradesh", "country": "India"}
    },
    "ongole": {
        "display_name": "Ongole, Prakasam District, Andhra Pradesh, India",
        "latitude": 15.500000,
        "longitude": 80.050000,
        "type": "city",
        "address": {"city": "Ongole", "district": "Prakasam", "state": "Andhra Pradesh", "country": "India"}
    },
    "nellore": {
        "display_name": "Nellore, Sri Potti Sriramulu Nellore District, Andhra Pradesh, India",
        "latitude": 14.442600,
        "longitude": 79.986500,
        "type": "city",
        "address": {"city": "Nellore", "district": "Nellore", "state": "Andhra Pradesh", "country": "India"}
    },
    "tirupati": {
        "display_name": "Tirupati, Tirupati District, Andhra Pradesh, India",
        "latitude": 13.628800,
        "longitude": 79.419200,
        "type": "city",
        "address": {"city": "Tirupati", "district": "Tirupati", "state": "Andhra Pradesh", "country": "India"}
    },
    "chittoor": {
        "display_name": "Chittoor, Chittoor District, Andhra Pradesh, India",
        "latitude": 13.217200,
        "longitude": 79.100300,
        "type": "city",
        "address": {"city": "Chittoor", "district": "Chittoor", "state": "Andhra Pradesh", "country": "India"}
    },
    "kadapa": {
        "display_name": "Kadapa (Cuddapah), YSR District, Andhra Pradesh, India",
        "latitude": 14.467300,
        "longitude": 78.824200,
        "type": "city",
        "address": {"city": "Kadapa", "district": "YSR", "state": "Andhra Pradesh", "country": "India"}
    },
    "srikakulam": {
        "display_name": "Srikakulam, Srikakulam District, Andhra Pradesh, India",
        "latitude": 18.294900,
        "longitude": 83.893800,
        "type": "city",
        "address": {"city": "Srikakulam", "district": "Srikakulam", "state": "Andhra Pradesh", "country": "India"}
    },
    "vizianagaram": {
        "display_name": "Vizianagaram, Vizianagaram District, Andhra Pradesh, India",
        "latitude": 18.106700,
        "longitude": 83.395600,
        "type": "city",
        "address": {"city": "Vizianagaram", "district": "Vizianagaram", "state": "Andhra Pradesh", "country": "India"}
    },
    "hyderabad": {
        "display_name": "Hyderabad, Telangana, India",
        "latitude": 17.385044,
        "longitude": 78.486671,
        "type": "city",
        "address": {"city": "Hyderabad", "state": "Telangana", "country": "India"}
    },
    "warangal": {
        "display_name": "Warangal, Telangana, India",
        "latitude": 17.968901,
        "longitude": 79.594054,
        "type": "city",
        "address": {"city": "Warangal", "state": "Telangana", "country": "India"}
    },
    "nizamabad": {
        "display_name": "Nizamabad, Telangana, India",
        "latitude": 18.672500,
        "longitude": 78.094100,
        "type": "city",
        "address": {"city": "Nizamabad", "state": "Telangana", "country": "India"}
    },
    "karimnagar": {
        "display_name": "Karimnagar, Telangana, India",
        "latitude": 18.438600,
        "longitude": 79.128800,
        "type": "city",
        "address": {"city": "Karimnagar", "state": "Telangana", "country": "India"}
    },
    "khammam": {
        "display_name": "Khammam, Telangana, India",
        "latitude": 17.247300,
        "longitude": 80.151400,
        "type": "city",
        "address": {"city": "Khammam", "state": "Telangana", "country": "India"}
    },
    # Major National Agricultural Centers
    "delhi": {
        "display_name": "New Delhi, Delhi, India",
        "latitude": 28.613939,
        "longitude": 77.209021,
        "type": "city",
        "address": {"city": "New Delhi", "state": "Delhi", "country": "India"}
    },
    "bengaluru": {
        "display_name": "Bengaluru, Karnataka, India",
        "latitude": 12.971599,
        "longitude": 77.594563,
        "type": "city",
        "address": {"city": "Bengaluru", "state": "Karnataka", "country": "India"}
    },
    "bangalore": {
        "display_name": "Bengaluru (Bangalore), Karnataka, India",
        "latitude": 12.971599,
        "longitude": 77.594563,
        "type": "city",
        "address": {"city": "Bengaluru", "state": "Karnataka", "country": "India"}
    },
    "chennai": {
        "display_name": "Chennai, Tamil Nadu, India",
        "latitude": 13.082680,
        "longitude": 80.270718,
        "type": "city",
        "address": {"city": "Chennai", "state": "Tamil Nadu", "country": "India"}
    },
    "mumbai": {
        "display_name": "Mumbai, Maharashtra, India",
        "latitude": 19.076090,
        "longitude": 72.877426,
        "type": "city",
        "address": {"city": "Mumbai", "state": "Maharashtra", "country": "India"}
    },
    "pune": {
        "display_name": "Pune, Maharashtra, India",
        "latitude": 18.520430,
        "longitude": 73.856744,
        "type": "city",
        "address": {"city": "Pune", "state": "Maharashtra", "country": "India"}
    },
    "nagpur": {
        "display_name": "Nagpur (Orange Basin), Maharashtra, India",
        "latitude": 21.145800,
        "longitude": 79.088200,
        "type": "city",
        "address": {"city": "Nagpur", "state": "Maharashtra", "country": "India"}
    },
    "nashik": {
        "display_name": "Nashik (Grape & Onion Hub), Maharashtra, India",
        "latitude": 19.997500,
        "longitude": 73.789800,
        "type": "city",
        "address": {"city": "Nashik", "state": "Maharashtra", "country": "India"}
    },
    "ludhiana": {
        "display_name": "Ludhiana (Punjab Wheat & Rice Basin), Punjab, India",
        "latitude": 30.901000,
        "longitude": 75.857300,
        "type": "city",
        "address": {"city": "Ludhiana", "state": "Punjab", "country": "India"}
    },
    "amritsar": {
        "display_name": "Amritsar, Punjab, India",
        "latitude": 31.634000,
        "longitude": 74.872300,
        "type": "city",
        "address": {"city": "Amritsar", "state": "Punjab", "country": "India"}
    },
    "karnal": {
        "display_name": "Karnal (Central Soil Salinity Research Hub), Haryana, India",
        "latitude": 29.685700,
        "longitude": 76.990500,
        "type": "city",
        "address": {"city": "Karnal", "state": "Haryana", "country": "India"}
    },
    "thanjavur": {
        "display_name": "Thanjavur (Cauvery Delta Rice Bowl), Tamil Nadu, India",
        "latitude": 10.787000,
        "longitude": 79.137800,
        "type": "city",
        "address": {"city": "Thanjavur", "state": "Tamil Nadu", "country": "India"}
    },
    "coimbatore": {
        "display_name": "Coimbatore, Tamil Nadu, India",
        "latitude": 11.016800,
        "longitude": 76.955800,
        "type": "city",
        "address": {"city": "Coimbatore", "state": "Tamil Nadu", "country": "India"}
    },
    "lucknow": {
        "display_name": "Lucknow, Uttar Pradesh, India",
        "latitude": 26.846700,
        "longitude": 80.946200,
        "type": "city",
        "address": {"city": "Lucknow", "state": "Uttar Pradesh", "country": "India"}
    },
    "varanasi": {
        "display_name": "Varanasi, Uttar Pradesh, India",
        "latitude": 25.317600,
        "longitude": 82.973900,
        "type": "city",
        "address": {"city": "Varanasi", "state": "Uttar Pradesh", "country": "India"}
    },
    "patna": {
        "display_name": "Patna, Bihar, India",
        "latitude": 25.594100,
        "longitude": 85.137600,
        "type": "city",
        "address": {"city": "Patna", "state": "Bihar", "country": "India"}
    },
    "jaipur": {
        "display_name": "Jaipur, Rajasthan, India",
        "latitude": 26.912400,
        "longitude": 75.787300,
        "type": "city",
        "address": {"city": "Jaipur", "state": "Rajasthan", "country": "India"}
    },
    "ahmedabad": {
        "display_name": "Ahmedabad, Gujarat, India",
        "latitude": 23.022500,
        "longitude": 72.571400,
        "type": "city",
        "address": {"city": "Ahmedabad", "state": "Gujarat", "country": "India"}
    },
    # International Agricultural Reference Hubs
    "champagne": {
        "display_name": "Champagne-Ardenne Agricultural Parcel, Marne, France",
        "latitude": 48.955000,
        "longitude": 4.355000,
        "type": "agricultural_parcel",
        "address": {"region": "Champagne-Ardenne", "department": "Marne", "country": "France"}
    },
    "paris": {
        "display_name": "Paris, Île-de-France, France",
        "latitude": 48.856614,
        "longitude": 2.352222,
        "type": "city",
        "address": {"city": "Paris", "country": "France"}
    },
    "iowa": {
        "display_name": "Story County Corn Belt, Iowa, United States",
        "latitude": 42.030800,
        "longitude": -93.581500,
        "type": "agricultural_region",
        "address": {"county": "Story County", "state": "Iowa", "country": "United States"}
    },
    "des moines": {
        "display_name": "Des Moines, Iowa, United States",
        "latitude": 41.586836,
        "longitude": -93.624959,
        "type": "city",
        "address": {"city": "Des Moines", "state": "Iowa", "country": "United States"}
    },
    "salta": {
        "display_name": "Salta Gran Chaco Soybean Basin, Salta, Argentina",
        "latitude": -24.782100,
        "longitude": -65.423200,
        "type": "agricultural_basin",
        "address": {"province": "Salta", "country": "Argentina"}
    },
    "riverina": {
        "display_name": "Riverina Murrumbidgee Irrigated Farm, New South Wales, Australia",
        "latitude": -34.502000,
        "longitude": 146.401000,
        "type": "agricultural_region",
        "address": {"region": "Riverina", "state": "New South Wales", "country": "Australia"}
    },
    "london": {
        "display_name": "London, Greater London, England, United Kingdom",
        "latitude": 51.507446,
        "longitude": -0.127765,
        "type": "city",
        "address": {"city": "London", "country": "United Kingdom"}
    },
}

def parse_coordinate_query(query: str) -> Optional[Tuple[float, float]]:
    """
    Checks if user input is already a latitude, longitude coordinate pair.
    e.g., '16.81, 81.52', '16.810244, 81.526592', '16.81N 81.52E'
    """
    clean = query.strip()
    # Match standard formats like "16.81, 81.52" or "16.81 81.52"
    coord_match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*[,;/ ]\s*([+-]?\d+(?:\.\d+)?)$", clean)
    if coord_match:
        try:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    # Match format like "16.81N, 81.52E"
    hem_match = re.match(r"^(\d+(?:\.\d+)?)\s*([NSns])\s*[,;/ ]\s*(\d+(?:\.\d+)?)\s*([EWew])$", clean)
    if hem_match:
        try:
            lat_val = float(hem_match.group(1))
            lat_hem = hem_match.group(2).upper()
            lat = -lat_val if lat_hem == 'S' else lat_val

            lon_val = float(hem_match.group(3))
            lon_hem = hem_match.group(4).upper()
            lon = -lon_val if lon_hem == 'W' else lon_val

            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    return None

def lookup_gazetteer(query: str) -> Optional[Dict[str, Any]]:
    """
    Looks up clean query in the local agricultural gazetteer with exact & partial matching.
    """
    clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if not clean:
        return None

    # 1. Exact normalized key match
    if clean in AGRICULTURAL_GAZETTEER:
        match = AGRICULTURAL_GAZETTEER[clean]
        return {
            "query": query,
            "display_name": match["display_name"],
            "latitude": match["latitude"],
            "longitude": match["longitude"],
            "type": match["type"],
            "address": match["address"],
        }

    # 2. Check for key contained inside query (e.g. 'Tadepalligudem town' or 'Guntur field')
    for key, match in AGRICULTURAL_GAZETTEER.items():
        if key in clean or clean in key:
            return {
                "query": query,
                "display_name": match["display_name"],
                "latitude": match["latitude"],
                "longitude": match["longitude"],
                "type": match["type"],
                "address": match["address"],
            }

    return None

async def geocode_query(query: str) -> Dict[str, Any]:
    """
    Geocodes a query string (city, village, district, state, place name, or coordinates)
    into latitude, longitude, and formatted display name with multi-tiered resolution:
    1. Direct coordinate parsing (e.g. '16.81, 81.52')
    2. OpenStreetMap Nominatim live search with 6-second timeout
    3. Built-in high-precision Agricultural Gazetteer fallback for guaranteed reliability
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Please enter a location name (city, village, district, or place).")

    # 1. Check if user typed direct coordinates
    coords = parse_coordinate_query(clean_query)
    if coords:
        lat, lon = coords
        # Try quick reverse lookup or construct friendly display
        return {
            "query": clean_query,
            "display_name": f"Coordinate Location ({lat:.5f}° N/S, {lon:.5f}° E/W)",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "type": "coordinate",
            "address": {"coordinates": f"{lat:.5f}, {lon:.5f}"},
        }

    # 2. Try Nominatim with a fast timeout
    nominatim_error: Optional[str] = None
    try:
        params = {
            "q": clean_query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en",
        }

        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(NOMINATIM_SEARCH_URL, params=params, headers=headers)
            if response.status_code == 200:
                results = response.json()
                if results and len(results) > 0:
                    top_result = results[0]
                    lat = float(top_result["lat"])
                    lon = float(top_result["lon"])
                    return {
                        "query": query,
                        "display_name": top_result.get("display_name", clean_query),
                        "latitude": round(lat, 6),
                        "longitude": round(lon, 6),
                        "type": top_result.get("type", "place"),
                        "address": top_result.get("address", {}),
                    }
    except Exception as exc:
        nominatim_error = str(exc)

    # 3. Fallback to Agricultural Gazetteer
    gazetteer_match = lookup_gazetteer(clean_query)
    if gazetteer_match:
        return gazetteer_match

    # 4. If neither worked, provide a helpful, friendly message with popular examples
    examples = "Tadepalligudem, Bhimavaram, Vijayawada, Guntur, Rajahmundry, Eluru, Kurnool, Champagne, or coordinates like '16.81, 81.52'"
    raise LookupError(
        f"We couldn't locate '{clean_query}'. Please check the spelling or try searching a nearby town, district, or coordinates (e.g. {examples})."
    )

async def get_location_details(lat: float, lon: float) -> Dict[str, Any]:
    """
    Reverse geocodes latitude and longitude into human-readable place information.
    """
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError("Latitude must be between -90 and 90, and longitude between -180 and 180.")

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(NOMINATIM_REVERSE_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "display_name": data.get("display_name", f"{lat:.5f}, {lon:.5f}"),
                "address": data.get("address", {}),
            }
    except Exception:
        # Check gazetteer reverse approximation
        for key, match in AGRICULTURAL_GAZETTEER.items():
            dist = math.sqrt((lat - match["latitude"])**2 + (lon - match["longitude"])**2)
            if dist < 0.05:  # within ~5 km
                return {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "display_name": match["display_name"],
                    "address": match["address"],
                }

        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "display_name": f"Agricultural Parcel ({lat:.5f}° N/S, {lon:.5f}° E/W)",
            "address": {},
        }

def calculate_aoi(lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
    """
    Computes an Area of Interest (AOI) with bounding box and GeoJSON Polygon geometry.
    This GeoJSON geometry representation is directly compatible with Sentinel-2 / STAC satellite queries.
    """
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError("Coordinates out of geographic range.")
    if radius_km <= 0:
        raise ValueError("Radius must be greater than 0.")

    # Earth radius in kilometers (approximate mean radius)
    r_earth = 6371.0

    # 1 degree of latitude is approx 111.32 km
    delta_lat = radius_km / 111.32
    # 1 degree of longitude depends on latitude
    lat_rad = math.radians(lat)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        delta_lon = radius_km / 111.32
    else:
        delta_lon = radius_km / (111.32 * cos_lat)

    min_lat = max(-90.0, round(lat - delta_lat, 6))
    max_lat = min(90.0, round(lat + delta_lat, 6))
    min_lon = max(-180.0, round(lon - delta_lon, 6))
    max_lon = min(180.0, round(lon + delta_lon, 6))

    # Generate a smooth 36-point circle approximation polygon for geographic AOI
    # [longitude, latitude] pairs in standard GeoJSON counter-clockwise order
    circle_coordinates: List[List[float]] = []
    num_points = 36
    angular_dist = radius_km / r_earth

    for i in range(num_points):
        bearing = math.radians((i * 360.0) / num_points)
        pt_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist)
            + math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing)
        )
        pt_lon_rad = math.radians(lon) + math.atan2(
            math.sin(bearing) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(pt_lat_rad),
        )
        pt_lat = round(math.degrees(pt_lat_rad), 6)
        pt_lon = round(math.degrees(pt_lon_rad), 6)
        circle_coordinates.append([pt_lon, pt_lat])

    # Close the ring
    circle_coordinates.append(circle_coordinates[0])

    # Bounding box polygon (rectangle)
    bbox_coordinates = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]

    return {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "radius_km": radius_km,
        "radius_meters": radius_km * 1000.0,
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "geometry": {
            "type": "Polygon",
            "coordinates": [circle_coordinates],
        },
        "bbox_geometry": {
            "type": "Polygon",
            "coordinates": [bbox_coordinates],
        },
        "area_sq_km": round(math.pi * (radius_km ** 2), 2),
    }
