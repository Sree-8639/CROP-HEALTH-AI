"""
ml/heat/landsat_lst.py:
Handles Landsat 8/9 Collection 2 Level-2 Surface Temperature (ST_B10) band data:
Calibration formula:
  LST_Kelvin = DN * 0.00341802 + 149.0
  LST_Celsius = LST_Kelvin - 273.15
Provides 30m resolution thermal infrared base layers for Visakhapatnam urban heat modeling.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "real_visakhapatnam_heat_benchmark.npz")


# Official USGS Landsat Collection 2 Level-2 Surface Temperature scaling coefficients
LST_SCALE_FACTOR = 0.00341802
LST_ADD_OFFSET = 149.0
KELVIN_TO_CELSIUS = 273.15


def dn_to_lst_celsius(dn_array: np.ndarray) -> np.ndarray:
    """
    Converts raw Landsat 8/9 Collection 2 Level-2 ST_B10 digital numbers (DN) to Land Surface Temperature in °C.
    
    Formula:
      LST_K = DN * 0.00341802 + 149.0
      LST_C = LST_K - 273.15
    """
    dn_arr = np.asarray(dn_array, dtype=np.float32)
    valid_mask = dn_arr > 0
    lst_kelvin = dn_arr * LST_SCALE_FACTOR + LST_ADD_OFFSET
    lst_celsius = np.where(valid_mask, lst_kelvin - KELVIN_TO_CELSIUS, np.nan)
    if np.ndim(dn_array) == 0:
        return float(lst_celsius)
    return lst_celsius.astype(np.float32)


def lst_celsius_to_dn(lst_celsius: np.ndarray) -> np.ndarray:
    """Inverts LST (°C) back to 16-bit integer DN."""
    c_arr = np.asarray(lst_celsius, dtype=np.float32)
    lst_kelvin = c_arr + KELVIN_TO_CELSIUS
    dn = (lst_kelvin - LST_ADD_OFFSET) / LST_SCALE_FACTOR
    res = np.clip(np.nan_to_num(dn, nan=0.0), 0, 65535).astype(np.uint16)
    if np.ndim(lst_celsius) == 0:
        return int(res)
    return res


class LandsatLSTTile:
    """
    Represents a Landsat 8/9 30m Surface Temperature acquisition over Visakhapatnam.
    
    Coordinates:
      Visakhapatnam AOI: [83.15, 17.62, 83.35, 17.80]
      Features: Industrial corridor (Steel Plant/Port), Dense Urban core,
      Forested ridges (Kailasagiri, Simhachalam), and Bay of Bengal coastline.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (256, 256), # 256 pixels at 30m = 7.68 km
        season: str = "summer", # "summer" (April-May) or "winter" (Dec-Jan)
        diurnal: str = "day",   # "day" (10:30 AM overpass) or "night" (10:00 PM overpass)
        random_seed: int = 42,
    ):
        self.grid_shape = grid_shape
        self.season = season.lower()
        self.diurnal = diurnal.lower()
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        if os.path.exists(REAL_DATA_PATH):
            data = np.load(REAL_DATA_PATH)
            if self.diurnal == "night" and "lst_modis_night" in data:
                lst = data["lst_modis_night"]
            elif self.season == "summer" and "lst_summer_30m" in data:
                lst = data["lst_summer_30m"]
            else:
                lst = data["lst_winter_30m"]
            
            # If grid_shape is different from 256x256, resize
            if lst.shape != self.grid_shape:
                import cv2
                lst = cv2.resize(lst.astype(np.float32), (self.grid_shape[1], self.grid_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            self.lst_celsius = lst.astype(np.float32)
            self.raw_dn = lst_celsius_to_dn(self.lst_celsius)
        else:
            self.raw_dn, self.lst_celsius = self._generate_visakhapatnam_lst()

    def _generate_visakhapatnam_lst(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes physically realistic 30m Landsat thermal field over Visakhapatnam
        matching real-world microclimates:
          - Heavy Industrial / Port (Gajuwaka, Outer Harbour, Steel Plant)
          - High-density Commercial / Residential (Jagadamba, Dwaraka Nagar)
          - Vegetated Hill Ranges (Kailasagiri, Simhachalam Reserve Forest)
          - Bay of Bengal Coastal Water Boundary
        """
        H, W = self.grid_shape
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)

        # Baseline regional background temperature (°C)
        if self.season == "summer":
            base_temp = 37.5 if self.diurnal == "day" else 28.5
        else:
            base_temp = 27.0 if self.diurnal == "day" else 20.0

        lst = np.full((H, W), base_temp, dtype=np.float32)

        # 1. Industrial Zone (South-West quadrant: Steel Plant & Port) -> Strong positive thermal anomaly
        ind_mask = (xx < 0.45) & (yy > 0.50)
        if self.diurnal == "day":
            ind_anomaly = 9.5 * np.exp(-((xx - 0.25)**2 + (yy - 0.70)**2) / 0.05)
        else:
            # High thermal mass keeps industrial concrete/slag warm at night
            ind_anomaly = 5.0 * np.exp(-((xx - 0.25)**2 + (yy - 0.70)**2) / 0.06)
        lst += ind_anomaly.astype(np.float32)

        # 2. Dense Commercial Urban Core (Central East: Jagadamba, Siripuram, RTC)
        urban_anomaly = (4.5 if self.diurnal == "day" else 3.8) * np.exp(-((xx - 0.65)**2 + (yy - 0.55)**2) / 0.03)
        lst += urban_anomaly.astype(np.float32)

        # 3. Forested Hill Ranges (Simhachalam NW and Kailasagiri NE ridges) -> Strong cooling refuge
        ridge1 = np.exp(-((xx - 0.35)**2 + (yy - 0.25)**2) / 0.02) # Simhachalam
        ridge2 = np.exp(-((xx - 0.80)**2 + (yy - 0.25)**2) / 0.02) # Kailasagiri
        cooling = (6.5 if self.diurnal == "day" else 3.5) * (ridge1 + ridge2)
        lst -= cooling.astype(np.float32)

        # 4. Coastal Interface (Far East edge: Bay of Bengal)
        sea_mask = xx > 0.88
        sea_temp = 29.0 if (self.season == "summer" and self.diurnal == "day") else 24.5
        lst[sea_mask] = sea_temp

        # Natural sensor noise and atmospheric micro-variations
        noise = self.rng.normal(0, 0.45, (H, W))
        lst += noise.astype(np.float32)

        # Convert Celsius to official Landsat ST_B10 raw DN
        raw_dn = lst_celsius_to_dn(lst)
        # Recalculate exact Celsius through official formula to ensure bit-level numerical consistency
        calibrated_lst = dn_to_lst_celsius(raw_dn)

        return raw_dn, calibrated_lst
