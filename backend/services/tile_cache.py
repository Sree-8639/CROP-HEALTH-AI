"""
backend/services/tile_cache.py:
High-performance tile caching engine for Copernicus Sentinel-2 and Sentinel-1 rasters.
Caches fetched Float32 band arrays, masks, and scene metadata on disk and in memory
to eliminate redundant network calls and accelerate end-to-end live analysis.
"""

import os
import io
import json
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
import numpy as np


class TileCache:
    """
    Disk-backed and memory-backed cache for satellite raster tiles.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_memory_items: int = 50, ttl_seconds: float = 86400 * 7):
        if cache_dir is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(backend_dir, ".cache", "tiles")
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, Tuple[float, Dict[str, Any], Dict[str, np.ndarray]]] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def generate_cache_key(
        sensor: str,
        geometry: Dict[str, Any],
        timerange_str: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a deterministic SHA-256 cache key from query parameters.
        """
        canonical_payload = {
            "sensor": sensor.lower(),
            "geometry": geometry,
            "timerange": timerange_str,
            "extra": extra_params or {},
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        hasher = hashlib.sha256()
        hasher.update(serialized.encode("utf-8"))
        return f"{sensor.lower()}_{hasher.hexdigest()[:24]}"

    def get(self, cache_key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        """
        Retrieves cached raster bands and metadata if present and not expired.
        Returns:
            Tuple of (bands_dict, metadata_dict) or None if cache miss.
        """
        # 1. Check memory cache first
        if cache_key in self._memory_cache:
            ts, meta, bands = self._memory_cache[cache_key]
            if time.time() - ts < self.ttl_seconds:
                # Return deep copy of arrays to prevent external in-place mutations
                copied_bands = {k: v.copy() for k, v in bands.items()}
                return copied_bands, meta.copy()
            else:
                del self._memory_cache[cache_key]

        # 2. Check disk cache
        meta_file = os.path.join(self.cache_dir, f"{cache_key}_meta.json")
        npz_file = os.path.join(self.cache_dir, f"{cache_key}_bands.npz")

        if os.path.exists(meta_file) and os.path.exists(npz_file):
            try:
                # Check file modification time for TTL
                mtime = os.path.getmtime(npz_file)
                if time.time() - mtime > self.ttl_seconds:
                    return None

                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                with np.load(npz_file) as loaded_npz:
                    bands = {k: loaded_npz[k] for k in loaded_npz.files}

                # Store into memory cache (manage LRU limit)
                if len(self._memory_cache) >= self.max_memory_items:
                    # Remove oldest entry
                    oldest_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k][0])
                    del self._memory_cache[oldest_key]

                self._memory_cache[cache_key] = (mtime, meta, bands)
                copied_bands = {k: v.copy() for k, v in bands.items()}
                return copied_bands, meta.copy()
            except Exception:
                return None

        return None

    def put(self, cache_key: str, bands: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
        """
        Stores raster bands and metadata into both memory and disk caches.
        """
        now = time.time()
        # Save to memory cache
        if len(self._memory_cache) >= self.max_memory_items:
            oldest_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k][0])
            del self._memory_cache[oldest_key]

        self._memory_cache[cache_key] = (now, metadata.copy(), {k: v.copy() for k, v in bands.items()})

        # Save to disk cache
        try:
            meta_file = os.path.join(self.cache_dir, f"{cache_key}_meta.json")
            npz_file = os.path.join(self.cache_dir, f"{cache_key}_bands.npz")

            clean_meta = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            clean_meta["cached_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(clean_meta, f, indent=2)

            np.savez_compressed(npz_file, **bands)
        except Exception:
            # Non-fatal if disk write fails
            pass


# Global singleton cache
tile_cache = TileCache()

