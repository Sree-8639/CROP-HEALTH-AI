"use client";

import { useEffect, useRef, useState } from "react";
import L from "leaflet";

export type LayerType =
  | "satellite"
  | "ndvi"
  | "ndmi"
  | "ndre"
  | "evi"
  | "condition"
  | "change"
  | "sentinel1_vv"
  | "sentinel1_vh"
  | "sentinel1_ratio"
  | "quality_mask"
  | "stress"
  | "none";

interface FieldMapProps {
  latitude: number;
  longitude: number;
  locationName: string;
  radiusKm: number;
  onCoordinatesChange?: (lat: number, lon: number) => void;
  satelliteOverlayImage?: string | null;
  ndviOverlayImage?: string | null;
  ndmiOverlayImage?: string | null;
  ndreOverlayImage?: string | null;
  eviOverlayImage?: string | null;
  conditionOverlayImage?: string | null;
  changeOverlayImage?: string | null;
  sentinel1VvOverlayImage?: string | null;
  sentinel1VhOverlayImage?: string | null;
  sentinel1RatioOverlayImage?: string | null;
  qualityMaskOverlayImage?: string | null;
  stressOverlayImage?: string | null;
  selectedLayer?: LayerType;
  onLayerChange?: (layer: LayerType) => void;
  satelliteOverlayBounds?: [[number, number], [number, number]] | null;
}

export default function FieldMap({
  latitude,
  longitude,
  locationName,
  radiusKm,
  onCoordinatesChange,
  satelliteOverlayImage,
  ndviOverlayImage,
  ndmiOverlayImage,
  ndreOverlayImage,
  eviOverlayImage,
  conditionOverlayImage,
  changeOverlayImage,
  sentinel1VvOverlayImage,
  sentinel1VhOverlayImage,
  sentinel1RatioOverlayImage,
  qualityMaskOverlayImage,
  stressOverlayImage,
  selectedLayer: externalSelectedLayer,
  onLayerChange,
  satelliteOverlayBounds,
}: FieldMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const circleRef = useRef<L.Circle | null>(null);
  const overlayRef = useRef<L.ImageOverlay | null>(null);

  // Active overlay layer: "satellite" | "ndvi" | "ndmi" | "ndre" | "evi" | "none"
  const [internalLayer, setInternalLayer] = useState<LayerType>("satellite");
  const activeLayer = externalSelectedLayer !== undefined ? externalSelectedLayer : internalLayer;

  const handleSelectLayer = (layer: LayerType) => {
    setInternalLayer(layer);
    if (onLayerChange) {
      onLayerChange(layer);
    }
  };

  // Automatically switch to available index overlay if satellite is not yet active
  useEffect(() => {
    if (externalSelectedLayer === undefined) {
      if (ndviOverlayImage) {
        setInternalLayer("ndvi");
      } else if (satelliteOverlayImage) {
        setInternalLayer("satellite");
      }
    }
  }, [ndviOverlayImage, satelliteOverlayImage, externalSelectedLayer]);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Base Layers
    const streetLayer = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }
    );

    const satelliteLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri &mdash; Source: Esri, USDA, USGS, AeroGRID, IGN, and GIS User Community",
        maxZoom: 18,
      }
    );

    // Initialize Leaflet map
    const map = L.map(mapContainerRef.current, {
      center: [latitude, longitude],
      zoom: 14,
      layers: [satelliteLayer],
      zoomControl: true,
    });

    // Layer control for switching between Satellite and Normal Street Map
    const baseMaps = {
      "🛰️ Satellite View": satelliteLayer,
      "🗺️ Normal Map": streetLayer,
    };
    L.control.layers(baseMaps, undefined, { position: "topright" }).addTo(map);

    // Map click listener to allow adjusting location marker
    if (onCoordinatesChange) {
      map.on("click", (e: L.LeafletMouseEvent) => {
        onCoordinatesChange(e.latlng.lat, e.latlng.lng);
      });
    }

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      markerRef.current = null;
      circleRef.current = null;
      overlayRef.current = null;
    };
  }, []); // Run once on mount

  // Update marker, circle, and view when coordinates or radius change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    map.setView([latitude, longitude], map.getZoom() || 14, { animate: true });

    // Custom SVG Marker Icon
    const customIcon = L.divIcon({
      className: "custom-field-pin",
      html: `
        <div style="position: relative; transform: translate(-50%, -100%);">
          <svg width="34" height="42" viewBox="0 0 34 42" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 0C7.61116 0 0 7.61116 0 17C0 27.5 17 42 17 42C17 42 34 27.5 34 17C34 7.61116 26.3888 0 17 0Z" fill="#10B981"/>
            <circle cx="17" cy="16" r="7" fill="#064E3B"/>
            <circle cx="17" cy="16" r="4" fill="#FFFFFF"/>
          </svg>
        </div>
      `,
      iconSize: [34, 42],
      iconAnchor: [17, 42],
      popupAnchor: [0, -38],
    });

    const popupHtml = `
      <div style="font-family: inherit; font-size: 13px; line-height: 1.4;">
        <strong style="color: #34D399; font-size: 14px; display: block; margin-bottom: 4px;">📍 ${locationName}</strong>
        <div style="color: #CBD5E1; margin-bottom: 2px;"><strong>Latitude:</strong> ${latitude.toFixed(6)}</div>
        <div style="color: #CBD5E1; margin-bottom: 4px;"><strong>Longitude:</strong> ${longitude.toFixed(6)}</div>
        <div style="display: inline-block; background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 6px; border-radius: 4px; color: #6EE7B7; font-size: 11px;">
          AOI Radius: ${radiusKm} km
        </div>
      </div>
    `;

    // Manage Marker
    if (markerRef.current) {
      markerRef.current.setLatLng([latitude, longitude]);
      markerRef.current.bindPopup(popupHtml);
    } else {
      const marker = L.marker([latitude, longitude], { icon: customIcon }).addTo(map);
      marker.bindPopup(popupHtml);
      markerRef.current = marker;
    }

    // Manage AOI Area Circle (radius in meters)
    const radiusMeters = radiusKm * 1000;
    if (circleRef.current) {
      circleRef.current.setLatLng([latitude, longitude]);
      circleRef.current.setRadius(radiusMeters);
    } else {
      const circle = L.circle([latitude, longitude], {
        radius: radiusMeters,
        color: "#10B981",
        weight: 2,
        dashArray: "4, 6",
        fillColor: "#10B981",
        fillOpacity: 0.18,
      }).addTo(map);
      circleRef.current = circle;
    }

    // Fit map bounds smoothly around the circle if not zoomed into overlay
    if (circleRef.current && !overlayRef.current) {
      const bounds = circleRef.current.getBounds();
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15, animate: true });
    }
  }, [latitude, longitude, radiusKm, locationName]);

  // Manage Image Overlay (Satellite RGB vs NDVI / NDMI / NDRE / EVI Layers)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }

    let activeImage: string | null = null;
    if (activeLayer === "satellite") {
      activeImage = satelliteOverlayImage || null;
    } else if (activeLayer === "ndvi") {
      activeImage = ndviOverlayImage || null;
    } else if (activeLayer === "ndmi") {
      activeImage = ndmiOverlayImage || null;
    } else if (activeLayer === "ndre") {
      activeImage = ndreOverlayImage || null;
    } else if (activeLayer === "evi") {
      activeImage = eviOverlayImage || null;
    } else if (activeLayer === "condition") {
      activeImage = conditionOverlayImage || null;
    } else if (activeLayer === "change") {
      activeImage = changeOverlayImage || null;
    } else if (activeLayer === "sentinel1_vv") {
      activeImage = sentinel1VvOverlayImage || null;
    } else if (activeLayer === "sentinel1_vh") {
      activeImage = sentinel1VhOverlayImage || null;
    } else if (activeLayer === "sentinel1_ratio") {
      activeImage = sentinel1RatioOverlayImage || null;
    } else if (activeLayer === "quality_mask") {
      activeImage = qualityMaskOverlayImage || null;
    } else if (activeLayer === "stress") {
      activeImage = stressOverlayImage || null;
    }

    if (activeImage) {
      let bounds: L.LatLngBoundsExpression;

      if (satelliteOverlayBounds) {
        bounds = satelliteOverlayBounds;
      } else if (circleRef.current) {
        bounds = circleRef.current.getBounds();
      } else {
        const delta = radiusKm / 111.32;
        bounds = [
          [latitude - delta, longitude - delta],
          [latitude + delta, longitude + delta],
        ];
      }

      const overlay = L.imageOverlay(activeImage, bounds, {
        opacity: activeLayer === "satellite" ? 0.88 : 0.90,
        interactive: true,
      }).addTo(map);

      overlayRef.current = overlay;
    }
  }, [
    activeLayer,
    satelliteOverlayImage,
    ndviOverlayImage,
    ndmiOverlayImage,
    ndreOverlayImage,
    eviOverlayImage,
    conditionOverlayImage,
    changeOverlayImage,
    sentinel1VvOverlayImage,
    sentinel1VhOverlayImage,
    sentinel1RatioOverlayImage,
    satelliteOverlayBounds,
    latitude,
    longitude,
    radiusKm,
  ]);

  const hasAnyOverlay = Boolean(
    satelliteOverlayImage ||
      ndviOverlayImage ||
      ndmiOverlayImage ||
      ndreOverlayImage ||
      eviOverlayImage ||
      conditionOverlayImage ||
      changeOverlayImage ||
      sentinel1VvOverlayImage ||
      sentinel1VhOverlayImage ||
      sentinel1RatioOverlayImage
  );

  return (
    <div className="relative w-full h-[480px] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Layer Toggle Controls: Optical vs SAR Layers */}
      {hasAnyOverlay && (
        <div className="absolute top-3 left-3 z-[1000] bg-slate-900/90 backdrop-blur-md p-1.5 rounded-xl border border-slate-700/80 shadow-xl flex flex-wrap items-center gap-1 max-w-[92%]">
          {changeOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "change" ? "none" : "change")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "change"
                  ? "bg-gradient-to-r from-amber-500 to-emerald-500 text-slate-950 font-bold shadow-md shadow-amber-500/20"
                  : "text-amber-300 hover:text-white hover:bg-slate-800 border border-amber-500/30"
              }`}
            >
              <span>🔄 Change Map</span>
            </button>
          )}

          {conditionOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "condition" ? "none" : "condition")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "condition"
                  ? "bg-gradient-to-r from-emerald-500 to-lime-500 text-slate-950 font-bold shadow-md shadow-emerald-500/20"
                  : "text-emerald-300 hover:text-white hover:bg-slate-800 border border-emerald-500/30"
              }`}
            >
              <span>🎯 Crop Condition</span>
            </button>
          )}

          {satelliteOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "satellite" ? "none" : "satellite")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "satellite"
                  ? "bg-emerald-500 text-slate-950 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              <span>🛰️ Satellite</span>
            </button>
          )}

          {ndviOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "ndvi" ? "none" : "ndvi")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "ndvi"
                  ? "bg-emerald-500 text-slate-950 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              <span>🌿 NDVI</span>
            </button>
          )}

          {ndmiOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "ndmi" ? "none" : "ndmi")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "ndmi"
                  ? "bg-cyan-500 text-slate-950 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              <span>💧 NDMI</span>
            </button>
          )}

          {ndreOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "ndre" ? "none" : "ndre")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "ndre"
                  ? "bg-lime-500 text-slate-950 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              <span>🌾 NDRE</span>
            </button>
          )}

          {eviOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "evi" ? "none" : "evi")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "evi"
                  ? "bg-teal-500 text-slate-950 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              <span>🍃 EVI</span>
            </button>
          )}

          {/* Sentinel-1 SAR Map Layers */}
          {sentinel1VvOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "sentinel1_vv" ? "none" : "sentinel1_vv")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "sentinel1_vv"
                  ? "bg-sky-400 text-slate-950 font-bold shadow-md shadow-sky-500/20"
                  : "text-sky-300 hover:text-white hover:bg-slate-800 border border-sky-500/30"
              }`}
            >
              <span>📡 S1 VV</span>
            </button>
          )}

          {sentinel1VhOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "sentinel1_vh" ? "none" : "sentinel1_vh")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "sentinel1_vh"
                  ? "bg-violet-400 text-slate-950 font-bold shadow-md shadow-violet-500/20"
                  : "text-violet-300 hover:text-white hover:bg-slate-800 border border-violet-500/30"
              }`}
            >
              <span>📡 S1 VH</span>
            </button>
          )}

          {sentinel1RatioOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "sentinel1_ratio" ? "none" : "sentinel1_ratio")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "sentinel1_ratio"
                  ? "bg-amber-400 text-slate-950 font-bold shadow-md shadow-amber-500/20"
                  : "text-amber-300 hover:text-white hover:bg-slate-800 border border-amber-500/30"
              }`}
            >
              <span>📐 VV/VH</span>
            </button>
          )}

          {qualityMaskOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "quality_mask" ? "none" : "quality_mask")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "quality_mask"
                  ? "bg-purple-500 text-slate-950 font-bold shadow-md shadow-purple-500/20"
                  : "text-purple-300 hover:text-white hover:bg-slate-800 border border-purple-500/30"
              }`}
              title="Multimodal Quality Mask (Valid, Partial, Invalid)"
            >
              <span>🛡️ Quality Mask</span>
            </button>
          )}

          {stressOverlayImage && (
            <button
              onClick={() => handleSelectLayer(activeLayer === "stress" ? "none" : "stress")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                activeLayer === "stress"
                  ? "bg-rose-500 text-white font-bold shadow-md shadow-rose-500/25 ring-1 ring-rose-400"
                  : "text-rose-300 hover:text-white hover:bg-slate-800 border border-rose-500/30"
              }`}
              title="AI Early Stress Detection Probability Map"
            >
              <span>⚡ Stress Map</span>
            </button>
          )}

          <button
            onClick={() => handleSelectLayer("none")}
            className={`px-2.5 py-1.5 rounded-lg text-xs transition ${
              activeLayer === "none"
                ? "bg-slate-700 text-white font-medium"
                : "text-slate-400 hover:text-slate-200"
            }`}
            title="Hide all image overlays"
          >
            <span>Off</span>
          </button>
        </div>
      )}

      {/* Legend Badge */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-slate-900/90 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-700/80 text-xs text-slate-300 flex flex-wrap items-center gap-3 pointer-events-none shadow-lg">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span>
          <span>Center Marker</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-4 h-0 border-t-2 border-dashed border-emerald-400 inline-block"></span>
          <span>AOI ({radiusKm} km)</span>
        </div>
        {activeLayer === "condition" && conditionOverlayImage && (
          <div className="flex items-center gap-1.5 text-emerald-300">
            <span className="w-2.5 h-2.5 rounded bg-emerald-400 inline-block animate-pulse"></span>
            <span>Heuristic Crop Condition Overlay</span>
          </div>
        )}
        {activeLayer === "ndvi" && ndviOverlayImage && (
          <div className="flex items-center gap-1.5 text-emerald-300">
            <span className="w-2.5 h-2.5 rounded bg-emerald-400 inline-block animate-pulse"></span>
            <span>NDVI Spectral Overlay</span>
          </div>
        )}
        {activeLayer === "ndmi" && ndmiOverlayImage && (
          <div className="flex items-center gap-1.5 text-cyan-300">
            <span className="w-2.5 h-2.5 rounded bg-cyan-400 inline-block animate-pulse"></span>
            <span>NDMI Moisture Overlay</span>
          </div>
        )}
        {activeLayer === "ndre" && ndreOverlayImage && (
          <div className="flex items-center gap-1.5 text-lime-300">
            <span className="w-2.5 h-2.5 rounded bg-lime-400 inline-block animate-pulse"></span>
            <span>NDRE Red-Edge Overlay</span>
          </div>
        )}
        {activeLayer === "evi" && eviOverlayImage && (
          <div className="flex items-center gap-1.5 text-teal-300">
            <span className="w-2.5 h-2.5 rounded bg-teal-400 inline-block animate-pulse"></span>
            <span>EVI Enhanced Overlay</span>
          </div>
        )}
        {activeLayer === "satellite" && satelliteOverlayImage && (
          <div className="flex items-center gap-1.5 text-emerald-300">
            <span className="w-2.5 h-2.5 rounded bg-emerald-400 inline-block animate-pulse"></span>
            <span>Sentinel-2 RGB Preview</span>
          </div>
        )}
        {activeLayer === "sentinel1_vv" && sentinel1VvOverlayImage && (
          <div className="flex items-center gap-2 text-sky-300">
            <span className="w-2.5 h-2.5 rounded bg-sky-400 inline-block animate-pulse"></span>
            <span className="font-medium">Sentinel-1 VV Backscatter:</span>
            <div className="w-16 h-2 rounded bg-gradient-to-r from-black via-slate-400 to-white border border-slate-600 inline-block" />
            <span className="text-[10px] text-slate-400">-25 dB (Smooth/Water) → -5 dB (Rough/Dense)</span>
          </div>
        )}
        {activeLayer === "sentinel1_vh" && sentinel1VhOverlayImage && (
          <div className="flex items-center gap-2 text-violet-300">
            <span className="w-2.5 h-2.5 rounded bg-violet-400 inline-block animate-pulse"></span>
            <span className="font-medium">Sentinel-1 VH Backscatter:</span>
            <div className="w-16 h-2 rounded bg-gradient-to-r from-purple-950 via-teal-500 to-yellow-300 border border-slate-600 inline-block" />
            <span className="text-[10px] text-slate-400">-30 dB → -10 dB (Volume / Canopy Density)</span>
          </div>
        )}
        {activeLayer === "sentinel1_ratio" && sentinel1RatioOverlayImage && (
          <div className="flex items-center gap-2 text-amber-300">
            <span className="w-2.5 h-2.5 rounded bg-amber-400 inline-block animate-pulse"></span>
            <span className="font-medium">VV/VH Ratio:</span>
            <div className="w-16 h-2 rounded bg-gradient-to-r from-blue-600 via-emerald-400 to-rose-600 border border-slate-600 inline-block" />
            <span className="text-[10px] text-slate-400">2 dB → 12 dB (Vegetation Structure vs Surface)</span>
          </div>
        )}
        {activeLayer === "quality_mask" && qualityMaskOverlayImage && (
          <div className="flex items-center gap-2 text-purple-300">
            <span className="w-2.5 h-2.5 rounded bg-purple-400 inline-block animate-pulse"></span>
            <span className="font-medium">Multimodal Quality:</span>
            <div className="flex items-center gap-1 text-[10px]">
              <span className="px-1.5 py-0.5 rounded bg-emerald-600 text-white font-semibold">Valid (S1+S2)</span>
              <span className="px-1.5 py-0.5 rounded bg-amber-600 text-white font-semibold">Partial (Opt Only)</span>
              <span className="px-1.5 py-0.5 rounded bg-rose-600 text-white font-semibold">Invalid / Cloud</span>
            </div>
          </div>
        )}
        {activeLayer === "stress" && stressOverlayImage && (
          <div className="flex items-center gap-2 text-rose-300">
            <span className="w-2.5 h-2.5 rounded bg-rose-400 inline-block animate-pulse"></span>
            <span className="font-medium">AI Early Stress Probability:</span>
            <div className="flex items-center gap-1 text-[10px]">
              <span className="px-1.5 py-0.5 rounded bg-emerald-600 text-white font-semibold">Nominal (&lt;0.35)</span>
              <span className="px-1.5 py-0.5 rounded bg-amber-500 text-slate-950 font-bold">Potential (0.35-0.65)</span>
              <span className="px-1.5 py-0.5 rounded bg-rose-600 text-white font-semibold">Elevated (&gt;0.65)</span>
              <span className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">NoData</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
