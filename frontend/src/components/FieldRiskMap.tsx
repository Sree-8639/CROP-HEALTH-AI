"use client";

import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { PixelInspectionData } from "@/components/PixelInspectionPanel";
import { API_BASE_URL, API_ROUTES } from "@/config/api";

export interface SpatialLayerMeta {
  layer_key: string;
  layer_name: string;
  layer_type: string;
  description: string;
  bounds: number[]; // [min_lon, min_lat, max_lon, max_lat]
  image_png_base64?: string | null;
  legend?: Record<string, any>;
}

interface FieldRiskMapProps {
  analysisId: string;
  layers: Record<string, SpatialLayerMeta>;
  selectedLayer: string;
  geometry?: any;
  latitude?: number;
  longitude?: number;
  radiusKm?: number;
  isDemo?: boolean;
  observationDate?: string;
  onSelectLayer?: (layerKey: string) => void;
  onPixelInspect?: (result: PixelInspectionData | null) => void;
  onInspectionLoading?: (loading: boolean) => void;
}

export default function FieldRiskMap({
  analysisId,
  layers,
  selectedLayer,
  geometry,
  latitude = 16.805,
  longitude = 81.525,
  radiusKm = 1.5,
  isDemo = false,
  observationDate,
  onSelectLayer,
  onPixelInspect,
  onInspectionLoading,
}: FieldRiskMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const polygonLayerRef = useRef<L.Layer | null>(null);
  const overlayRef = useRef<L.ImageOverlay | null>(null);
  const crosshairMarkerRef = useRef<L.Marker | null>(null);

  const [opacity, setOpacity] = useState<number>(0.85);
  const [isExporting, setIsExporting] = useState<string | null>(null);
  const [clickedCoords, setClickedCoords] = useState<{ lat: number; lon: number } | null>(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const satelliteLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri, USDA, USGS, AeroGRID",
        maxZoom: 19,
      }
    );

    const streetLayer = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
      }
    );

    const map = L.map(mapContainerRef.current, {
      center: [latitude, longitude],
      zoom: 15,
      layers: [satelliteLayer],
      zoomControl: true,
    });

    const baseMaps = {
      "🛰️ Satellite": satelliteLayer,
      "🗺️ Streets": streetLayer,
    };
    L.control.layers(baseMaps, undefined, { position: "topright" }).addTo(map);

    // Click handler for pixel inspection
    map.on("click", async (e: L.LeafletMouseEvent) => {
      const lat = e.latlng.lat;
      const lon = e.latlng.lng;
      setClickedCoords({ lat, lon });

      // Move crosshair pin
      const crosshairIcon = L.divIcon({
        className: "custom-crosshair-pin",
        html: `
          <div style="position: relative; width: 24px; height: 24px; transform: translate(-50%, -50%);">
            <div style="position: absolute; inset: 0; border: 2px solid #38bdf8; border-radius: 50%; box-shadow: 0 0 10px #38bdf8; animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
            <div style="position: absolute; inset: 4px; background: #38bdf8; border-radius: 50%; border: 2px solid #ffffff;"></div>
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      if (crosshairMarkerRef.current) {
        crosshairMarkerRef.current.setLatLng([lat, lon]);
      } else {
        const marker = L.marker([lat, lon], { icon: crosshairIcon }).addTo(map);
        crosshairMarkerRef.current = marker;
      }

      // Query Backend Pixel Inspection Endpoint
      if (onInspectionLoading) onInspectionLoading(true);

      try {
        const res = await fetch(API_ROUTES.FIELD_RISK_PIXEL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            analysis_id: analysisId,
            latitude: lat,
            longitude: lon,
          }),
        });

        if (res.ok) {
          const data: PixelInspectionData = await res.json();
          if (onPixelInspect) onPixelInspect(data);
        } else {
          console.error("Pixel inspection failed with status:", res.status);
          if (onPixelInspect) onPixelInspect(null);
        }
      } catch (err) {
        console.error("Error inspecting pixel:", err);
        if (onPixelInspect) onPixelInspect(null);
      } finally {
        if (onInspectionLoading) onInspectionLoading(false);
      }
    });

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      polygonLayerRef.current = null;
      overlayRef.current = null;
      crosshairMarkerRef.current = null;
    };
  }, []);

  // Update Field Boundary (Polygon / Circle)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (polygonLayerRef.current) {
      map.removeLayer(polygonLayerRef.current);
      polygonLayerRef.current = null;
    }

    if (geometry) {
      const geoJsonLayer = L.geoJSON(geometry, {
        style: {
          color: "#10b981",
          weight: 2.5,
          dashArray: "5, 7",
          fillColor: "#10b981",
          fillOpacity: 0.06,
        },
      }).addTo(map);

      polygonLayerRef.current = geoJsonLayer;
      try {
        const b = geoJsonLayer.getBounds();
        if (b.isValid()) {
          map.fitBounds(b, { padding: [35, 35], maxZoom: 16 });
        }
      } catch {
        // Fallback
      }
    } else if (latitude && longitude && radiusKm) {
      const circle = L.circle([latitude, longitude], {
        radius: radiusKm * 1000,
        color: "#10b981",
        weight: 2.5,
        dashArray: "5, 7",
        fillColor: "#10b981",
        fillOpacity: 0.06,
      }).addTo(map);

      polygonLayerRef.current = circle;
      map.fitBounds(circle.getBounds(), { padding: [35, 35], maxZoom: 16 });
    }
  }, [geometry, latitude, longitude, radiusKm]);

  // Update Raster Overlay on Layer Change or Opacity Change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }

    const currentLayer = layers[selectedLayer];
    if (!currentLayer || !currentLayer.image_png_base64 || !currentLayer.bounds) {
      return;
    }

    // bounds: [min_lon, min_lat, max_lon, max_lat]
    const b = currentLayer.bounds;
    const leafletBounds: L.LatLngBoundsExpression = [
      [b[1], b[0]], // [min_lat, min_lon]
      [b[3], b[2]], // [max_lat, max_lon]
    ];

    const src = currentLayer.image_png_base64.startsWith("data:")
      ? currentLayer.image_png_base64
      : `data:image/png;base64,${currentLayer.image_png_base64}`;

    const overlay = L.imageOverlay(src, leafletBounds, {
      opacity: opacity,
      interactive: true,
    }).addTo(map);

    overlayRef.current = overlay;
  }, [layers, selectedLayer, opacity]);

  // Handle Export Downloads
  const handleExport = async (fmt: "geotiff" | "csv" | "json" | "png") => {
    setIsExporting(fmt);
    try {
      const url = `${API_BASE_URL}/api/v1/analysis/field-risk-map/${analysisId}/export?format=${fmt}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Export failed with HTTP ${res.status}`);

      const blob = await res.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = downloadUrl;

      // Extract filename from header or build fallback
      const disposition = res.headers.get("content-disposition");
      let filename = `field_risk_${selectedLayer}_${analysisId.slice(0, 8)}.${fmt === "geotiff" ? "tif" : fmt}`;
      if (disposition && disposition.includes("filename=")) {
        filename = disposition.split("filename=")[1].replace(/"/g, "").trim();
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error(`Export ${fmt} failed:`, err);
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsExporting(null);
    }
  };

  const currentLayerMeta = layers[selectedLayer];

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 backdrop-blur-md shadow-2xl space-y-4">
      {/* Top Map Bar: Active Layer Title & Export Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-400 font-bold text-sm tracking-wide">
              {currentLayerMeta?.layer_name || "Field Risk Layer"}
            </span>
            {isDemo && (
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                DEMO DATA
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5 line-clamp-1">
            {currentLayerMeta?.description || "Spatial canopy condition and risk intelligence raster overlay"}
          </p>
        </div>

        {/* Action Controls: Opacity Slider & Exports */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Opacity Slider */}
          <div className="flex items-center gap-1.5 bg-slate-950/70 border border-slate-800 px-2.5 py-1 rounded-xl text-xs text-slate-300">
            <span className="text-[11px] text-slate-400">Opacity:</span>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-16 h-1.5 accent-emerald-500 cursor-pointer"
              title="Adjust layer transparency"
            />
            <span className="font-mono text-[10px] w-6 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>

          {/* Export Dropdown / Buttons */}
          <div className="flex items-center gap-1 bg-slate-950/70 border border-slate-800 p-1 rounded-xl">
            <button
              onClick={() => handleExport("geotiff")}
              disabled={isExporting !== null}
              title="Download 7-Band GeoTIFF (EPSG projected UTM 10m stack)"
              type="button"
              className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 transition disabled:opacity-50"
            >
              {isExporting === "geotiff" ? "Exporting..." : "💾 GeoTIFF"}
            </button>
            <button
              onClick={() => handleExport("csv")}
              disabled={isExporting !== null}
              title="Download field summary statistics CSV"
              type="button"
              className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition disabled:opacity-50"
            >
              {isExporting === "csv" ? "..." : "📄 CSV"}
            </button>
            <button
              onClick={() => handleExport("json")}
              disabled={isExporting !== null}
              title="Download comprehensive metadata JSON"
              type="button"
              className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition disabled:opacity-50"
            >
              {isExporting === "json" ? "..." : "{ } JSON"}
            </button>
            <button
              onClick={() => handleExport("png")}
              disabled={isExporting !== null}
              title="Download risk map preview PNG"
              type="button"
              className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition disabled:opacity-50"
            >
              {isExporting === "png" ? "..." : "🖼️ PNG"}
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Map Container */}
      <div className="relative w-full h-[520px] rounded-xl overflow-hidden border border-slate-800 bg-slate-950 shadow-inner">
        <div ref={mapContainerRef} className="w-full h-full z-0" />

        {/* Floating Instruction Badge */}
        <div className="absolute bottom-3 left-3 z-[1000] bg-slate-950/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 text-[11px] text-slate-300 flex items-center gap-2 shadow-lg">
          <span className="text-emerald-400">📍</span>
          <span>Click any map pixel to inspect radiometric values &amp; evidence</span>
          {clickedCoords && (
            <span className="font-mono text-emerald-300 ml-1">
              ({clickedCoords.lat.toFixed(4)}, {clickedCoords.lon.toFixed(4)})
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
