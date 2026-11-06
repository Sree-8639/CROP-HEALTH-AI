"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export interface GeoPoint {
  lat: number;
  lng: number;
}

export type AOIType = "radius" | "polygon";

export interface PresetAOI {
  id: string;
  name: string;
  module: "crop" | "deforestation" | "heat";
  region: string;
  center: GeoPoint;
  radiusKm: number;
  description: string;
  polygonPoints?: GeoPoint[];
}

export const PRESET_AOIS: PresetAOI[] = [
  {
    id: "tadepalligudem_paddy",
    name: "Tadepalligudem Paddy Basin",
    module: "crop",
    region: "West Godavari, Andhra Pradesh",
    center: { lat: 16.8102, lng: 81.5266 },
    radiusKm: 2.0,
    description: "Intensive irrigated paddy delta with active phenological cycles.",
  },
  {
    id: "krishna_delta_maize",
    name: "Krishna Delta Maize Zone",
    module: "crop",
    region: "Krishna Basin, Andhra Pradesh",
    center: { lat: 16.5062, lng: 80.6480 },
    radiusKm: 2.5,
    description: "Multi-crop commercial farmland with alluvial soil.",
  },
  {
    id: "hasdeo_arand_forest",
    name: "Hasdeo Arand Forest Frontier",
    module: "deforestation",
    region: "Chhattisgarh, India",
    center: { lat: 22.8500, lng: 82.6500 },
    radiusKm: 5.0,
    description: "Dense Sal canopy under fragmentation threat from mining corridors.",
  },
  {
    id: "western_ghats_canopy",
    name: "Western Ghats Wet Evergreen",
    module: "deforestation",
    region: "Karnataka, India",
    center: { lat: 12.9000, lng: 75.7000 },
    radiusKm: 4.0,
    description: "High-biodiversity tropical rainforest canopy out-of-region test.",
  },
  {
    id: "visakhapatnam_urban",
    name: "Visakhapatnam Metropolitan",
    module: "heat",
    region: "Andhra Pradesh, India",
    center: { lat: 17.7200, lng: 83.2500 },
    radiusKm: 5.0,
    description: "High-density coastal urban area with significant thermal contrast.",
  },
  {
    id: "vijayawada_corridor",
    name: "Vijayawada City Core",
    module: "heat",
    region: "Andhra Pradesh, India",
    center: { lat: 16.5100, lng: 80.6400 },
    radiusKm: 3.5,
    description: "Inland urban core with high summer surface temperatures.",
  },
];

export interface SharedAOIContextType {
  aoiType: AOIType;
  setAoiType: (type: AOIType) => void;
  center: GeoPoint;
  setCenter: (point: GeoPoint) => void;
  radiusKm: number;
  setRadiusKm: (radius: number) => void;
  polygonPoints: GeoPoint[];
  setPolygonPoints: (points: GeoPoint[]) => void;
  addPolygonPoint: (point: GeoPoint) => void;
  clearPolygon: () => void;
  startDate: string;
  setStartDate: (d: string) => void;
  endDate: string;
  setEndDate: (d: string) => void;
  setDatePreset: (preset: "30d" | "90d" | "monsoon" | "pre-monsoon") => void;
  selectedPresetId: string | null;
  applyPreset: (presetId: string) => void;
  getGeoJsonGeometry: () => any;
  getBbox: () => [number, number, number, number];
}

const SharedAOIContext = createContext<SharedAOIContextType | undefined>(undefined);

export function SharedAOIProvider({ children }: { children: React.ReactNode }) {
  const [aoiType, setAoiType] = useState<AOIType>("radius");
  const [center, setCenter] = useState<GeoPoint>({ lat: 16.8102, lng: 81.5266 });
  const [radiusKm, setRadiusKm] = useState<number>(2.0);
  const [polygonPoints, setPolygonPoints] = useState<GeoPoint[]>([
    { lat: 16.8202, lng: 81.5166 },
    { lat: 16.8202, lng: 81.5366 },
    { lat: 16.8002, lng: 81.5366 },
    { lat: 16.8002, lng: 81.5166 },
  ]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>("tadepalligudem_paddy");

  // Shared date range
  const [startDate, setStartDate] = useState<string>("2024-01-01");
  const [endDate, setEndDate] = useState<string>("2024-01-31");

  const addPolygonPoint = (pt: GeoPoint) => {
    setPolygonPoints((prev) => [...prev, pt]);
  };

  const clearPolygon = () => {
    setPolygonPoints([]);
  };

  const setDatePreset = (preset: "30d" | "90d" | "monsoon" | "pre-monsoon") => {
    const now = new Date();
    if (preset === "30d") {
      setEndDate(now.toISOString().split("T")[0]);
      setStartDate(new Date(now.getTime() - 30 * 86400000).toISOString().split("T")[0]);
    } else if (preset === "90d") {
      setEndDate(now.toISOString().split("T")[0]);
      setStartDate(new Date(now.getTime() - 90 * 86400000).toISOString().split("T")[0]);
    } else if (preset === "monsoon") {
      setStartDate("2024-07-01");
      setEndDate("2024-09-30");
    } else if (preset === "pre-monsoon") {
      setStartDate("2024-04-01");
      setEndDate("2024-05-31");
    }
  };

  const applyPreset = (presetId: string) => {
    const preset = PRESET_AOIS.find((p) => p.id === presetId);
    if (!preset) return;
    setSelectedPresetId(preset.id);
    setCenter(preset.center);
    setRadiusKm(preset.radiusKm);
    if (preset.polygonPoints && preset.polygonPoints.length >= 3) {
      setPolygonPoints(preset.polygonPoints);
    } else {
      // Generate default bounding box points
      const dLat = preset.radiusKm / 111.0;
      const dLng = preset.radiusKm / (111.0 * Math.cos((preset.center.lat * Math.PI) / 180));
      setPolygonPoints([
        { lat: preset.center.lat + dLat, lng: preset.center.lng - dLng },
        { lat: preset.center.lat + dLat, lng: preset.center.lng + dLng },
        { lat: preset.center.lat - dLat, lng: preset.center.lng + dLng },
        { lat: preset.center.lat - dLat, lng: preset.center.lng - dLng },
      ]);
    }
  };

  const getBbox = (): [number, number, number, number] => {
    if (aoiType === "polygon" && polygonPoints.length >= 3) {
      const lats = polygonPoints.map((p) => p.lat);
      const lngs = polygonPoints.map((p) => p.lng);
      return [Math.min(...lngs), Math.min(...lats), Math.max(...lngs), Math.max(...lats)];
    }
    const dLat = radiusKm / 111.0;
    const dLng = radiusKm / (111.0 * Math.cos((center.lat * Math.PI) / 180));
    return [
      center.lng - dLng,
      center.lat - dLat,
      center.lng + dLng,
      center.lat + dLat,
    ];
  };

  const getGeoJsonGeometry = () => {
    if (aoiType === "polygon" && polygonPoints.length >= 3) {
      const coords = polygonPoints.map((p) => [p.lng, p.lat]);
      // Ensure closed polygon
      if (
        coords[0][0] !== coords[coords.length - 1][0] ||
        coords[0][1] !== coords[coords.length - 1][1]
      ) {
        coords.push([coords[0][0], coords[0][1]]);
      }
      return {
        type: "Polygon",
        coordinates: [coords],
      };
    }

    // Circular approximation with 32 vertices
    const steps = 32;
    const coords: [number, number][] = [];
    const dLat = radiusKm / 111.0;
    const dLng = radiusKm / (111.0 * Math.cos((center.lat * Math.PI) / 180));

    for (let i = 0; i < steps; i++) {
      const theta = (i / steps) * 2 * Math.PI;
      const x = center.lng + dLng * Math.cos(theta);
      const y = center.lat + dLat * Math.sin(theta);
      coords.push([Number(x.toFixed(6)), Number(y.toFixed(6))]);
    }
    coords.push(coords[0]);

    return {
      type: "Polygon",
      coordinates: [coords],
    };
  };

  return (
    <SharedAOIContext.Provider
      value={{
        aoiType,
        setAoiType,
        center,
        setCenter,
        radiusKm,
        setRadiusKm,
        polygonPoints,
        setPolygonPoints,
        addPolygonPoint,
        clearPolygon,
        startDate,
        setStartDate,
        endDate,
        setEndDate,
        setDatePreset,
        selectedPresetId,
        applyPreset,
        getGeoJsonGeometry,
        getBbox,
      }}
    >
      {children}
    </SharedAOIContext.Provider>
  );
}

export function useSharedAOI() {
  const ctx = useContext(SharedAOIContext);
  if (!ctx) {
    throw new Error("useSharedAOI must be used within a SharedAOIProvider");
  }
  return ctx;
}

