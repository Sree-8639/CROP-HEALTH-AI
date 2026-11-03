"use client";

import { useState, useMemo } from "react";

export type IndexType = "ndvi" | "ndmi" | "ndre" | "evi";
export type SarFeatureType = "vv" | "vh" | "vv_vh_ratio";
export type ChartMode = "optical" | "sar";

export interface ObservationDataPoint {
  observation_id: string;
  date: string;
  datetime?: string;
  cloud_coverage: number;
  valid_pixel_percentage: number;
  quality: string;
  satellite?: string;
  satellite_availability?: {
    sentinel2: boolean;
    sentinel1: boolean;
  };
  ndvi: { mean: number | null; median?: number | null; min?: number | null; max?: number | null };
  ndmi: { mean: number | null; median?: number | null; min?: number | null; max?: number | null };
  ndre: { mean: number | null; median?: number | null; min?: number | null; max?: number | null };
  evi: { mean: number | null; median?: number | null; min?: number | null; max?: number | null };
  sar?: {
    observation_id?: string | null;
    date?: string | null;
    vv?: { mean: number | null; median?: number | null; min?: number | null; max?: number | null } | null;
    vh?: { mean: number | null; median?: number | null; min?: number | null; max?: number | null } | null;
    vv_vh_ratio?: { mean: number | null; median?: number | null } | null;
    status?: string;
    valid_pixel_percentage?: number;
    product_type?: string;
  } | null;
}

interface TimeSeriesChartProps {
  observations: ObservationDataPoint[];
  selectedIndex?: IndexType;
  multiIndexMode?: boolean;
  selectedIndices?: IndexType[];
  onSelectObservation?: (obs: ObservationDataPoint) => void;
  selectedObsAId?: string | null;
  selectedObsBId?: string | null;
  initialMode?: ChartMode;
}

const OPTICAL_CONFIG: Record<IndexType, { label: string; color: string; unit: string }> = {
  ndvi: { label: "NDVI (Greenness)", color: "#10b981", unit: "" },
  ndmi: { label: "NDMI (Moisture)", color: "#06b6d4", unit: "" },
  ndre: { label: "NDRE (Chlorophyll)", color: "#f59e0b", unit: "" },
  evi: { label: "EVI (Enhanced)", color: "#a855f7", unit: "" },
};

const SAR_CONFIG: Record<SarFeatureType, { label: string; color: string; unit: string }> = {
  vv: { label: "VV Backscatter", color: "#38bdf8", unit: "dB" },
  vh: { label: "VH Backscatter", color: "#c084fc", unit: "dB" },
  vv_vh_ratio: { label: "VV/VH Cross-Ratio", color: "#fb923c", unit: "dB" },
};

export default function TimeSeriesChart({
  observations,
  selectedIndex = "ndvi",
  multiIndexMode = false,
  selectedIndices = ["ndvi"],
  onSelectObservation,
  selectedObsAId,
  selectedObsBId,
  initialMode = "optical",
}: TimeSeriesChartProps) {
  const [chartMode, setChartMode] = useState<ChartMode>(initialMode);
  const [selectedSarFeature, setSelectedSarFeature] = useState<SarFeatureType>("vv");
  const [sarMultiMode, setSarMultiMode] = useState(false);
  const [selectedSarFeatures, setSelectedSarFeatures] = useState<SarFeatureType[]>(["vv", "vh"]);

  const [hoveredPoint, setHoveredPoint] = useState<{
    obs: ObservationDataPoint;
    idx: number;
    x: number;
    y: number;
    value: number | null;
    seriesKey: string;
    seriesLabel: string;
    color: string;
    unit: string;
  } | null>(null);

  // SVG chart dimensions
  const width = 800;
  const height = 340;
  const padding = { top: 30, right: 30, bottom: 50, left: 65 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Active series keys depending on mode
  const activeSeries = useMemo(() => {
    if (chartMode === "optical") {
      return multiIndexMode ? selectedIndices : [selectedIndex];
    } else {
      return sarMultiMode ? selectedSarFeatures : [selectedSarFeature];
    }
  }, [chartMode, multiIndexMode, selectedIndices, selectedIndex, sarMultiMode, selectedSarFeatures, selectedSarFeature]);

  // Helper to extract value from an observation
  const getObsValue = (obs: ObservationDataPoint, key: string): number | null => {
    if (chartMode === "optical") {
      const optKey = key as IndexType;
      return obs[optKey]?.mean ?? null;
    } else {
      if (!obs.sar) return null;
      if (key === "vv") return obs.sar.vv?.mean ?? null;
      if (key === "vh") return obs.sar.vh?.mean ?? null;
      if (key === "vv_vh_ratio") return obs.sar.vv_vh_ratio?.mean ?? null;
      return null;
    }
  };

  // Determine Y-axis min/max
  const { yMin, yMax } = useMemo(() => {
    const allValues: number[] = [];

    observations.forEach((obs) => {
      activeSeries.forEach((key) => {
        const val = getObsValue(obs, key);
        if (val !== null && val !== undefined && !isNaN(val)) {
          allValues.push(val);
        }
      });
    });

    if (chartMode === "optical") {
      let minVal = 0.0;
      let maxVal = 1.0;
      if (allValues.length > 0) {
        const dataMin = Math.min(...allValues);
        const dataMax = Math.max(...allValues);
        minVal = Math.max(-0.5, Math.floor((dataMin - 0.05) * 10) / 10);
        maxVal = Math.min(1.0, Math.ceil((dataMax + 0.05) * 10) / 10);
        if (maxVal <= minVal) maxVal = minVal + 0.2;
      }
      return { yMin: minVal, yMax: maxVal };
    } else {
      // SAR decibel scale (typically -25 to +15 dB)
      let minVal = -25.0;
      let maxVal = 5.0;
      if (allValues.length > 0) {
        const dataMin = Math.min(...allValues);
        const dataMax = Math.max(...allValues);
        minVal = Math.floor(dataMin - 2);
        maxVal = Math.ceil(dataMax + 2);
        if (maxVal <= minVal) maxVal = minVal + 5;
      }
      return { yMin: minVal, yMax: maxVal };
    }
  }, [observations, chartMode, activeSeries]);

  // Coordinate scales
  const getX = (index: number) => {
    if (observations.length <= 1) return padding.left + plotWidth / 2;
    return padding.left + (index / (observations.length - 1)) * plotWidth;
  };

  const getY = (value: number | null) => {
    if (value === null || isNaN(value)) return padding.top + plotHeight;
    const clamped = Math.max(yMin, Math.min(yMax, value));
    return padding.top + plotHeight - ((clamped - yMin) / (yMax - yMin)) * plotHeight;
  };

  // Grid lines (5 horizontal lines)
  const yTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = yMin + (i / (count - 1)) * (yMax - yMin);
      ticks.push(roundToTwo(val));
    }
    return ticks;
  }, [yMin, yMax]);

  return (
    <div className="relative w-full overflow-hidden bg-slate-900/80 rounded-2xl border border-slate-800 p-4 sm:p-6 shadow-xl backdrop-blur-md">
      {/* Top Header: Title, Satellite Mode Switcher, and Legends */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-5 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
              <span>{chartMode === "optical" ? "📈 Optical Spectral Trajectory" : "📡 Sentinel-1 SAR Backscatter Time-Series"}</span>
            </h3>
            <span className="text-xs font-normal text-slate-400">
              ({observations.length} {observations.length === 1 ? "observation" : "observations"})
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {chartMode === "optical"
              ? "Sentinel-2 Multi-spectral vegetation & moisture indices (0.0 to 1.0 unitless scale)."
              : "Sentinel-1 C-band synthetic aperture radar backscatter in decibels (dB). Independent scale."}
          </p>
        </div>

        {/* Mode Selector Toggle: [ 🌿 Optical Indices ] [ 📡 SAR Features ] */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setChartMode("optical")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              chartMode === "optical"
                ? "bg-emerald-500 text-slate-950 shadow-md font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <span>🌿 Optical Indices</span>
          </button>
          <button
            onClick={() => setChartMode("sar")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              chartMode === "sar"
                ? "bg-gradient-to-r from-sky-400 to-indigo-500 text-slate-950 shadow-md font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <span>📡 SAR Backscatter</span>
          </button>
        </div>
      </div>

      {/* Sub-toolbar for SAR Mode Feature Selection */}
      {chartMode === "sar" && (
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-medium">SAR Feature:</span>
            {(["vv", "vh", "vv_vh_ratio"] as SarFeatureType[]).map((fKey) => {
              const conf = SAR_CONFIG[fKey];
              const isSelected = sarMultiMode ? selectedSarFeatures.includes(fKey) : selectedSarFeature === fKey;
              return (
                <button
                  key={fKey}
                  onClick={() => {
                    if (sarMultiMode) {
                      if (selectedSarFeatures.includes(fKey)) {
                        if (selectedSarFeatures.length > 1) {
                          setSelectedSarFeatures(selectedSarFeatures.filter((x) => x !== fKey));
                        }
                      } else {
                        setSelectedSarFeatures([...selectedSarFeatures, fKey]);
                      }
                    } else {
                      setSelectedSarFeature(fKey);
                    }
                  }}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition flex items-center gap-1.5 ${
                    isSelected
                      ? "text-slate-950 font-bold shadow-sm"
                      : "text-slate-400 hover:text-slate-200 bg-slate-900 border border-slate-800"
                  }`}
                  style={{
                    backgroundColor: isSelected ? conf.color : undefined,
                  }}
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: isSelected ? "#0f172a" : conf.color }} />
                  <span>{conf.label}</span>
                </button>
              );
            })}
          </div>

          <button
            onClick={() => setSarMultiMode(!sarMultiMode)}
            className={`px-2.5 py-1 rounded-lg text-xs border transition ${
              sarMultiMode
                ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                : "text-slate-400 hover:text-white border-slate-700 bg-slate-900"
            }`}
          >
            {sarMultiMode ? "Multi-Curve Active" : "Compare SAR Curves"}
          </button>
        </div>
      )}

      {/* Legend display for active series */}
      <div className="flex flex-wrap items-center gap-3 text-xs mb-3">
        {activeSeries.map((sKey) => {
          const conf = chartMode === "optical" ? OPTICAL_CONFIG[sKey as IndexType] : SAR_CONFIG[sKey as SarFeatureType];
          if (!conf) return null;
          return (
            <div key={sKey} className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: conf.color }} />
              <span className="text-slate-200 font-medium">{conf.label}</span>
              {conf.unit && <span className="text-[10px] text-slate-400">({conf.unit})</span>}
            </div>
          );
        })}
      </div>

      {/* SVG Viewport */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[580px] select-none"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <defs>
            {activeSeries.map((sKey) => {
              const conf = chartMode === "optical" ? OPTICAL_CONFIG[sKey as IndexType] : SAR_CONFIG[sKey as SarFeatureType];
              if (!conf) return null;
              return (
                <linearGradient key={`grad-${sKey}`} id={`gradient-${sKey}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={conf.color} stopOpacity="0.25" />
                  <stop offset="100%" stopColor={conf.color} stopOpacity="0.0" />
                </linearGradient>
              );
            })}
          </defs>

          {/* Horizontal Grid lines */}
          {yTicks.map((tickVal) => {
            const y = getY(tickVal);
            return (
              <g key={`ytick-${tickVal}`}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="#334155"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
                <text
                  x={padding.left - 10}
                  y={y + 4}
                  fill="#94a3b8"
                  fontSize="11"
                  textAnchor="end"
                  fontFamily="sans-serif"
                >
                  {chartMode === "sar" ? `${tickVal.toFixed(1)} dB` : tickVal.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Vertical date lines and X-axis ticks */}
          {observations.map((obs, i) => {
            const x = getX(i);
            const isObsA = obs.observation_id === selectedObsAId;
            const isObsB = obs.observation_id === selectedObsBId;

            return (
              <g key={`xtick-${obs.observation_id}`}>
                {(isObsA || isObsB) && (
                  <line
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={padding.top + plotHeight}
                    stroke={isObsA ? "#38bdf8" : "#f59e0b"}
                    strokeWidth="2"
                    strokeDasharray="2 2"
                  />
                )}
                <text
                  x={x}
                  y={padding.top + plotHeight + 22}
                  fill="#94a3b8"
                  fontSize="11"
                  textAnchor="middle"
                  fontFamily="sans-serif"
                >
                  {formatDateLabel(obs.date)}
                </text>
              </g>
            );
          })}

          {/* Plot Lines & Area Fills for each active series */}
          {activeSeries.map((sKey) => {
            const conf = chartMode === "optical" ? OPTICAL_CONFIG[sKey as IndexType] : SAR_CONFIG[sKey as SarFeatureType];
            if (!conf) return null;

            const points = observations.map((obs, i) => ({
              x: getX(i),
              y: getY(getObsValue(obs, sKey)),
              val: getObsValue(obs, sKey),
              obs,
              i,
            }));

            const validPoints = points.filter((p) => p.val !== null);
            if (validPoints.length === 0) return null;

            const pathD = validPoints.reduce((acc, p, index) => {
              return `${acc} ${index === 0 ? "M" : "L"} ${p.x} ${p.y}`;
            }, "");

            const firstP = validPoints[0];
            const lastP = validPoints[validPoints.length - 1];
            const areaD = `${pathD} L ${lastP.x} ${padding.top + plotHeight} L ${firstP.x} ${padding.top + plotHeight} Z`;

            return (
              <g key={`series-${sKey}`}>
                {/* Shaded Area fill */}
                <path d={areaD} fill={`url(#gradient-${sKey})`} />

                {/* Main Curve Line */}
                <path
                  d={pathD}
                  fill="none"
                  stroke={conf.color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Data Points */}
                {validPoints.map((p) => {
                  const isHovered = hoveredPoint?.obs.observation_id === p.obs.observation_id && hoveredPoint?.seriesKey === sKey;
                  const isObsA = p.obs.observation_id === selectedObsAId;
                  const isObsB = p.obs.observation_id === selectedObsBId;

                  return (
                    <g
                      key={`point-${sKey}-${p.obs.observation_id}`}
                      className="cursor-pointer transition-transform"
                      onClick={() => onSelectObservation && onSelectObservation(p.obs)}
                      onMouseEnter={() =>
                        setHoveredPoint({
                          obs: p.obs,
                          idx: p.i,
                          x: p.x,
                          y: p.y,
                          value: p.val,
                          seriesKey: sKey,
                          seriesLabel: conf.label,
                          color: conf.color,
                          unit: conf.unit,
                        })
                      }
                    >
                      {(isObsA || isObsB) && (
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r="10"
                          fill="none"
                          stroke={isObsA ? "#38bdf8" : "#f59e0b"}
                          strokeWidth="2.5"
                          className="animate-pulse"
                        />
                      )}

                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={isHovered ? 6.5 : isObsA || isObsB ? 5.5 : 4}
                        fill={isHovered ? "#ffffff" : conf.color}
                        stroke="#0f172a"
                        strokeWidth="2"
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Floating Hover Tooltip */}
      {hoveredPoint && (
        <div
          className="absolute z-20 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 px-3.5 py-2.5 rounded-xl bg-slate-950/95 border border-slate-700/80 shadow-2xl text-xs backdrop-blur-md transition-all"
          style={{
            left: `${(hoveredPoint.x / width) * 100}%`,
            top: `${(hoveredPoint.y / height) * 100}%`,
          }}
        >
          <div className="flex items-center justify-between gap-3 border-b border-slate-800 pb-1.5 mb-1.5">
            <span className="font-semibold text-white">📅 {hoveredPoint.obs.date}</span>
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                hoveredPoint.obs.quality === "Excellent"
                  ? "bg-emerald-500/20 text-emerald-300"
                  : hoveredPoint.obs.quality === "Good"
                  ? "bg-teal-500/20 text-teal-300"
                  : hoveredPoint.obs.quality === "Fair"
                  ? "bg-amber-500/20 text-amber-300"
                  : "bg-rose-500/20 text-rose-300"
              }`}
            >
              {hoveredPoint.obs.quality}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between gap-4">
              <span className="text-slate-400 capitalize">{hoveredPoint.seriesLabel}:</span>
              <span className="font-mono font-bold" style={{ color: hoveredPoint.color }}>
                {hoveredPoint.value !== null
                  ? `${hoveredPoint.value.toFixed(3)}${hoveredPoint.unit ? ` ${hoveredPoint.unit}` : ""}`
                  : "N/A"}
              </span>
            </div>

            {chartMode === "sar" && hoveredPoint.obs.sar && (
              <div className="flex items-center justify-between gap-4">
                <span className="text-slate-400">SAR Date:</span>
                <span className="font-mono text-slate-200">{hoveredPoint.obs.sar.date || "Matched window"}</span>
              </div>
            )}

            <div className="flex items-center justify-between gap-4">
              <span className="text-slate-400">Cloud Coverage:</span>
              <span className="font-mono text-slate-200">{hoveredPoint.obs.cloud_coverage.toFixed(1)}%</span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-slate-400">Valid Pixels:</span>
              <span className="font-mono text-slate-200">{hoveredPoint.obs.valid_pixel_percentage.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function roundToTwo(num: number): number {
  return Math.round(num * 100) / 100;
}

function formatDateLabel(dateStr: string): string {
  try {
    const parts = dateStr.split("-");
    if (parts.length === 3) {
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const monthIdx = parseInt(parts[1], 10) - 1;
      return `${monthNames[monthIdx]} ${parseInt(parts[2], 10)}`;
    }
  } catch {}
  return dateStr;
}
