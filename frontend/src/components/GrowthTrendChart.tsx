"use client";

import React, { useState, useMemo } from "react";

export interface HistoricalObservationRecord {
  date: string;
  indices: Record<string, number | null>;
  cloud_coverage: number;
  valid_pixel_percentage: number;
  nodata_pct: number;
  quality_status: string;
  sequential_features?: Record<
    string,
    {
      previous_val: number | null;
      abs_change: number | null;
      rate_of_change_per_day: number | null;
      rolling_mean_3: number | null;
      rolling_std_3: number | null;
      interval_days: number | null;
    }
  >;
}

export interface PredictionIntervalItem {
  lower: number;
  upper: number;
  confidence_level: number;
}

export interface GrowthTrendChartProps {
  historicalObservations: HistoricalObservationRecord[];
  forecastDates: string[];
  forecastValues: Record<string, number[]>;
  predictionIntervals?: Record<string, PredictionIntervalItem[]>;
  selectedIndex: string; // "NDVI" | "NDMI" | "NDRE" | "EVI" | "VV" | "VH"
  onSelectIndex?: (index: string) => void;
  isDemo?: boolean;
}

const INDEX_CONFIG: Record<
  string,
  {
    label: string;
    description: string;
    color: string;
    fillColor: string;
    unit: string;
    minY: number;
    maxY: number;
  }
> = {
  NDVI: {
    label: "NDVI (Normalized Difference Veg. Index)",
    description: "Canopy greenness, biomass volume, and chlorophyll absorption density",
    color: "#10b981", // emerald-500
    fillColor: "rgba(16, 185, 129, 0.15)",
    unit: "",
    minY: 0.0,
    maxY: 1.0,
  },
  NDMI: {
    label: "NDMI (Normalized Difference Moisture Index)",
    description: "Canopy water content and internal leaf cellular hydration",
    color: "#06b6d4", // cyan-500
    fillColor: "rgba(6, 182, 212, 0.15)",
    unit: "",
    minY: -0.2,
    maxY: 0.8,
  },
  NDRE: {
    label: "NDRE (Normalized Difference Red Edge)",
    description: "Red-edge sensitivity for dense, saturated canopies and nitrogen status",
    color: "#f59e0b", // amber-500
    fillColor: "rgba(245, 158, 11, 0.15)",
    unit: "",
    minY: 0.0,
    maxY: 0.8,
  },
  EVI: {
    label: "EVI (Enhanced Vegetation Index)",
    description: "Atmospherically-corrected canopy structural vigor across dense foliage",
    color: "#a855f7", // purple-500
    fillColor: "rgba(168, 85, 247, 0.15)",
    unit: "",
    minY: 0.0,
    maxY: 1.0,
  },
  VV: {
    label: "VV SAR Backscatter",
    description: "Vertical-vertical radar return sensitive to vertical stems and surface moisture",
    color: "#38bdf8", // sky-400
    fillColor: "rgba(56, 189, 248, 0.15)",
    unit: "dB",
    minY: -22.0,
    maxY: -5.0,
  },
  VH: {
    label: "VH SAR Backscatter",
    description: "Cross-polarized radar return sensitive to volume scattering within canopy",
    color: "#c084fc", // violet-400
    fillColor: "rgba(192, 132, 252, 0.15)",
    unit: "dB",
    minY: -28.0,
    maxY: -10.0,
  },
};

export default function GrowthTrendChart({
  historicalObservations,
  forecastDates,
  forecastValues,
  predictionIntervals = {},
  selectedIndex,
  onSelectIndex,
  isDemo = false,
}: GrowthTrendChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<{
    date: string;
    isForecast: boolean;
    value: number;
    interval?: PredictionIntervalItem;
    cloud?: number;
    quality?: string;
    x: number;
    y: number;
  } | null>(null);

  const config = INDEX_CONFIG[selectedIndex] || INDEX_CONFIG["NDVI"];

  // Filter historical observations with non-null values for the active index
  const validHistorical = useMemo(() => {
    return historicalObservations
      .map((obs) => ({
        date: obs.date,
        value: obs.indices[selectedIndex] ?? null,
        cloud: obs.cloud_coverage,
        quality: obs.quality_status,
      }))
      .filter((item): item is { date: string; value: number; cloud: number; quality: string } => item.value !== null);
  }, [historicalObservations, selectedIndex]);

  const activeForecast = useMemo(() => {
    const vals = forecastValues[selectedIndex] || [];
    const intervals = predictionIntervals[selectedIndex] || [];
    const items: Array<{
      date: string;
      value: number;
      interval?: PredictionIntervalItem;
    }> = [];

    forecastDates.forEach((d, i) => {
      const v = vals[i];
      if (v !== undefined && v !== null && !isNaN(v)) {
        items.push({
          date: d,
          value: v,
          interval: intervals[i] || undefined,
        });
      }
    });

    return items;
  }, [forecastDates, forecastValues, predictionIntervals, selectedIndex]);

  // Combine all points to compute scales
  const allValues = useMemo(() => {
    const list: number[] = [];
    validHistorical.forEach((h) => list.push(h.value));
    activeForecast.forEach((f) => {
      list.push(f.value);
      if (f.interval) {
        list.push(f.interval.lower);
        list.push(f.interval.upper);
      }
    });
    return list;
  }, [validHistorical, activeForecast]);

  // SVG Dimensions
  const svgWidth = 840;
  const svgHeight = 360;
  const padding = { top: 35, right: 35, bottom: 50, left: 65 };
  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  // Total timeline points
  const timelineDates = useMemo(() => {
    const set = new Set<string>();
    validHistorical.forEach((h) => set.add(h.date));
    activeForecast.forEach((f) => set.add(f.date));
    return Array.from(set).sort();
  }, [validHistorical, activeForecast]);

  const dateToX = useMemo(() => {
    const map = new Map<string, number>();
    const count = timelineDates.length;
    if (count <= 1) {
      timelineDates.forEach((d) => map.set(d, padding.left + plotWidth / 2));
    } else {
      timelineDates.forEach((d, i) => {
        const x = padding.left + (i / (count - 1)) * plotWidth;
        map.set(d, x);
      });
    }
    return map;
  }, [timelineDates, plotWidth, padding.left]);

  // Y Scale calculation
  const { yMin, yMax } = useMemo(() => {
    if (allValues.length === 0) {
      return { yMin: config.minY, yMax: config.maxY };
    }
    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);
    const span = Math.max(0.05, dataMax - dataMin);
    const pad = span * 0.15;
    return {
      yMin: Math.max(config.minY, Number((dataMin - pad).toFixed(3))),
      yMax: Math.min(config.maxY + (config.unit === "dB" ? 5 : 0.1), Number((dataMax + pad).toFixed(3))),
    };
  }, [allValues, config]);

  const valToY = (val: number) => {
    const range = yMax - yMin;
    if (range <= 0) return padding.top + plotHeight / 2;
    const ratio = (val - yMin) / range;
    return padding.top + (1 - ratio) * plotHeight;
  };

  // Build Historical Line Path
  const historicalPath = useMemo(() => {
    if (validHistorical.length < 2) return "";
    return validHistorical
      .map((pt, i) => {
        const x = dateToX.get(pt.date) ?? padding.left;
        const y = valToY(pt.value);
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }, [validHistorical, dateToX, yMin, yMax]);

  // Build Forecast Line Path
  const forecastPath = useMemo(() => {
    if (activeForecast.length === 0) return "";
    // Connect from last historical point if available for seamless continuity
    const pts: Array<{ x: number; y: number }> = [];
    if (validHistorical.length > 0) {
      const lastHist = validHistorical[validHistorical.length - 1];
      pts.push({
        x: dateToX.get(lastHist.date) ?? padding.left,
        y: valToY(lastHist.value),
      });
    }
    activeForecast.forEach((pt) => {
      pts.push({
        x: dateToX.get(pt.date) ?? padding.left,
        y: valToY(pt.value),
      });
    });

    return pts
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
      .join(" ");
  }, [validHistorical, activeForecast, dateToX, yMin, yMax]);

  // Build Prediction Interval Polygon
  const intervalPolygon = useMemo(() => {
    const withIntervals = activeForecast.filter((f) => f.interval !== undefined);
    if (withIntervals.length === 0) return "";

    const upperPoints: Array<{ x: number; y: number }> = [];
    const lowerPoints: Array<{ x: number; y: number }> = [];

    // Anchor at last historical observation
    if (validHistorical.length > 0) {
      const lastHist = validHistorical[validHistorical.length - 1];
      const anchorX = dateToX.get(lastHist.date) ?? padding.left;
      const anchorY = valToY(lastHist.value);
      upperPoints.push({ x: anchorX, y: anchorY });
      lowerPoints.push({ x: anchorX, y: anchorY });
    }

    withIntervals.forEach((pt) => {
      const x = dateToX.get(pt.date) ?? padding.left;
      upperPoints.push({ x, y: valToY(pt.interval!.upper) });
      lowerPoints.push({ x, y: valToY(pt.interval!.lower) });
    });

    const forward = upperPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const backward = lowerPoints
      .reverse()
      .map((p) => `L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
      .join(" ");

    return `${forward} ${backward} Z`;
  }, [validHistorical, activeForecast, dateToX, yMin, yMax]);

  // Split line X position
  const splitX = useMemo(() => {
    if (validHistorical.length === 0) return null;
    const lastHist = validHistorical[validHistorical.length - 1];
    return dateToX.get(lastHist.date) ?? null;
  }, [validHistorical, dateToX]);

  // Y-axis grid ticks
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const count = 5;
    const step = (yMax - yMin) / count;
    for (let i = 0; i <= count; i++) {
      ticks.push(Number((yMin + i * step).toFixed(3)));
    }
    return ticks;
  }, [yMin, yMax]);

  return (
    <div className="space-y-4">
      {/* Index Selector Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-400 font-semibold mr-1">Forecast Target:</span>
          {["NDVI", "NDMI", "NDRE", "EVI", "VV", "VH"].map((idx) => {
            const isSelected = selectedIndex === idx;
            const hasData =
              (forecastValues[idx] && forecastValues[idx].length > 0) ||
              historicalObservations.some((o) => o.indices[idx] !== null && o.indices[idx] !== undefined);

            return (
              <button
                key={idx}
                onClick={() => onSelectIndex && onSelectIndex(idx)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition flex items-center gap-1.5 ${
                  isSelected
                    ? "bg-slate-100 text-slate-950 shadow-md ring-2 ring-emerald-500/50"
                    : "bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800"
                } ${!hasData ? "opacity-50" : ""}`}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: INDEX_CONFIG[idx]?.color || "#10b981" }}
                />
                <span>{idx}</span>
                {!hasData && <span className="text-[10px] text-slate-500">(N/A)</span>}
              </button>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span
              className="w-3.5 h-1 rounded"
              style={{ backgroundColor: config.color }}
            />
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: config.color }} />
            <span>Observed</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-4 h-0.5 border-t-2 border-dashed"
              style={{ borderColor: config.color }}
            />
            <span className="w-2 h-2 rotate-45 border" style={{ borderColor: config.color }} />
            <span>Forecast</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-3.5 h-2.5 rounded border border-slate-600"
              style={{ backgroundColor: config.fillColor }}
            />
            <span>80% Interval</span>
          </div>
        </div>
      </div>

      {/* Target Description */}
      <div className="flex items-center justify-between text-xs text-slate-400 px-1">
        <div>
          <strong className="text-white font-semibold">{config.label}:</strong>{" "}
          <span>{config.description}</span>
        </div>
        {isDemo && (
          <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-mono font-bold">
            SYNTHETIC PROJECTION
          </span>
        )}
      </div>

      {/* SVG Canvas Container */}
      <div className="relative w-full overflow-x-auto bg-slate-950/80 rounded-2xl border border-slate-800 p-2 sm:p-4">
        {validHistorical.length === 0 && activeForecast.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-slate-400 gap-2">
            <span className="text-2xl">📡</span>
            <span className="text-sm font-semibold text-slate-300">
              No historical observations found for {selectedIndex}.
            </span>
            <span className="text-xs text-slate-500 max-w-md text-center">
              {selectedIndex.startsWith("V")
                ? "SAR backscatter data requires concurrent Sentinel-1 acquisitions within the temporal window."
                : "Multispectral Sentinel-2 observations are required for optical vegetation indices."}
            </span>
          </div>
        ) : (
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="w-full h-auto max-h-[380px] select-none"
          >
            <defs>
              <linearGradient id={`grad-${selectedIndex}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={config.color} stopOpacity="0.25" />
                <stop offset="100%" stopColor={config.color} stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {/* Grid Lines & Y Ticks */}
            {yTicks.map((tick) => {
              const y = valToY(tick);
              return (
                <g key={tick}>
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={svgWidth - padding.right}
                    y2={y}
                    stroke="#1e293b"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={padding.left - 10}
                    y={y + 4}
                    fill="#64748b"
                    fontSize="10"
                    fontFamily="monospace"
                    textAnchor="end"
                  >
                    {tick.toFixed(config.unit === "dB" ? 1 : 2)}
                    {config.unit ? ` ${config.unit}` : ""}
                  </text>
                </g>
              );
            })}

            {/* Forecast Split Line */}
            {splitX !== null && (
              <g>
                <line
                  x1={splitX}
                  y1={padding.top - 10}
                  x2={splitX}
                  y2={svgHeight - padding.bottom}
                  stroke="#475569"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                />
                <rect
                  x={splitX - 45}
                  y={padding.top - 24}
                  width="90"
                  height="16"
                  rx="4"
                  fill="#1e293b"
                  stroke="#334155"
                />
                <text
                  x={splitX}
                  y={padding.top - 12}
                  fill="#94a3b8"
                  fontSize="9"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  FORECAST START
                </text>
              </g>
            )}

            {/* Shaded Prediction Interval Band */}
            {intervalPolygon && (
              <path
                d={intervalPolygon}
                fill={config.fillColor}
                stroke="none"
                className="transition-all duration-300"
              />
            )}

            {/* Historical Observation Line */}
            {historicalPath && (
              <path
                d={historicalPath}
                fill="none"
                stroke={config.color}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* Forecast Line */}
            {forecastPath && (
              <path
                d={forecastPath}
                fill="none"
                stroke={config.color}
                strokeWidth="2.5"
                strokeDasharray="6 4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* Historical Points */}
            {validHistorical.map((pt, i) => {
              const cx = dateToX.get(pt.date) ?? padding.left;
              const cy = valToY(pt.value);
              const isHovered = hoveredPoint?.date === pt.date && !hoveredPoint.isForecast;

              return (
                <g key={`hist-${i}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 6 : 4}
                    fill={isHovered ? "#ffffff" : config.color}
                    stroke="#020617"
                    strokeWidth="2"
                    className="cursor-pointer transition-all"
                    onMouseEnter={() =>
                      setHoveredPoint({
                        date: pt.date,
                        isForecast: false,
                        value: pt.value,
                        cloud: pt.cloud,
                        quality: pt.quality,
                        x: cx,
                        y: cy,
                      })
                    }
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                </g>
              );
            })}

            {/* Forecast Points */}
            {activeForecast.map((pt, i) => {
              const cx = dateToX.get(pt.date) ?? padding.left;
              const cy = valToY(pt.value);
              const isHovered = hoveredPoint?.date === pt.date && hoveredPoint.isForecast;

              return (
                <g key={`fc-${i}`}>
                  {/* Diamond shape for forecast */}
                  <rect
                    x={cx - (isHovered ? 5 : 4)}
                    y={cy - (isHovered ? 5 : 4)}
                    width={isHovered ? 10 : 8}
                    height={isHovered ? 10 : 8}
                    transform={`rotate(45 ${cx} ${cy})`}
                    fill={isHovered ? "#ffffff" : "#0f172a"}
                    stroke={config.color}
                    strokeWidth="2"
                    className="cursor-pointer transition-all"
                    onMouseEnter={() =>
                      setHoveredPoint({
                        date: pt.date,
                        isForecast: true,
                        value: pt.value,
                        interval: pt.interval,
                        x: cx,
                        y: cy,
                      })
                    }
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                </g>
              );
            })}

            {/* X-axis Date Labels (Sampled to avoid crowding) */}
            {timelineDates.map((dateStr, idx) => {
              const total = timelineDates.length;
              // Sample every N dates if crowded
              const step = total > 12 ? Math.ceil(total / 8) : 1;
              if (idx % step !== 0 && idx !== total - 1) return null;

              const x = dateToX.get(dateStr) ?? padding.left;
              const isForecastDate = activeForecast.some((f) => f.date === dateStr);

              return (
                <text
                  key={dateStr}
                  x={x}
                  y={svgHeight - padding.bottom + 22}
                  fill={isForecastDate ? "#38bdf8" : "#94a3b8"}
                  fontSize="10"
                  fontFamily="monospace"
                  textAnchor="middle"
                  fontWeight={isForecastDate ? "bold" : "normal"}
                >
                  {dateStr.slice(5)}
                </text>
              );
            })}

            {/* Tooltip Overlay */}
            {hoveredPoint && (
              <g
                transform={`translate(${
                  hoveredPoint.x > svgWidth - 170 ? hoveredPoint.x - 160 : hoveredPoint.x + 12
                }, ${Math.max(padding.top, hoveredPoint.y - 65)})`}
                className="pointer-events-none"
              >
                <rect
                  width="150"
                  height={hoveredPoint.isForecast ? 72 : 62}
                  rx="8"
                  fill="#0f172a"
                  stroke="#334155"
                  strokeWidth="1"
                  filter="drop-shadow(0 4px 6px rgba(0, 0, 0, 0.4))"
                />
                <text x="12" y="18" fill="#94a3b8" fontSize="10" fontFamily="monospace">
                  📅 {hoveredPoint.date}
                </text>
                <text x="12" y="34" fill="#ffffff" fontSize="12" fontWeight="bold">
                  {selectedIndex}: {hoveredPoint.value.toFixed(config.unit === "dB" ? 2 : 4)}
                  {config.unit ? ` ${config.unit}` : ""}
                </text>
                {hoveredPoint.isForecast ? (
                  <text x="12" y="52" fill="#38bdf8" fontSize="9" fontFamily="monospace">
                    {hoveredPoint.interval
                      ? `80% CI: [${hoveredPoint.interval.lower.toFixed(2)}, ${hoveredPoint.interval.upper.toFixed(2)}]`
                      : "Forecast (Point Est.)"}
                  </text>
                ) : (
                  <text x="12" y="50" fill="#10b981" fontSize="9" fontFamily="monospace">
                    Observed • Cloud: {hoveredPoint.cloud?.toFixed(0) ?? 0}%
                  </text>
                )}
              </g>
            )}
          </svg>
        )}
      </div>
    </div>
  );
}
