"use client";

import React, { useState } from "react";

export interface FieldComparisonItemData {
  field_id: string;
  field_name: string;
  crop_type: string;
  area_hectares: number;
  observation_date: string;
  data_quality_tier: string;
  ndvi_mean: number | null;
  ndmi_mean: number | null;
  ndre_mean: number | null;
  evi_mean: number | null;
  growth_trend_slope: number | null;
  elevated_risk_pct: number;
  stress_probability: number | null;
  stress_class: string | null;
}

export interface FieldComparisonResponseData {
  status: string;
  compared_fields: FieldComparisonItemData[];
  temporal_alignment_compatible: boolean;
  temporal_discrepancy_days: number;
  comparison_warnings: string[];
  is_demo: boolean;
}

interface FieldComparisonPanelProps {
  comparisonData: FieldComparisonResponseData | null;
  availableFieldIds: { id: string; name: string }[];
  onRunComparison: (fieldIds: string[]) => void;
  isLoading?: boolean;
}

export default function FieldComparisonPanel({
  comparisonData,
  availableFieldIds,
  onRunComparison,
  isLoading = false,
}: FieldComparisonPanelProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>(
    availableFieldIds.slice(0, 2).map((f) => f.id)
  );

  const toggleFieldSelection = (id: string) => {
    if (selectedIds.includes(id)) {
      if (selectedIds.length > 2) {
        setSelectedIds(selectedIds.filter((item) => item !== id));
      }
    } else {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const handleCompareClick = () => {
    if (selectedIds.length >= 2) {
      onRunComparison(selectedIds);
    }
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">⚖️</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Multi-Field Comparative Intelligence
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-950/80 text-sky-300 border border-sky-800/60 uppercase">
              Side-by-Side Analysis
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Compare phenological metrics, canopy vigor, risk coverage, and growth trajectories across multiple agricultural parcels.
          </p>
        </div>

        <button
          onClick={handleCompareClick}
          disabled={isLoading || selectedIds.length < 2}
          className="px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white transition flex items-center gap-2 shadow-sm self-start sm:self-auto"
        >
          {isLoading ? (
            <>
              <span className="animate-spin text-sm">↻</span>
              <span>Comparing Fields...</span>
            </>
          ) : (
            <>
              <span>⚖️</span>
              <span>Run Field Comparison</span>
            </>
          )}
        </button>
      </div>

      {/* Field Selector Chips */}
      <div className="mb-5 bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80">
        <span className="text-xs font-semibold text-slate-300 block mb-2">
          Select Fields to Compare (Minimum 2):
        </span>
        <div className="flex flex-wrap gap-2">
          {availableFieldIds.map((f) => {
            const isSelected = selectedIds.includes(f.id);
            return (
              <button
                key={f.id}
                onClick={() => toggleFieldSelection(f.id)}
                className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition flex items-center gap-1.5 ${
                  isSelected
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-sm"
                    : "bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700"
                }`}
              >
                <span>{isSelected ? "✓" : "+"}</span>
                <span>{f.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Temporal Incompatibility / Discrepancy Alert */}
      {comparisonData && !comparisonData.temporal_alignment_compatible && (
        <div className="mb-5 bg-amber-950/40 border border-amber-500/40 rounded-xl p-4 flex items-start gap-3">
          <span className="text-xl flex-shrink-0">⚠️</span>
          <div>
            <h4 className="text-xs font-black text-amber-300 uppercase tracking-wide">
              Temporal Alignment Warning: {comparisonData.temporal_discrepancy_days} Days Discrepancy
            </h4>
            <div className="text-xs text-amber-200/80 mt-1 space-y-1">
              {comparisonData.comparison_warnings.map((warn, i) => (
                <p key={i}>{warn}</p>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Note: Vegetative indices compared across differing phenological growth stages may reflect seasonal maturity rather than relative crop health.
            </p>
          </div>
        </div>
      )}

      {/* Comparison Table */}
      {comparisonData && comparisonData.compared_fields && comparisonData.compared_fields.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
                <th className="py-3 px-3">Field Name</th>
                <th className="py-3 px-3">Crop Type</th>
                <th className="py-3 px-3 font-mono">Area (ha)</th>
                <th className="py-3 px-3">Observation</th>
                <th className="py-3 px-3 font-mono">NDVI</th>
                <th className="py-3 px-3 font-mono">NDMI</th>
                <th className="py-3 px-3 font-mono">NDRE</th>
                <th className="py-3 px-3 font-mono">EVI</th>
                <th className="py-3 px-3 font-mono">Growth Slope</th>
                <th className="py-3 px-3">Elevated Risk %</th>
                <th className="py-3 px-3">Stress Prob</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {comparisonData.compared_fields.map((f) => (
                <tr key={f.field_id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3 px-3 font-bold text-white font-sans">
                    {f.field_name}
                  </td>
                  <td className="py-3 px-3 text-emerald-400 font-sans">
                    {f.crop_type}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {f.area_hectares.toFixed(1)} ha
                  </td>
                  <td className="py-3 px-3 text-sky-300">
                    {f.observation_date}
                  </td>
                  <td className="py-3 px-3 font-bold text-emerald-400">
                    {f.ndvi_mean !== null ? f.ndvi_mean.toFixed(3) : "—"}
                  </td>
                  <td className="py-3 px-3 text-sky-400">
                    {f.ndmi_mean !== null ? f.ndmi_mean.toFixed(3) : "—"}
                  </td>
                  <td className="py-3 px-3 text-amber-400">
                    {f.ndre_mean !== null ? f.ndre_mean.toFixed(3) : "—"}
                  </td>
                  <td className="py-3 px-3 text-emerald-300">
                    {f.evi_mean !== null ? f.evi_mean.toFixed(3) : "—"}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {f.growth_trend_slope !== null
                      ? `${(f.growth_trend_slope * 100).toFixed(2)}%/d`
                      : "—"}
                  </td>
                  <td className="py-3 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        f.elevated_risk_pct > 15
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                      }`}
                    >
                      {f.elevated_risk_pct.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    {f.stress_probability !== null ? (
                      <span className="text-slate-300">
                        {(f.stress_probability * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-slate-500">N/A</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-6 text-slate-500 text-xs">
          Select 2 or more fields above and click &quot;Run Field Comparison&quot; to inspect comparative metrics.
        </div>
      )}
    </div>
  );
}
