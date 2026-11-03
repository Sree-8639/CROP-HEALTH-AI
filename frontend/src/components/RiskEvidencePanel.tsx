"use client";

import React, { useState } from "react";

export interface RiskEvidenceData {
  classification_methodology: {
    rule_name: string;
    elevated_risk_threshold: string;
    moderate_risk_threshold: string;
    low_risk_threshold: string;
    insufficient_evidence_threshold: string;
  };
  data_quality: {
    valid_pixel_percentage: number;
    nodata_percentage: number;
    observation_count: number;
    s1_available: boolean;
    s2_available: boolean;
  };
  model_provenance: {
    model_name: string;
    model_version: string;
    is_demo_model: boolean;
  };
  warnings: string[];
  scientific_limitations: string[];
}

interface RiskEvidencePanelProps {
  evidence: RiskEvidenceData;
}

export default function RiskEvidencePanel({ evidence }: RiskEvidencePanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const method = evidence.classification_methodology || {};
  const dq = evidence.data_quality || {};
  const prov = evidence.model_provenance || {};

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 backdrop-blur-md shadow-xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-base">🔬</span>
          <h4 className="text-sm font-bold text-white tracking-wide">
            Classification Methodology & Scientific Provenance
          </h4>
        </div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          type="button"
          className="px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
        >
          {isOpen ? "Collapse Details" : "Inspect Rules"}
        </button>
      </div>

      {/* Rules Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-1.5 text-rose-400 font-semibold text-xs">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span>Elevated Potential Risk Rule</span>
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed font-mono">
            {method.elevated_risk_threshold || "Stress Prob ≥ 0.65 OR (Stress Prob ≥ 0.50 AND NDMI < 0.0) OR Severe Spectral Collapse"}
          </p>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold text-xs">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span>Moderate Potential Risk Rule</span>
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed font-mono">
            {method.moderate_risk_threshold || "Stress Prob in [0.35, 0.65) OR NDMI < 0.12 OR NDRE < 0.22 OR Negative ΔNDMI"}
          </p>
        </div>
      </div>

      {/* Collapsible Deep-Dive Details */}
      {isOpen && (
        <div className="space-y-4 pt-2 border-t border-slate-800/80 animate-in fade-in duration-200">
          {/* Provenance & Inputs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
            <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1">
                Active AI Model
              </span>
              <span className="font-semibold text-white block truncate">
                {prov.model_name || "MultimodalCropStressDetector"}
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                Version: {prov.model_version || "v1.0.0"}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1">
                Sensor Alignment
              </span>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    dq.s2_available
                      ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30"
                      : "bg-slate-800 text-slate-400"
                  }`}
                >
                  Sentinel-2 {dq.s2_available ? "✓" : "✗"}
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    dq.s1_available
                      ? "bg-sky-500/10 text-sky-300 border border-sky-500/30"
                      : "bg-slate-800 text-slate-400"
                  }`}
                >
                  Sentinel-1 {dq.s1_available ? "✓" : "✗"}
                </span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1">
                Temporal Stack Depth
              </span>
              <span className="text-base font-bold font-mono text-slate-100">
                {dq.observation_count ?? 1} acquisitions
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Multi-temporal baseline
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1">
                NoData Protection
              </span>
              <span className="text-xs font-semibold text-slate-200 block">
                Strict Quality Mask
              </span>
              <span className="text-[10px] text-slate-400 leading-tight block mt-0.5">
                Clouds &amp; shadows are never converted to zero values
              </span>
            </div>
          </div>

          {/* Scientific Limitations & Non-Causation Advisory */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="flex items-center gap-1.5 text-amber-400 text-xs font-semibold">
              <span>⚠️</span>
              <span>Agronomic Non-Causation Disclaimer &amp; Limitations</span>
            </div>
            <ul className="text-xs text-slate-300 space-y-1.5 list-disc pl-4 leading-relaxed">
              {evidence.scientific_limitations?.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
