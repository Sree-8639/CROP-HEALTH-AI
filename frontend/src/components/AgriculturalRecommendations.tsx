"use client";

import React from "react";

export interface AgriculturalRecommendationData {
  id: string;
  category: string;
  title: string;
  action: string;
  reason: string;
  evidence_source: string;
  evidence_date: string;
  data_quality_limitations: string[];
  field_verification_required: boolean;
  priority: string;
  created_at: string;
}

interface AgriculturalRecommendationsProps {
  recommendations: AgriculturalRecommendationData[];
  cropType?: string;
  isDemo?: boolean;
}

export default function AgriculturalRecommendations({
  recommendations,
  cropType,
  isDemo = false,
}: AgriculturalRecommendationsProps) {
  const getPriorityStyle = (priority: string) => {
    switch (priority.toLowerCase()) {
      case "high":
        return {
          badge: "bg-rose-500/20 text-rose-300 border-rose-500/40",
          cardBorder: "border-rose-900/40",
          icon: "⚡",
        };
      case "medium":
        return {
          badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
          cardBorder: "border-amber-900/30",
          icon: "⚠️",
        };
      case "low":
        return {
          badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
          cardBorder: "border-emerald-900/30",
          icon: "🌱",
        };
      default:
        return {
          badge: "bg-sky-500/20 text-sky-300 border-sky-500/40",
          cardBorder: "border-sky-900/30",
          icon: "ℹ️",
        };
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case "field scouting":
        return "🔍";
      case "moisture verification":
        return "💧";
      case "satellite recheck":
        return "🛰️";
      case "nutrient assessment":
        return "🧪";
      case "temporal review":
        return "⏱️";
      default:
        return "📋";
    }
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">📋</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Evidence-Based Agricultural Recommendations
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
              NON-PRESCRIPTIVE
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Observational follow-up actions and ground-truth verification steps linked directly to satellite evidence.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
            {recommendations.length} Action Items
          </span>
        </div>
      </div>

      {/* Safety Notice Banner */}
      <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 mb-5 flex items-start gap-2.5 text-xs text-slate-400">
        <span className="text-base flex-shrink-0">🛡️</span>
        <div>
          <strong className="text-slate-300">Agronomic Safety Principle:</strong> Recommendations suggest observational scouting, physical soil testing, or sensor review. This platform does not prescribe fertilizers, chemical formulations, pesticide applications, or irrigation dosages without physical agronomist inspection.
        </div>
      </div>

      {/* Recommendations Cards List */}
      {recommendations.length === 0 ? (
        <div className="text-center py-8 text-slate-500 text-xs">
          No active recommendations for this field observation.
        </div>
      ) : (
        <div className="space-y-3.5">
          {recommendations.map((rec) => {
            const pStyle = getPriorityStyle(rec.priority);
            const catIcon = getCategoryIcon(rec.category);

            return (
              <div
                key={rec.id}
                className={`bg-slate-950/70 rounded-xl p-4 border ${pStyle.cardBorder} hover:border-slate-700 transition flex flex-col justify-between`}
              >
                <div>
                  {/* Top bar with category and priority badge */}
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{catIcon}</span>
                      <span className="text-xs font-bold text-slate-300 uppercase tracking-wide">
                        {rec.category}
                      </span>
                      <span className="text-slate-600">•</span>
                      <h4 className="text-sm font-black text-white tracking-tight">
                        {rec.title}
                      </h4>
                    </div>

                    <div className="flex items-center gap-2">
                      {rec.field_verification_required && (
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                          <span>🔍</span>
                          <span>Field Verification Required</span>
                        </span>
                      )}

                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${pStyle.badge}`}
                      >
                        {pStyle.icon} {rec.priority}
                      </span>
                    </div>
                  </div>

                  {/* Action */}
                  <div className="bg-slate-900/80 rounded-lg p-3 border border-slate-800/80 mb-3 text-xs leading-relaxed text-slate-200">
                    <strong className="text-emerald-400 font-semibold block mb-0.5">
                      Suggested Action:
                    </strong>
                    {rec.action}
                  </div>

                  {/* Reason & Evidence */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-400 mb-3">
                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 block mb-0.5">
                        Scientific Rationale
                      </span>
                      <p className="leading-snug text-slate-300">{rec.reason}</p>
                    </div>

                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 block mb-0.5">
                        Evidence Base ({rec.evidence_date})
                      </span>
                      <p className="leading-snug font-mono text-[11px] text-sky-300">
                        {rec.evidence_source}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Data Quality Limitations */}
                {rec.data_quality_limitations && rec.data_quality_limitations.length > 0 && (
                  <div className="pt-2 border-t border-slate-800/60 text-[10px] text-slate-500 flex items-center gap-1.5">
                    <span>⚠️</span>
                    <span>Limitations: {rec.data_quality_limitations.join(" • ")}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
