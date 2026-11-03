"use client";

import Link from "next/link";
import { API_ROUTES } from "@/config/api";

interface TargetMetric {
  name: string;
  target: string;
  description: string;
}

interface BenchmarkCategory {
  title: string;
  icon: string;
  accent: string;
  description: string;
  metrics: TargetMetric[];
  evaluationDatasetPlanned: string;
}

const BENCHMARKS: BenchmarkCategory[] = [
  {
    title: "CROP CONDITION",
    icon: "🎯",
    accent: "emerald",
    description: "Multimodal classification of crop condition into empirical health tiers using fused optical spectral and SAR roughness features.",
    evaluationDatasetPlanned: "Stratified test split with ground-truth agronomist field survey observations across distinct agro-climatic zones.",
    metrics: [
      { name: "Accuracy", target: "91.4%", description: "Target overall multi-class classification accuracy across condition zones." },
      { name: "Precision", target: "89.7%", description: "Target macro-averaged precision minimizing false healthy/stressed classifications." },
      { name: "Recall", target: "90.8%", description: "Target sensitivity in identifying true compromised canopy zones." },
      { name: "F1 Score", target: "90.2%", description: "Target harmonic mean balancing precision and sensitivity." },
    ],
  },
  {
    title: "EARLY STRESS DETECTION",
    icon: "⚡",
    accent: "amber",
    description: "Sub-visual stress detection leveraging SAR microwave canopy penetration before chlorophyll depletion becomes visible in optical red bands.",
    evaluationDatasetPlanned: "Time-series validation split tracking early irrigation cutoff and nitrogen deficiency test plots.",
    metrics: [
      { name: "Precision", target: "87.3%", description: "Target positive predictive value for true vegetative or moisture stress." },
      { name: "Recall", target: "84.9%", description: "Target detection rate for incipient stress events 7-10 days pre-symptom." },
      { name: "F1 Score", target: "86.1%", description: "Target balanced performance on imbalanced early-stress datasets." },
      { name: "ROC-AUC", target: "0.91", description: "Target area under the receiver operating characteristic curve." },
    ],
  },
  {
    title: "GROWTH & TREND PREDICTION",
    icon: "📈",
    accent: "sky",
    description: "Multi-temporal trajectory forecasting combining historical optical trajectory curves and SAR biophysical structural accumulation.",
    evaluationDatasetPlanned: "Multi-year temporal backtesting across full crop phenology lifecycles.",
    metrics: [
      { name: "R² (Coeff of Determination)", target: "0.88", description: "Target explained variance across multi-temporal biomass trajectories." },
      { name: "RMSE", target: "0.31", description: "Target root-mean-squared error in normalized biomass index units." },
      { name: "MAE", target: "0.22", description: "Target mean absolute error in trajectory projection." },
      { name: "MAPE", target: "6.4%", description: "Target mean absolute percentage error across canopy growth stages." },
    ],
  },
  {
    title: "FIELD BOUNDARY & RISK MAPPING",
    icon: "🗺️",
    accent: "purple",
    description: "Spatial segmentation and field risk zoning combining high-contrast optical edges with dielectric SAR boundary reflections.",
    evaluationDatasetPlanned: "Cadastral field boundaries and high-resolution drone orthomosaics.",
    metrics: [
      { name: "IoU (Intersection over Union)", target: "0.86", description: "Target Jaccard similarity index for agricultural parcel delineation." },
      { name: "Dice Coefficient", target: "0.91", description: "Target spatial overlap metric between predicted and true parcel masks." },
    ],
  },
];

export default function ModelPerformancePage() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-sky-500 selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-xl overflow-hidden border border-emerald-500/40 bg-emerald-950/60 shadow-sm flex items-center justify-center">
                <img src="/logo.png" alt="Crop Health AI Logo" className="h-full w-full object-cover" />
              </div>
              <span className="font-bold text-lg text-white tracking-tight">CropHealth</span>
            </Link>
            <span className="text-xs text-sky-400 font-medium px-2.5 py-0.5 rounded-full bg-sky-500/10 border border-sky-500/20">
              Future AI Architecture
            </span>
          </div>

          <nav className="flex items-center gap-6">
            <Link href="/" className="text-sm font-medium text-slate-400 hover:text-white transition">
              Home
            </Link>
            <Link href="/analyze" className="text-sm font-medium text-slate-400 hover:text-white transition">
              Analyze
            </Link>
            <Link href="/history" className="text-sm font-medium text-slate-400 hover:text-white transition flex items-center gap-1">
              <span>📈</span>
              <span>History</span>
            </Link>
            <span className="text-sm font-semibold text-sky-400 border-b-2 border-sky-400 pb-0.5">
              Model Benchmarks
            </span>
            <a
              href={API_ROUTES.DOCS}
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono text-slate-400 hover:text-sky-400 transition hidden sm:inline"
            >
              API Docs ↗
            </a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-10 space-y-10">
        {/* Title */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-sky-500/30 bg-sky-500/10 text-sky-300 text-xs font-semibold uppercase tracking-wider">
            <span>🔬</span>
            <span>Target Specification Benchmarks</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight">
            AI Model Performance Architecture
          </h1>
          <p className="text-slate-400 text-sm sm:text-base max-w-3xl mx-auto leading-relaxed">
            Target performance metrics and evaluation methodology for the planned multimodal fusion and predictive intelligence models.
          </p>
        </div>

        {/* Mandatory Ethical & Scientific Disclosure Banner */}
        <div className="p-6 rounded-2xl bg-amber-950/40 border-2 border-amber-500/60 text-amber-200 text-xs sm:text-sm space-y-2 shadow-xl backdrop-blur-md">
          <div className="flex items-center gap-2.5 font-bold text-amber-400 text-sm sm:text-base">
            <span className="text-lg">⚠️</span>
            <span>SPECIFICATION TARGET BENCHMARK DISCLOSURE (MULTIMODAL PROTOCOL)</span>
          </div>
          <p className="leading-relaxed text-amber-200/90">
            The quantitative metrics displayed below represent <strong>TARGET BENCHMARKS</strong> supplied by the product specification for predictive AI models. The platform implements the real <strong>Sentinel-1 C-band SAR layer</strong> and the <strong>Multimodal Fusion preparation service</strong> (data alignment, NoData handling, and temporal matching).
          </p>
          <p className="leading-relaxed text-amber-300/80 font-mono text-[11px]">
            Status: Held-out empirical validation and fusion model training are conducted in scientific field trials. No fabricated performance results are reported as achieved.
          </p>
        </div>

        {/* Planned Multimodal Fusion Architecture Diagram */}
        <section className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-xl">
          <div className="border-b border-slate-800 pb-4">
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <span>🏗️ Planned Target Architecture Pipeline</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              End-to-end data flow from multi-sensor orbit acquisition to farmer-ready explainable insights.
            </p>
          </div>

          <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 font-mono text-xs overflow-x-auto text-slate-300">
            <pre className="leading-relaxed text-sky-300">
{`USER LOCATION / AGRICULTURAL AOI
↓
SATELLITE DATA ACQUISITION
↓
┌────────────────────────────────────────────────────────┐
│  Sentinel-2 (Optical L2A)   │   Sentinel-1 (SAR GRD)   │
│  B02, B03, B04, B05, B08    │   VV Backscatter (dB)    │
│  NDVI, NDMI, NDRE, EVI      │   VH Backscatter (dB)    │
│  Canopy Chlorophyll & Water │   VV/VH Cross-Ratio (dB) │
└────────────────────────────────────────────────────────┘
↓
SPECTRAL FEATURES + SAR FEATURES
↓
DATA FUSION SERVICE (CRS, Extent, Resampling, NoData Filtering)
↓
HISTORICAL MULTI-TEMPORAL TRAJECTORY
↓
MULTIMODAL FUSION AI MODELS
↓
CURRENT STATE + EARLY STRESS DETECTION + GROWTH PREDICTION
↓
EXPLAINABLE AI (Feature Attributions & Spectral Evidence)
↓
CONFIDENCE ESTIMATION + UNCERTAINTY QUANTIFICATION
↓
HIGH-RESOLUTION FIELD RISK MAP → FARMER DASHBOARD`}
            </pre>
          </div>
        </section>

        {/* 4 Target Benchmark Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {BENCHMARKS.map((bench) => (
            <div
              key={bench.title}
              className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-5 shadow-xl hover:border-slate-700 transition"
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{bench.icon}</span>
                    <h3 className="font-black text-white text-base tracking-tight">{bench.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{bench.description}</p>
                </div>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20 whitespace-nowrap">
                  Target Goal
                </span>
              </div>

              {/* Target Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                {bench.metrics.map((m) => (
                  <div
                    key={m.name}
                    className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1"
                  >
                    <span className="text-[11px] text-slate-400 block font-medium">{m.name}</span>
                    <div className="text-2xl font-black text-white font-mono">{m.target}</div>
                    <p className="text-[10px] text-slate-500 leading-tight">{m.description}</p>
                  </div>
                ))}
              </div>

              {/* Planned Evaluation Protocol */}
              <div className="pt-2 border-t border-slate-800/80 space-y-1 text-xs">
                <span className="text-[11px] font-semibold text-slate-400 block">
                  Planned Evaluation Protocol:
                </span>
                <p className="text-[11px] text-slate-400 leading-relaxed font-mono">
                  {bench.evaluationDatasetPlanned}
                </p>
                <div className="pt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span>Target Source: Project Specification</span>
                  <span className="text-amber-400">Status: Pending Future Training</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Multimodal Ingestion Readiness Checklist */}
        <section className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-5 shadow-xl">
          <div className="border-b border-slate-800 pb-3">
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <span>✅ Multimodal Ingestion Readiness</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Infrastructure completed to prepare aligned inputs for predictive satellite models.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Sentinel-1 C-Band SAR Service</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Direct integration querying CDSE / Sentinel Hub for Sentinel-1 GRD. Backscatter calculation in linear and decibel units with VV/VH cross-ratio.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Temporal Window Matching</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Matches Sentinel-2 optical scenes with the closest Sentinel-1 observation within a configurable 7-day window with explicit out-of-window handling.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Fusion Dataset Preparation</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Created <code>backend/services/fusion_service.py</code> to assemble aligned multimodal feature tensors (optical + SAR + quality + CRS transform).
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Spatial Raster Alignment</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Rigorous bilinear/nearest-neighbor raster resampling ensuring optical and SAR 2D arrays match pixel-for-pixel before tensor ingestion.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Deterministic DEMO Mode</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Fixed, reproducible SAR arrays for offline demonstration and testing without external credential dependency.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">✓ Multi-Layer Interactive Map</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Leaflet interface toggling between Satellite RGB, NDVI, NDMI, NDRE, EVI, Crop Condition, S1 VV, S1 VH, and VV/VH Ratio.
              </p>
            </div>
          </div>
        </section>

        {/* Back Link */}
        <div className="text-center pt-4">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 text-sm font-semibold transition"
          >
            <span>← Return to Field Analysis</span>
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Crop Health AI. Sentinel-1 & Sentinel-2 Multimodal Architecture.</p>
          <p className="text-slate-400">Multimodal Remote Sensing & Predictive Architecture</p>
        </div>
      </footer>
    </div>
  );
}
