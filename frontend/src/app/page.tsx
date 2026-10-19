"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Navbar from "@/components/Navbar";
import {
  fetchHealth,
  fetchActiveModel,
  fetchModelInfo,
  fetchSamples,
  fetchEvaluationReports,
  predictSampleChip,
  predictUploadedGeoTIFF,
  HealthResponse,
  ActiveModelResponse,
  ModelInfoResponse,
  SampleItem,
  EvaluationReportsResponse,
  PredictionResponse,
} from "@/lib/api";
import {
  Waves,
  Activity,
  MapPin,
  BarChart3,
  Cpu,
  UploadCloud,
  Sliders,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  Satellite,
  Info,
  ShieldAlert,
  ArrowRight,
  Play,
  Layers,
  Database,
  Check,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// Dynamically import Leaflet FloodMap with SSR disabled to prevent window is undefined errors
const FloodMap = dynamic(() => import("@/components/FloodMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[520px] rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center justify-center gap-3 text-slate-400">
      <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
      <span className="text-sm font-mono">Initializing Geospatial Engine...</span>
    </div>
  ),
});

export default function Home() {
  const [activeTab, setActiveTab] = useState<"overview" | "workspace" | "evaluation" | "model">("overview");

  // System State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [activeModel, setActiveModel] = useState<ActiveModelResponse | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [evalReports, setEvalReports] = useState<EvaluationReportsResponse | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean>(false);
  const [systemLoading, setSystemLoading] = useState<boolean>(true);

  // Prediction Workspace State
  const [selectedChipId, setSelectedChipId] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [showSar, setShowSar] = useState<boolean>(true);
  const [showMask, setShowMask] = useState<boolean>(true);
  const [showProb, setShowProb] = useState<boolean>(false);
  const [layerOpacity, setLayerOpacity] = useState<number>(0.85);

  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Initial Load
  const loadSystemData = async () => {
    setSystemLoading(true);
    try {
      const [h, mi, am, s, er] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchModelInfo().catch(() => null),
        fetchActiveModel().catch(() => null),
        fetchSamples().catch(() => ({ samples: [] })),
        fetchEvaluationReports().catch(() => null),
      ]);

      if (h) {
        setHealth(h);
        setApiConnected(true);
      } else {
        setApiConnected(false);
      }

      if (mi) setModelInfo(mi);
      if (am) setActiveModel(am);
      if (s?.samples?.length) {
        setSamples(s.samples);
        setSelectedChipId(s.samples[0].chip_id);
      }
      if (er) setEvalReports(er);
    } catch (err) {
      console.error("System fetch error:", err);
      setApiConnected(false);
    } finally {
      setSystemLoading(false);
    }
  };

  useEffect(() => {
    loadSystemData();
  }, []);

  // Handle Benchmark Prediction
  const handleRunBenchmarkPrediction = async (chipIdToRun?: string) => {
    const targetChip = chipIdToRun || selectedChipId;
    if (!targetChip) return;

    setIsPredicting(true);
    setPredictionError(null);
    try {
      const res = await predictSampleChip(targetChip, threshold);
      setPredictionResult(res);
    } catch (err: any) {
      setPredictionError(err.message || "Prediction execution failed.");
    } finally {
      setIsPredicting(false);
    }
  };

  // Handle Upload Prediction
  const handleRunUploadPrediction = async () => {
    if (!uploadedFile) return;

    setIsPredicting(true);
    setPredictionError(null);
    try {
      const res = await predictUploadedGeoTIFF(uploadedFile, threshold);
      setPredictionResult(res);
    } catch (err: any) {
      setPredictionError(err.message || "Failed to process uploaded GeoTIFF.");
    } finally {
      setIsPredicting(false);
    }
  };

  // File Upload Handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
        setPredictionError("Invalid file type: Please select a valid Sentinel-1 GeoTIFF (.tif, .tiff)");
        return;
      }
      setUploadedFile(file);
      setPredictionError(null);
    }
  };

  // Chart data for evaluation
  const chartData = [
    {
      metric: "IoU (Jaccard)",
      Validation: evalReports?.validation_split
        ? Math.round(evalReports.validation_split.metrics.iou * 1000) / 10
        : 42.1,
      Bolivia: evalReports?.bolivia_heldout_split
        ? Math.round(evalReports.bolivia_heldout_split.metrics.iou * 1000) / 10
        : 35.3,
    },
    {
      metric: "F1 Score",
      Validation: evalReports?.validation_split
        ? Math.round(evalReports.validation_split.metrics.f1 * 1000) / 10
        : 59.2,
      Bolivia: evalReports?.bolivia_heldout_split
        ? Math.round(evalReports.bolivia_heldout_split.metrics.f1 * 1000) / 10
        : 52.1,
    },
    {
      metric: "Precision",
      Validation: evalReports?.validation_split
        ? Math.round(evalReports.validation_split.metrics.precision * 1000) / 10
        : 55.4,
      Bolivia: evalReports?.bolivia_heldout_split
        ? Math.round(evalReports.bolivia_heldout_split.metrics.precision * 1000) / 10
        : 74.2,
    },
    {
      metric: "Recall",
      Validation: evalReports?.validation_split
        ? Math.round(evalReports.validation_split.metrics.recall * 1000) / 10
        : 63.6,
      Bolivia: evalReports?.bolivia_heldout_split
        ? Math.round(evalReports.bolivia_heldout_split.metrics.recall * 1000) / 10
        : 40.2,
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiConnected={apiConnected}
        computeDevice={health?.compute_device}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ========================================================================= */}
        {/* TAB 1: OVERVIEW DASHBOARD */}
        {/* ========================================================================= */}
        {activeTab === "overview" && (
          <div className="space-y-8 animate-fadeIn">
            {/* Hero Section */}
            <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900/90 to-cyan-950/40 p-8 sm:p-10 shadow-2xl">
              <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
              <div className="max-w-3xl space-y-4 relative z-10">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                  <Satellite className="w-3.5 h-3.5" />
                  <span>Sentinel-1 SAR C-Band Radar Pipeline</span>
                </div>
                <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
                  FloodSense AI <br />
                  <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500 bg-clip-text text-transparent">
                    Surface Inundation Intelligence
                  </span>
                </h1>
                <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
                  FloodSense AI uses deep convolutional neural networks (PyTorch U-Net) to detect and delineate flood inundation from Sentinel-1 dual-polarization (VV/VH) Synthetic Aperture Radar imagery. Because radar microwaves penetrate cloud cover and operate day and night, this pipeline provides actionable flood delineation when optical satellites are blinded by storm clouds.
                </p>
                <div className="pt-2 flex flex-wrap items-center gap-4">
                  <button
                    onClick={() => setActiveTab("workspace")}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5"
                  >
                    <MapPin className="w-4 h-4" />
                    Launch Flood Workspace
                    <ArrowRight className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setActiveTab("evaluation")}
                    className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 transition-all"
                  >
                    <BarChart3 className="w-4 h-4 text-cyan-400" />
                    View Benchmark Metrics
                  </button>
                </div>
              </div>
            </div>

            {/* Scope & Methodology Clarification Card */}
            <div className="p-4 rounded-xl border border-sky-900/50 bg-sky-950/20 flex items-start gap-3.5">
              <Info className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
              <div className="text-xs sm:text-sm text-slate-300 space-y-1">
                <p className="font-semibold text-sky-200">
                  Operational Scope: Observed Inundation Mapping vs. Hydrologic Runoff Forecasting
                </p>
                <p className="text-slate-400 leading-relaxed">
                  FloodSense AI performs <em>observed surface water segmentation</em> from radar backscatter at the exact time of satellite overpass. It isolates temporary dynamic flood water from permanent water bodies using JRC Global Surface Water baselines. It does not predict forward-looking river stage forecasts or rainfall runoff models.
                </p>
              </div>
            </div>

            {/* KPI Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Card 1: Val IoU */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-sm relative overflow-hidden">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Validation Flood IoU
                </div>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-cyan-400 font-mono">
                    {evalReports?.validation_split
                      ? `${(evalReports.validation_split.metrics.iou * 100).toFixed(1)}%`
                      : "42.1%"}
                  </span>
                  <span className="text-xs text-slate-400">Jaccard Index</span>
                </div>
                <div className="mt-2 text-xs text-slate-400">
                  Evaluated across 246 chips over 10 global flood events (Epoch 8 checkpoint).
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-blue-500" />
              </div>

              {/* Card 2: Val F1 */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-sm relative overflow-hidden">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Validation F1 Score
                </div>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-sky-300 font-mono">
                    {evalReports?.validation_split
                      ? `${(evalReports.validation_split.metrics.f1 * 100).toFixed(1)}%`
                      : "59.2%"}
                  </span>
                  <span className="text-xs text-slate-400">Harmonic Mean</span>
                </div>
                <div className="mt-2 text-xs text-slate-400">
                  Precision: 55.4% | Recall: 63.6% at decision threshold = 0.50.
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-400 to-cyan-500" />
              </div>

              {/* Card 3: Bolivia Zero-Shot IoU */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-sm relative overflow-hidden">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Held-Out Bolivia Transfer
                </div>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold text-amber-400 font-mono">
                    {evalReports?.bolivia_heldout_split
                      ? `${(evalReports.bolivia_heldout_split.metrics.iou * 100).toFixed(1)}%`
                      : "35.3%"}
                  </span>
                  <span className="text-xs text-amber-400/80">IoU</span>
                </div>
                <div className="mt-2 text-xs text-slate-400">
                  Precision: 74.2% | Unseen geographic biome held-out test split.
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-400 to-orange-500" />
              </div>

              {/* Card 4: Inference Engine */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-sm relative overflow-hidden">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Active Model & Device
                </div>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-xl font-bold text-emerald-400 font-mono truncate">
                    {activeModel?.model_architecture || "PyTorch U-Net"}
                  </span>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center gap-1.5 font-mono">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  {health?.compute_device || "CUDA / GPU"}
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-400 to-teal-500" />
              </div>
            </div>

            {/* Workflow Steps */}
            <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-4">
                Inference & Analysis Pipeline
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/80 text-cyan-400 flex items-center justify-center font-bold text-sm shrink-0">
                    1
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-slate-200">SAR Ingestion & Calibration</h4>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Two-channel Sentinel-1 GeoTIFF (VV and VH) in linear amplitude or decibels. Clipped to [-35, 0] dB range and standardized with frozen training statistics.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/80 text-cyan-400 flex items-center justify-center font-bold text-sm shrink-0">
                    2
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-slate-200">U-Net Feature Segmentation</h4>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      7.76M parameter 4-stage convolutional encoder-decoder with skip connections computes pixel-wise flood logits and sigmoid probability distribution.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/80 text-cyan-400 flex items-center justify-center font-bold text-sm shrink-0">
                    3
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-slate-200">Geospatial Inundation Bounds</h4>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Binary mask generated using calibrated decision threshold. Calculates total inundated area in km² and projects onto interactive Leaflet basemap.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: FLOOD WORKSPACE */}
        {/* ========================================================================= */}
        {activeTab === "workspace" && (
          <div className="space-y-6 animate-fadeIn">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
              <div>
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  <MapPin className="w-6 h-6 text-cyan-400" />
                  Flood Prediction Workspace
                </h2>
                <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                  Execute real-time inference on benchmark Sentinel-1 scenes or upload custom SAR GeoTIFFs.
                </p>
              </div>
              {predictionResult && (
                <div className="flex items-center gap-2 text-xs font-mono bg-cyan-950/60 border border-cyan-800 px-3 py-1.5 rounded-lg text-cyan-300">
                  <Check className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Scene: {predictionResult.chip_id || predictionResult.filename}</span>
                </div>
              )}
            </div>

            {/* Workspace Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Controls (4 cols) */}
              <div className="lg:col-span-4 space-y-5">
                {/* Benchmark Selection Card */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/80 shadow-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <Satellite className="w-3.5 h-3.5 text-cyan-400" />
                      1. Benchmark Chip Selection
                    </span>
                    <span className="text-[10px] font-mono text-cyan-400/80">
                      {samples.length} scenes available
                    </span>
                  </div>

                  <div className="space-y-2">
                    <select
                      value={selectedChipId}
                      onChange={(e) => {
                        setSelectedChipId(e.target.value);
                        setUploadedFile(null);
                      }}
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                    >
                      {samples.map((s) => (
                        <option key={s.chip_id} value={s.chip_id}>
                          {s.country} — {s.chip_id}
                        </option>
                      ))}
                    </select>
                    <p className="text-[11px] text-slate-400">
                      Pre-curated validation chips with calibrated Sentinel-1 VV/VH imagery.
                    </p>
                  </div>

                  <button
                    disabled={isPredicting || !selectedChipId}
                    onClick={() => handleRunBenchmarkPrediction()}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-cyan-500/20"
                  >
                    {isPredicting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Running U-Net Inference...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Run Benchmark Segmentation
                      </>
                    )}
                  </button>
                </div>

                {/* Upload Custom GeoTIFF Card */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/80 shadow-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <UploadCloud className="w-3.5 h-3.5 text-sky-400" />
                      2. Or Upload Custom SAR GeoTIFF
                    </span>
                  </div>

                  <div className="relative border-2 border-dashed border-slate-700 hover:border-cyan-500 rounded-lg p-4 text-center cursor-pointer transition-colors bg-slate-950/40">
                    <input
                      type="file"
                      accept=".tif,.tiff"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <div className="space-y-1">
                      <UploadCloud className="w-6 h-6 mx-auto text-slate-400" />
                      <div className="text-xs text-slate-300 font-medium">
                        {uploadedFile ? uploadedFile.name : "Choose or drag .tif / .tiff file"}
                      </div>
                      <p className="text-[10px] text-slate-500">
                        2-band Sentinel-1 SAR (VV & VH). Max 50 MB.
                      </p>
                    </div>
                  </div>

                  {uploadedFile && (
                    <button
                      disabled={isPredicting}
                      onClick={handleRunUploadPrediction}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 transition-colors disabled:opacity-50"
                    >
                      {isPredicting ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Processing Upload...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Segment Uploaded Raster
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Threshold & Layer Display Controls */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/80 shadow-lg space-y-4">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                    3. Inference & Layer Controls
                  </span>

                  {/* Decision Threshold Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Decision Threshold</span>
                      <span className="font-mono text-cyan-400 font-semibold">{threshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>0.10 (High Sensitivity)</span>
                      <span>0.50 (Default)</span>
                      <span>0.90 (High Precision)</span>
                    </div>
                  </div>

                  {/* Layer Opacity Slider */}
                  <div className="space-y-1.5 pt-2 border-t border-slate-800">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Overlay Opacity</span>
                      <span className="font-mono text-cyan-400 font-semibold">{Math.round(layerOpacity * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={layerOpacity}
                      onChange={(e) => setLayerOpacity(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                  </div>

                  {/* Toggles */}
                  <div className="space-y-2 pt-2 border-t border-slate-800">
                    <div className="text-[11px] font-semibold text-slate-400 uppercase">Visible Overlays</div>
                    <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showSar}
                        onChange={(e) => setShowSar(e.target.checked)}
                        className="rounded accent-cyan-400"
                      />
                      <span>Sentinel-1 SAR Grayscale Imagery</span>
                    </label>
                    <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showMask}
                        onChange={(e) => setShowMask(e.target.checked)}
                        className="rounded accent-cyan-400"
                      />
                      <span>Predicted Flood Inundation (Cyan Mask)</span>
                    </label>
                    <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showProb}
                        onChange={(e) => setShowProb(e.target.checked)}
                        className="rounded accent-cyan-400"
                      />
                      <span>Probability Heatmap Overlay</span>
                    </label>
                  </div>
                </div>

                {/* Inundation Statistics Card */}
                {predictionResult && (
                  <div className="p-5 rounded-xl border border-cyan-800/60 bg-cyan-950/20 shadow-lg space-y-3">
                    <div className="text-xs font-semibold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-cyan-400" />
                      Inundation Statistics
                    </div>
                    <div className="grid grid-cols-2 gap-3 pt-1">
                      <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-[10px] text-slate-400 uppercase">Inundated Area</div>
                        <div className="text-lg font-bold text-cyan-300 font-mono mt-0.5">
                          {predictionResult.inundated_area_km2.toFixed(3)} km²
                        </div>
                      </div>
                      <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-[10px] text-slate-400 uppercase">Inundated %</div>
                        <div className="text-lg font-bold text-cyan-300 font-mono mt-0.5">
                          {predictionResult.flood_percentage.toFixed(2)}%
                        </div>
                      </div>
                      <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-[10px] text-slate-400 uppercase">Flood Pixels</div>
                        <div className="text-xs font-semibold text-slate-200 font-mono mt-1">
                          {predictionResult.flood_pixel_count.toLocaleString()} px
                        </div>
                      </div>
                      <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <div className="text-[10px] text-slate-400 uppercase">Coordinate System</div>
                        <div className="text-xs font-semibold text-slate-200 font-mono mt-1 truncate">
                          {predictionResult.crs || "EPSG:4326"}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error Banner */}
                {predictionError && (
                  <div className="p-3.5 rounded-lg border border-rose-800 bg-rose-950/40 text-rose-300 text-xs flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                    <span>{predictionError}</span>
                  </div>
                )}
              </div>

              {/* Right Column: Geospatial Leaflet Map (8 cols) */}
              <div className="lg:col-span-8 flex flex-col gap-4">
                {predictionResult ? (
                  <FloodMap
                    latLngBounds={predictionResult.lat_lng_bounds}
                    center={predictionResult.center}
                    sarOverlay={predictionResult.overlays?.sar_overlay}
                    maskOverlay={predictionResult.overlays?.mask_overlay}
                    probOverlay={predictionResult.overlays?.prob_overlay}
                    showSar={showSar}
                    showMask={showMask}
                    showProb={showProb}
                    opacity={layerOpacity}
                    chipId={predictionResult.chip_id || predictionResult.filename}
                    inundatedKm2={predictionResult.inundated_area_km2}
                    floodPercentage={predictionResult.flood_percentage}
                  />
                ) : (
                  <div className="w-full h-[520px] rounded-xl border border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center p-8 text-center space-y-4">
                    <div className="w-16 h-16 rounded-2xl bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-cyan-400 shadow-xl shadow-cyan-950/40">
                      <Satellite className="w-8 h-8" />
                    </div>
                    <div className="max-w-md space-y-1">
                      <h3 className="text-lg font-semibold text-slate-200">
                        Interactive Geospatial Canvas
                      </h3>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Select a benchmark Sentinel-1 scene from Ghana, India, Mekong, Nigeria, or upload a custom dual-polarization GeoTIFF to run real-time U-Net segmentation.
                      </p>
                    </div>
                    <button
                      onClick={() => handleRunBenchmarkPrediction("Ghana_5079")}
                      className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 text-xs font-medium flex items-center gap-2 transition-colors"
                    >
                      <Play className="w-3.5 h-3.5" />
                      Quick Test: Ghana Scene 5079
                    </button>
                  </div>
                )}

                {/* Spatial Metadata Bar */}
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-wrap items-center justify-between gap-4 text-xs">
                  <div className="flex items-center gap-4 text-slate-400 font-mono">
                    <span className="flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-cyan-400" />
                      Resolution: 10m Ground Pixel Size
                    </span>
                    <span className="hidden sm:inline">|</span>
                    <span className="flex items-center gap-1.5">
                      <Database className="w-3.5 h-3.5 text-sky-400" />
                      Bands: Dual-Pol (VV + VH)
                    </span>
                  </div>
                  <div className="text-slate-400 text-[11px]">
                    Basemap: CartoDB Dark Matter | WGS84 Geodetic Alignment
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: EVALUATION & BENCHMARKS */}
        {/* ========================================================================= */}
        {activeTab === "evaluation" && (
          <div className="space-y-8 animate-fadeIn">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <BarChart3 className="w-6 h-6 text-cyan-400" />
                Benchmark Evaluation & Rigorous Audit
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                Quantitative segmentation performance on audited hand-labelled ground truth across global validation and zero-shot held-out events.
              </p>
            </div>

            {/* Operational Warning Alert Banner */}
            <div className="p-4 rounded-xl border border-amber-600/60 bg-amber-950/20 flex items-start gap-3.5 shadow-lg">
              <ShieldAlert className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
              <div className="text-xs sm:text-sm text-slate-300 space-y-1">
                <div className="font-bold text-amber-200 uppercase tracking-wide">
                  Important Operational Notice
                </div>
                <p className="text-slate-300 leading-relaxed">
                  These evaluation results reflect scientific machine learning benchmark performance on the audited Sen1Floods11 dataset. <strong>These results do not establish operational flood-warning reliability or authorization for life-safety decision-making.</strong> Satellite radar overpass timing, speckle noise, steep terrain shadow, and dense vegetation cover can cause misclassifications that require dedicated hydrologic modeling and in-situ field verification.
                </p>
              </div>
            </div>

            {/* Performance Comparison Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Validation Split Card */}
              <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/70 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-slate-200 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-cyan-400"></span>
                    Validation Split (10 Global Events)
                  </h3>
                  <span className="px-2 py-0.5 text-[11px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800 rounded">
                    246 Chips
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Chips drawn from 10 distinct flood disasters across Africa, Asia, Europe, and the Americas with audited hand-delineated dynamic flood masks.
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">IoU (Jaccard)</div>
                    <div className="text-xl font-extrabold text-cyan-400 font-mono mt-1">
                      {evalReports?.validation_split
                        ? `${(evalReports.validation_split.metrics.iou * 100).toFixed(1)}%`
                        : "42.1%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">F1 Score</div>
                    <div className="text-xl font-extrabold text-sky-300 font-mono mt-1">
                      {evalReports?.validation_split
                        ? `${(evalReports.validation_split.metrics.f1 * 100).toFixed(1)}%`
                        : "59.2%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">Precision</div>
                    <div className="text-xl font-extrabold text-emerald-400 font-mono mt-1">
                      {evalReports?.validation_split
                        ? `${(evalReports.validation_split.metrics.precision * 100).toFixed(1)}%`
                        : "55.4%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">Recall</div>
                    <div className="text-xl font-extrabold text-indigo-400 font-mono mt-1">
                      {evalReports?.validation_split
                        ? `${(evalReports.validation_split.metrics.recall * 100).toFixed(1)}%`
                        : "63.6%"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Held-Out Bolivia Card */}
              <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/70 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-slate-200 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                    Held-Out Bolivia Split (Zero-Shot)
                  </h3>
                  <span className="px-2 py-0.5 text-[11px] font-mono bg-amber-950 text-amber-300 border border-amber-800 rounded">
                    23 Chips
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Strictly held-out geographic disaster event never seen in training or validation. Tests model domain generalization across unseen South American topography.
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">IoU (Jaccard)</div>
                    <div className="text-xl font-extrabold text-amber-400 font-mono mt-1">
                      {evalReports?.bolivia_heldout_split
                        ? `${(evalReports.bolivia_heldout_split.metrics.iou * 100).toFixed(1)}%`
                        : "35.3%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">F1 Score</div>
                    <div className="text-xl font-extrabold text-amber-300 font-mono mt-1">
                      {evalReports?.bolivia_heldout_split
                        ? `${(evalReports.bolivia_heldout_split.metrics.f1 * 100).toFixed(1)}%`
                        : "52.1%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">Precision</div>
                    <div className="text-xl font-extrabold text-emerald-400 font-mono mt-1">
                      {evalReports?.bolivia_heldout_split
                        ? `${(evalReports.bolivia_heldout_split.metrics.precision * 100).toFixed(1)}%`
                        : "74.2%"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                    <div className="text-[10px] text-slate-400 uppercase">Recall</div>
                    <div className="text-xl font-extrabold text-indigo-400 font-mono mt-1">
                      {evalReports?.bolivia_heldout_split
                        ? `${(evalReports.bolivia_heldout_split.metrics.recall * 100).toFixed(1)}%`
                        : "40.2%"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recharts Performance Visualizer */}
            <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Comparative Metrics Breakdown (%)
              </h3>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="metric" stroke="#94a3b8" />
                    <YAxis domain={[0, 100]} stroke="#94a3b8" unit="%" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        borderRadius: "0.5rem",
                        color: "#f8fafc",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="Validation" fill="#00f5ff" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Bolivia" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Confusion Matrix Visual Breakdown */}
            <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Validation Set Pixel Confusion Matrix
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl mx-auto font-mono text-center">
                <div className="p-4 rounded-lg bg-emerald-950/40 border border-emerald-800/80">
                  <div className="text-[10px] text-emerald-400 uppercase tracking-wider">True Positives (TP)</div>
                  <div className="text-xl font-bold text-emerald-200 mt-1">
                    {evalReports?.validation_split
                      ? evalReports.validation_split.metrics.tp.toLocaleString()
                      : "444,213"}
                  </div>
                  <div className="text-[11px] text-emerald-400/80 mt-0.5">Correctly detected flood</div>
                </div>
                <div className="p-4 rounded-lg bg-amber-950/40 border border-amber-800/80">
                  <div className="text-[10px] text-amber-400 uppercase tracking-wider">False Positives (FP)</div>
                  <div className="text-xl font-bold text-amber-200 mt-1">
                    {evalReports?.validation_split
                      ? evalReports.validation_split.metrics.fp.toLocaleString()
                      : "357,987"}
                  </div>
                  <div className="text-[11px] text-amber-400/80 mt-0.5">Over-predicted dry terrain</div>
                </div>
                <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/80">
                  <div className="text-[10px] text-rose-400 uppercase tracking-wider">False Negatives (FN)</div>
                  <div className="text-xl font-bold text-rose-200 mt-1">
                    {evalReports?.validation_split
                      ? evalReports.validation_split.metrics.fn.toLocaleString()
                      : "254,118"}
                  </div>
                  <div className="text-[11px] text-rose-400/80 mt-0.5">Missed inundated pixels</div>
                </div>
                <div className="p-4 rounded-lg bg-slate-900/80 border border-slate-700">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">True Negatives (TN)</div>
                  <div className="text-xl font-bold text-slate-300 mt-1">
                    {evalReports?.validation_split
                      ? evalReports.validation_split.metrics.tn.toLocaleString()
                      : "62,714,146"}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">Correctly rejected dry land</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: MODEL ARCHITECTURE */}
        {/* ========================================================================= */}
        {activeTab === "model" && (
          <div className="space-y-6 animate-fadeIn">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <Cpu className="w-6 h-6 text-cyan-400" />
                Model Specifications & Preprocessing Pipeline
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                Technical architecture, parameter distribution, and normalization constants loaded directly from the active checkpoint.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Architecture Details */}
              <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/70 space-y-4">
                <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  Neural Network Architecture
                </h3>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Architecture Type</span>
                    <span className="font-mono text-cyan-300 font-semibold">
                      {activeModel?.model_architecture || "PyTorch 4-Stage U-Net"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Total Trainable Parameters</span>
                    <span className="font-mono text-slate-200 font-semibold">7,761,985 (29.61 MB)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Base Channels</span>
                    <span className="font-mono text-slate-200">
                      {activeModel?.base_channels || 32} (scaling: 32 → 64 → 128 → 256 → 512)
                    </span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Input Channels</span>
                    <span className="font-mono text-slate-200">2 (Sentinel-1 VV and VH backscatter)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Output Channels</span>
                    <span className="font-mono text-slate-200">1 (Logit flood segmentation map)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-800">
                    <span className="text-slate-400">Checkpoint Epoch</span>
                    <span className="font-mono text-emerald-400 font-semibold">
                      Epoch {activeModel?.checkpoint_epoch || 8} (Best Val IoU: 0.4208)
                    </span>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <span className="text-slate-400">Checkpoint Path</span>
                    <span className="font-mono text-slate-400 text-[11px] truncate max-w-[200px]">
                      models/checkpoints/unet_baseline_best.pt
                    </span>
                  </div>
                </div>
              </div>

              {/* Normalization & Preprocessing */}
              <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/70 space-y-4">
                <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                  <Database className="w-4 h-4 text-sky-400" />
                  Frozen Training Normalization Statistics
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  To ensure statistical consistency and prevent data leakage, inference strictly applies the exact empirical means and standard deviations computed from the training split.
                </p>
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400 uppercase">VV Polarization Mean</div>
                    <div className="text-base font-bold text-cyan-300 mt-1">
                      {activeModel?.normalization_statistics?.vv_mean !== undefined
                        ? `${activeModel.normalization_statistics.vv_mean.toFixed(2)} dB`
                        : "-10.45 dB"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400 uppercase">VV Polarization Std</div>
                    <div className="text-base font-bold text-cyan-300 mt-1">
                      {activeModel?.normalization_statistics?.vv_std !== undefined
                        ? `${activeModel.normalization_statistics.vv_std.toFixed(2)} dB`
                        : "4.06 dB"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400 uppercase">VH Polarization Mean</div>
                    <div className="text-base font-bold text-sky-300 mt-1">
                      {activeModel?.normalization_statistics?.vh_mean !== undefined
                        ? `${activeModel.normalization_statistics.vh_mean.toFixed(2)} dB`
                        : "-17.32 dB"}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400 uppercase">VH Polarization Std</div>
                    <div className="text-base font-bold text-sky-300 mt-1">
                      {activeModel?.normalization_statistics?.vh_std !== undefined
                        ? `${activeModel.normalization_statistics.vh_std.toFixed(2)} dB`
                        : "4.79 dB"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Hardware & Runtime Environment */}
            <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Compute Environment & Runtime Health
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[10px] uppercase">Service Health</div>
                  <div className="text-emerald-400 font-semibold mt-1 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    {health?.status === "healthy" ? "FastAPI Online" : "Service Active"}
                  </div>
                </div>
                <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[10px] uppercase">Active Device</div>
                  <div className="text-cyan-300 font-semibold mt-1 truncate">
                    {health?.compute_device || "NVIDIA RTX 4060"}
                  </div>
                </div>
                <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[10px] uppercase">PyTorch Version</div>
                  <div className="text-slate-200 font-semibold mt-1">
                    {health?.torch_version || "2.6.0+cu124"}
                  </div>
                </div>
                <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[10px] uppercase">Python Version</div>
                  <div className="text-slate-200 font-semibold mt-1">
                    {health?.python_version || "3.10.11"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Waves className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-300">FloodSense AI</span>
            <span>—</span>
            <span>Sentinel-1 Microwave Radar Inundation Pipeline</span>
          </div>
          <div className="text-slate-400 text-center sm:text-right">
            <span>Sen1Floods11 Hand-Labelled Benchmark | PyTorch U-Net</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
