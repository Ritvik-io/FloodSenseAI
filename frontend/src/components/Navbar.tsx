"use client";

import React from "react";
import { Waves, Activity, MapPin, BarChart3, Cpu } from "lucide-react";

interface NavbarProps {
  activeTab: "overview" | "workspace" | "evaluation" | "model";
  setActiveTab: (tab: "overview" | "workspace" | "evaluation" | "model") => void;
  apiConnected: boolean;
  computeDevice?: string;
}

export default function Navbar({
  activeTab,
  setActiveTab,
  apiConnected,
  computeDevice,
}: NavbarProps) {
  const tabs = [
    { id: "overview", label: "Overview", icon: Activity },
    { id: "workspace", label: "Flood Workspace", icon: MapPin },
    { id: "evaluation", label: "Evaluation & Benchmarks", icon: BarChart3 },
    { id: "model", label: "Model Architecture", icon: Cpu },
  ] as const;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Waves className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                FloodSense AI
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase bg-cyan-950 text-cyan-300 border border-cyan-800/60 rounded-full">
                v1.0 SAR
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono hidden sm:block">
              Sentinel-1 Radar Inundation Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 text-xs sm:text-sm font-medium rounded-lg transition-all duration-150 ${
                  isActive
                    ? "bg-slate-800 text-cyan-300 border border-cyan-500/30 shadow-md shadow-cyan-950/50"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                <span className="hidden md:inline">{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Health / Hardware Pill */}
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono border ${
              apiConnected
                ? "bg-emerald-950/60 border-emerald-800/60 text-emerald-300"
                : "bg-rose-950/60 border-rose-800/60 text-rose-300 animate-pulse"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                apiConnected ? "bg-emerald-400 shadow-[0_0_6px_#34d399]" : "bg-rose-400"
              }`}
            />
            <span className="hidden lg:inline">
              {apiConnected ? computeDevice || "GPU Connected" : "Connecting..."}
            </span>
            <span className="lg:hidden">{apiConnected ? "Online" : "Offline"}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
