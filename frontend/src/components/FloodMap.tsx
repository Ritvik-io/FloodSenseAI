"use client";

import React, { useEffect, useRef } from "react";
import L from "leaflet";

interface FloodMapProps {
  latLngBounds: [[number, number], [number, number]];
  center: [number, number];
  sarOverlay?: string;
  maskOverlay?: string;
  probOverlay?: string;
  showSar: boolean;
  showMask: boolean;
  showProb: boolean;
  opacity: number;
  chipId?: string;
  inundatedKm2?: number;
  floodPercentage?: number;
}

export default function FloodMap({
  latLngBounds,
  center,
  sarOverlay,
  maskOverlay,
  probOverlay,
  showSar,
  showMask,
  showProb,
  opacity,
  chipId,
  inundatedKm2,
  floodPercentage,
}: FloodMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const sarLayerRef = useRef<L.ImageOverlay | null>(null);
  const maskLayerRef = useRef<L.ImageOverlay | null>(null);
  const probLayerRef = useRef<L.ImageOverlay | null>(null);
  const rectLayerRef = useRef<L.Rectangle | null>(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }

    const map = L.map(mapContainerRef.current, {
      center: center || [0, 0],
      zoom: 12,
      zoomControl: true,
      attributionControl: true,
    });

    // OpenStreetMap Standard Basemap (no API key required)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      subdomains: 'abc',
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update bounds & overlays when prediction or layers change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !latLngBounds) return;

    // Bounds rectangle outline
    if (rectLayerRef.current) {
      rectLayerRef.current.remove();
    }
    const rect = L.rectangle(latLngBounds, {
      color: "#00F5FF",
      weight: 1.5,
      dashArray: "4, 4",
      fill: false,
    }).addTo(map);
    rectLayerRef.current = rect;

    // Remove existing image overlays
    if (sarLayerRef.current) {
      sarLayerRef.current.remove();
      sarLayerRef.current = null;
    }
    if (maskLayerRef.current) {
      maskLayerRef.current.remove();
      maskLayerRef.current = null;
    }
    if (probLayerRef.current) {
      probLayerRef.current.remove();
      probLayerRef.current = null;
    }

    // 1. SAR Grayscale Layer
    if (showSar && sarOverlay) {
      const sarLayer = L.imageOverlay(sarOverlay, latLngBounds, {
        opacity: 0.85,
        interactive: false,
      }).addTo(map);
      sarLayerRef.current = sarLayer;
    }

    // 2. Probability Heatmap Layer
    if (showProb && probOverlay) {
      const probLayer = L.imageOverlay(probOverlay, latLngBounds, {
        opacity: opacity,
        interactive: false,
      }).addTo(map);
      probLayerRef.current = probLayer;
    }

    // 3. Binary Flood Mask Layer
    if (showMask && maskOverlay) {
      const maskLayer = L.imageOverlay(maskOverlay, latLngBounds, {
        opacity: opacity,
        interactive: false,
      }).addTo(map);
      maskLayerRef.current = maskLayer;
    }

    // Fit map view to raster extent
    map.fitBounds(latLngBounds, { padding: [40, 40], maxZoom: 14 });
  }, [latLngBounds, sarOverlay, maskOverlay, probOverlay, showSar, showMask, showProb, opacity]);

  return (
    <div className="relative w-full h-[520px] rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950">
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Floating Geospatial Badge */}
      <div className="absolute top-4 left-4 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-lg px-3.5 py-2.5 shadow-lg text-xs space-y-1">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
          <span className="font-semibold text-slate-200">
            {chipId ? `Scene: ${chipId}` : "Custom GeoTIFF"}
          </span>
        </div>
        <div className="text-slate-400">
          Bounds: [{latLngBounds[0][0].toFixed(3)}°, {latLngBounds[0][1].toFixed(3)}°] to [
          {latLngBounds[1][0].toFixed(3)}°, {latLngBounds[1][1].toFixed(3)}°]
        </div>
        {inundatedKm2 !== undefined && (
          <div className="text-cyan-300 font-mono font-medium">
            Inundation: {inundatedKm2.toFixed(3)} km² ({floodPercentage?.toFixed(2)}%)
          </div>
        )}
      </div>

      {/* Map Legend */}
      <div className="absolute bottom-6 right-4 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-lg px-3.5 py-2.5 shadow-lg text-xs space-y-2">
        <div className="font-semibold text-slate-300 text-[11px] uppercase tracking-wider">
          Layer Legend
        </div>
        <div className="flex items-center gap-2.5">
          <span className="w-3.5 h-3.5 rounded bg-[#00F5FF] shadow-[0_0_8px_#00F5FF]"></span>
          <span className="text-slate-200 font-medium">Observed Flood Inundation</span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="w-3.5 h-3.5 rounded bg-slate-700 border border-slate-500"></span>
          <span className="text-slate-400">Dry Terrain / Land</span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="w-3.5 h-3.5 rounded border border-dashed border-cyan-400"></span>
          <span className="text-slate-400">Sentinel-1 Swath Extent</span>
        </div>
      </div>
    </div>
  );
}
