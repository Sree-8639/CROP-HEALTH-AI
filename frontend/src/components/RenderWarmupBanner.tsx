"use client";

import React, { useState, useEffect } from "react";
import { API_BASE } from "@/config/api";

/**
 * RenderWarmupBanner
 * Keeps Render free-tier containers warm via periodic background pings (every 10 minutes),
 * and renders a polite, informative loading state when a cold start (>2.5s) is detected.
 */
export default function RenderWarmupBanner() {
  const [status, setStatus] = useState<"idle" | "warming" | "online" | "failed">("idle");
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  const pingBackend = async () => {
    const startTime = Date.now();
    let timer: NodeJS.Timeout | null = null;

    // After 2.5s of waiting, show the warming up indicator
    const warmupTimer = setTimeout(() => {
      setStatus("warming");
      timer = setInterval(() => {
        setElapsedSec((prev) => prev + 1);
      }, 1000);
    }, 2500);

    try {
      const res = await fetch(`${API_BASE}/api/health`, {
        signal: AbortSignal.timeout(60000), // 60s timeout for cold start
      });

      clearTimeout(warmupTimer);
      if (timer) clearInterval(timer);

      if (res.ok) {
        setStatus("online");
        // Auto-dismiss the online confirmation after 4s
        setTimeout(() => {
          setStatus("idle");
        }, 4000);
      } else {
        setStatus("failed");
      }
    } catch (err) {
      clearTimeout(warmupTimer);
      if (timer) clearInterval(timer);
      setStatus("failed");
    }
  };

  useEffect(() => {
    // Initial ping on app mount
    pingBackend();

    // Background keep-warm ping every 10 minutes
    const keepWarmInterval = setInterval(() => {
      fetch(`${API_BASE}/api/health`).catch(() => {});
    }, 600000);

    return () => {
      clearInterval(keepWarmInterval);
    };
  }, []);

  if (status === "idle") return null;

  if (status === "warming") {
    return (
      <div className="bg-amber-950/90 border-b border-amber-500/50 text-amber-200 px-4 py-2.5 text-xs flex items-center justify-between shadow-lg backdrop-blur-md sticky top-0 z-50 animate-in fade-in duration-300">
        <div className="flex items-center gap-2.5 max-w-4xl mx-auto w-full">
          <svg
            className="animate-spin h-4 w-4 text-amber-400 shrink-0"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-bold text-white">Waking up Render backend service...</span>
            <span className="text-amber-300">
              (Free tier spin-up typically takes ~30–45s; elapsed: {elapsedSec}s)
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (status === "online") {
    return (
      <div className="bg-emerald-950/80 border-b border-emerald-500/40 text-emerald-200 px-4 py-1.5 text-xs flex items-center justify-center gap-2 shadow-md backdrop-blur-md sticky top-0 z-50 animate-in fade-in duration-300">
        <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
        <span className="font-semibold text-emerald-300">Backend service connected &amp; warm</span>
      </div>
    );
  }

  return (
    <div className="bg-rose-950/90 border-b border-rose-500/50 text-rose-200 px-4 py-2 text-xs flex items-center justify-between shadow-lg backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center gap-2 max-w-4xl mx-auto w-full justify-between">
        <div className="flex items-center gap-2">
          <span>⚠️</span>
          <span>Backend service unreachable or cold start timed out.</span>
        </div>
        <button
          onClick={pingBackend}
          className="px-2.5 py-1 rounded-lg bg-rose-800 hover:bg-rose-700 text-white font-bold transition text-[11px]"
        >
          Retry Connection
        </button>
      </div>
    </div>
  );
}

