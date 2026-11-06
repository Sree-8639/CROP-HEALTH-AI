"use client";

import React from "react";
import { SharedAOIProvider } from "@/context/SharedAOIContext";

export function AppProviders({ children }: { children: React.ReactNode }) {
  return <SharedAOIProvider>{children}</SharedAOIProvider>;
}

