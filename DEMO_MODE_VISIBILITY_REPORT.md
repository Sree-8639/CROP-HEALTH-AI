# Demo Mode Visibility & Provenance Disclosure Report

## Overview
Phase 20 guarantees that users are never misled about the origin or authenticity of satellite data and analytical outputs. The reusable component `GlobalDemoIndicator` is integrated across all primary pages of the platform.

---

## 1. Indicator States & Disclosure Rules

| Mode | Visual Presentation | Exact Disclosure Text | Condition |
|---|---|---|---|
| **DEMO MODE** | Amber banner with test-tube icon | `"DEMO MODE — Demonstration data is being used. Results do not represent live field analysis."` | Synthetic reflectance or mock demonstration datasets active (`is_demo == true`). |
| **LIVE SATELLITE DATA** | Emerald banner with pulsing dot | `"LIVE SATELLITE DATA • Source: Copernicus Sentinel-2 L2A • Acquired: [Date] • Cloud: [X]%"` | Live verified satellite scenes fetched from Copernicus Data Space. |
| **MIXED DATA** | Indigo banner with sync icon | `"MIXED DATA DISCLOSURE • Combines verified optical reflectance with synthetic demonstration SAR arrays."` | Optical observations are verified, but SAR data is supplemented via synthetic arrays. |
| **UNVERIFIED SOURCE** | Amber warning banner | `"DATA SOURCE UNVERIFIED • Observation provenance could not be validated against Copernicus catalog records."` | Missing catalog provenance or unconfirmed data source. |

---

## 2. Page-by-Page Integration Matrix

1. **Home (`/`)**:
   - Universal banner embedded at the top of the hero section.
   - Sample Analysis section (`#sample-analysis`) explicitly tagged: `"DEMO DATA · PRE-COMPUTED SAMPLE ANALYSIS"`.
2. **Analyze (`/analyze`)**:
   - Dynamic banner reflecting `satelliteImage?.is_demo` and Copernicus scene credentials.
   - Individual sub-panels (SAR, Stress, Growth, Risk Map) display secondary badges when operating in demonstrator mode.
3. **Farmer Dashboard (`/farmer-dashboard`)**:
   - Top banner updates dynamically based on loaded field overview (`dashboardData.field_overview.is_demo`).
   - Clear disclosure in field comparison and recommendation modules.
4. **Multi-Temporal History (`/history`)**:
   - Header banner identifies whether the time-series is synthetic or verified Copernicus archive.

---

## 3. Security & Credential Protection
- `GlobalDemoIndicator` never discloses API keys, OAuth tokens, internal URLs, or S3/cloud storage credentials.
- All disclosures focus strictly on scientific provenance, acquisition timestamps, and processing levels.
