import re
import os

target_path = r"c:\Users\shanm\Downloads\New folder (5)\crop-health-monitor\frontend\src\app\analyze\page.tsx"

with open(target_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update handleSearch to support customQuery, retain search term, and automatically configure AOI
old_handle_search = """  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const cleanQuery = query.trim();

    if (!cleanQuery) {
      setErrorMessage("Please enter a location name (city, village, district, or place).");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setConfirmedAOI(null);
    setSatelliteImage(null);
    setSatelliteError(null);
    setIndicesResult(null);
    setIndicesError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/geocode?query=${encodeURIComponent(cleanQuery)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Location not found. Please try another location.");
      }

      setCurrentLocation({
        query: cleanQuery,
        display_name: data.display_name,
        latitude: data.latitude,
        longitude: data.longitude,
      });

      // Clear search input on success
      setQuery("");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to search location.");
    } finally {
      setIsLoading(false);
    }
  };"""

new_handle_search = """  const handleSearch = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const cleanQuery = (customQuery !== undefined ? customQuery : query).trim();

    if (!cleanQuery) {
      setErrorMessage("Please enter a location name (city, village, district, or place).");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setSatelliteImage(null);
    setSatelliteError(null);
    setIndicesResult(null);
    setIndicesError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/geocode?query=${encodeURIComponent(cleanQuery)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Location not found. Please try another location.");
      }

      setCurrentLocation({
        query: cleanQuery,
        display_name: data.display_name,
        latitude: data.latitude,
        longitude: data.longitude,
      });

      // Keep user's search query visible so they clearly see what was found
      setQuery(cleanQuery);

      // Automatically configure the field AOI so satellite acquisition is immediately active
      try {
        const aoiResponse = await fetch(`${API_BASE_URL}/api/v1/aoi`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: data.latitude,
            longitude: data.longitude,
            radius_km: radiusKm,
          }),
        });

        if (aoiResponse.ok) {
          const aoiData = await aoiResponse.json();
          setConfirmedAOI(aoiData.aoi);
        }
      } catch (aoiErr) {
        console.warn("Auto-AOI setup warning", aoiErr);
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to search location.");
    } finally {
      setIsLoading(false);
    }
  };"""

# 2. Hero and Location Search Box enhancement with background banner and quick-select buttons
old_hero_and_search = """        {/* Title */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Agricultural Field Analysis
          </h1>
          <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
            Locate your field, acquire Sentinel-2 optical imagery, and analyze spectral vegetation indicators: NDVI, NDMI, NDRE, and EVI.
          </p>
        </div>

        {/* 1. Location Search Box */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <label htmlFor="location-input" className="block text-sm font-semibold text-slate-200">
            Field Location Search
          </label>
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
            <input
              id="location-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter city, village, district, state, or place (e.g., Tadepalligudem, Andhra Pradesh)..."
              className="flex-1 px-4 py-3 bg-slate-950 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition text-sm sm:text-base"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 text-slate-950 font-bold rounded-xl transition shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>Search</span>
                </>
              )}
            </button>
          </form>

          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Notice</strong>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}
        </div>"""

new_hero_and_search = """        {/* Scenic Agricultural Hero Banner */}
        <div className="relative rounded-3xl overflow-hidden border border-emerald-500/30 shadow-2xl p-6 sm:p-10 bg-slate-900">
          <div
            className="absolute inset-0 bg-cover bg-center opacity-35 mix-blend-luminosity scale-105 transition-transform duration-1000"
            style={{ backgroundImage: "url('/bg-tea-landscape.jpg')" }}
          />
          <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-slate-950/85 to-transparent" />
          <div className="relative z-10 max-w-2xl space-y-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              🌾 Precision Satellite Agricultural Intelligence
            </span>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Agricultural Field Analysis
            </h1>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Locate any farm worldwide, acquire Sentinel-1 SAR &amp; Sentinel-2 optical observations, and run real-time biophysical crop health analytics.
            </p>
          </div>
        </div>

        {/* 1. Location Search Box & Quick Agricultural Presets */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 backdrop-blur-md">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <label htmlFor="location-input" className="block text-sm font-semibold text-slate-200">
              Field Location Search &amp; Geocoding
            </label>
            <span className="text-xs text-slate-400">
              Instant coordinates, gazetteer lookup &amp; auto-AOI centering
            </span>
          </div>
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
            <input
              id="location-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter place, city, village or coordinates (e.g., Tadepalligudem, Bhimavaram, Vijayawada, Guntur)..."
              className="flex-1 px-4 py-3 bg-slate-950 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition text-sm sm:text-base"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 text-slate-950 font-bold rounded-xl transition shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Locating Field...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>Search Place</span>
                </>
              )}
            </button>
          </form>

          {/* Quick Agricultural Hotspots */}
          <div className="pt-1">
            <span className="text-xs text-slate-400 block mb-2 font-medium">Quick Agricultural Presets:</span>
            <div className="flex flex-wrap gap-2">
              {[
                { label: "🌾 Tadepalligudem, AP", query: "Tadepalligudem, Andhra Pradesh, India" },
                { label: "🌾 Bhimavaram, AP", query: "Bhimavaram, Andhra Pradesh, India" },
                { label: "🌾 Vijayawada, AP", query: "Vijayawada, Andhra Pradesh, India" },
                { label: "🌶️ Guntur, AP", query: "Guntur, Andhra Pradesh, India" },
                { label: "🌾 Ludhiana, Punjab", query: "Ludhiana, Punjab, India" },
                { label: "🍇 Champagne, France", query: "Champagne, France" },
                { label: "🌽 Iowa Corn Belt, USA", query: "Des Moines, Iowa, USA" },
              ].map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => {
                    setQuery(preset.query);
                    handleSearch(undefined, preset.query);
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-emerald-950 hover:text-emerald-300 hover:border-emerald-500/50 text-slate-300 border border-slate-700/80 transition"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Location Active Status Card */}
          {currentLocation && (
            <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-xs text-emerald-200 flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-base text-emerald-400">📍</span>
                <div>
                  <strong className="text-emerald-300 font-semibold text-sm block">
                    {currentLocation.display_name}
                  </strong>
                  <span className="text-slate-400 font-mono text-[11px]">
                    Exact Coordinates: {currentLocation.latitude.toFixed(4)}°N, {currentLocation.longitude.toFixed(4)}°E • Field Radius: {radiusKm} km
                  </span>
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold text-[11px] flex items-center gap-1">
                ✓ Field AOI Locked &amp; Ready
              </span>
            </div>
          )}

          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Location Notice</strong>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}
        </div>"""

# Replace in content (handling newline variations)
def do_replace(src, old, new):
    old_crlf = old.replace("\r\n", "\n").replace("\n", "\r\n")
    old_lf = old.replace("\r\n", "\n")
    if old_crlf in src:
        return src.replace(old_crlf, new.replace("\r\n", "\n").replace("\n", "\r\n"))
    elif old_lf in src:
        return src.replace(old_lf, new.replace("\r\n", "\n"))
    else:
        print("COULD NOT FIND BLOCK:")
        print(old[:100])
        return src

content = do_replace(content, old_handle_search, new_handle_search)
content = do_replace(content, old_hero_and_search, new_hero_and_search)

# Clean all Phase strings in analyze/page.tsx
phase_replacements = [
    ("Phase 15: Explainable AI & Uncertainty Active", "Explainable AI & Uncertainty Active"),
    ('<span className="text-[11px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">\n                Phase 12\n              </span>', '<span className="text-[11px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">\n                Multimodal Fusion\n              </span>'),
    ('<span className="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">\n                Phase 14\n              </span>', '<span className="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">\n                Growth Forecast\n              </span>'),
    ("Phase 11 C-Band SAR Pipeline", "Sentinel-1 C-Band SAR Pipeline"),
    ("Phase 12 Multimodal Feature-Fusion Pipeline", "Multimodal Feature-Fusion Pipeline"),
    ("Phase 12 Pipeline Protocol & Machine Learning Boundary", "Multimodal Pipeline Protocol & Machine Learning Boundary"),
    ("This phase completes the mathematical and geospatial fusion pipeline", "This pipeline completes the mathematical and geospatial fusion pipeline"),
    ("Phase 13 AI Stress Intelligence", "AI Early Crop Stress Intelligence"),
    ("Phase 14 Multimodal AI Growth Engine", "Multimodal AI Growth Engine"),
    ("Phase 16 Spatial Intelligence Layer", "Field-Level Spatial Intelligence Layer"),
    ("Phase 5 Multispectral Spectral Pipeline", "Sentinel-2 Multispectral Pipeline"),
    ("Phase 6 Intelligence Layer", "Crop Condition Intelligence Layer")
]

for old_s, new_s in phase_replacements:
    # try direct replace
    if old_s in content:
        content = content.replace(old_s, new_s)
    else:
        # try CRLF / LF conversion
        old_s_crlf = old_s.replace("\r\n", "\n").replace("\n", "\r\n")
        new_s_crlf = new_s.replace("\r\n", "\n").replace("\n", "\r\n")
        if old_s_crlf in content:
            content = content.replace(old_s_crlf, new_s_crlf)
        else:
            old_s_lf = old_s.replace("\r\n", "\n")
            new_s_lf = new_s.replace("\r\n", "\n")
            if old_s_lf in content:
                content = content.replace(old_s_lf, new_s_lf)
            else:
                print("Could not find string:", repr(old_s[:40]))

with open(target_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Analyze page updated successfully!")

