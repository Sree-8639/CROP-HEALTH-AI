target = r"crop-health-monitor\frontend\src\app\analyze\page.tsx"

with open(target, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update top-level background to be full-page aerial paddy field with translucent overlay
old_top = '''<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col relative">
      {/* Full-Page Ambient Aerial Farmland Backdrop */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none opacity-25 z-0"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      <div className="fixed inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/90 to-slate-950 pointer-events-none z-0" />'''

new_top = '''<div className="min-h-screen text-slate-100 flex flex-col relative overflow-x-hidden">
      {/* Full-Page High-Definition Aerial Farmland Background */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none -z-10"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      {/* Ambient glassmorphic tint so the green fields are vibrant while all UI components remain clear */}
      <div className="fixed inset-0 bg-gradient-to-b from-black/45 via-slate-950/60 to-black/75 pointer-events-none -z-10 backdrop-blur-[1px]" />'''

if old_top in text:
    text = text.replace(old_top, new_top)
    print("Updated top-level full-page background!")
else:
    print("Could not match old_top directly, trying line-by-line replacement")
    # Replace the backdrop divs
    text = text.replace(
        '<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col relative">',
        '<div className="min-h-screen text-slate-100 flex flex-col relative overflow-x-hidden">'
    )
    text = text.replace('opacity-25 z-0', 'opacity-100 -z-10')
    text = text.replace(
        'className="fixed inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/90 to-slate-950 pointer-events-none z-0"',
        'className="fixed inset-0 bg-gradient-to-b from-black/45 via-slate-950/60 to-black/75 pointer-events-none -z-10 backdrop-blur-[1px]"'
    )

# 2. Remove the isolated banner box and replace with floating clean title
old_banner = '''        {/* Scenic Agricultural Hero Banner */}
        <div className="relative rounded-3xl overflow-hidden border border-emerald-500/30 shadow-2xl p-6 sm:p-10 bg-slate-900">
          <div
            className="absolute inset-0 bg-cover bg-center opacity-60 scale-105 transition-transform duration-1000"
            style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
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
        </div>'''

new_banner = '''        {/* Page Title Header (Floating above the full-page background) */}
        <div className="text-center space-y-3 pt-2 pb-1">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950/85 text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-lg">
            <span>🌾</span>
            <span>Multimodal Satellite Agricultural Intelligence</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.95)]">
            Agricultural Field Analysis
          </h1>
          <p className="text-slate-200 text-sm sm:text-base max-w-2xl mx-auto drop-shadow-[0_2px_8px_rgba(0,0,0,0.95)] font-medium leading-relaxed">
            Locate any farm worldwide, acquire Sentinel-1 SAR &amp; Sentinel-2 optical observations, and run real-time biophysical crop health analytics.
          </p>
        </div>'''

if old_banner in text:
    text = text.replace(old_banner, new_banner)
    print("Replaced isolated banner box with clean floating header!")
else:
    print("Could not find old_banner directly, checking variation...")

# 3. Ensure panels use frosted glass styling
text = text.replace(
    'className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 backdrop-blur-md"',
    'className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-4 backdrop-blur-md"'
)
text = text.replace(
    'className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4"',
    'className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-4 backdrop-blur-md"'
)
text = text.replace(
    'className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6"',
    'className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-6 backdrop-blur-md"'
)

with open(target, "w", encoding="utf-8") as f:
    f.write(text)

print("Saved updated analyze/page.tsx!")

