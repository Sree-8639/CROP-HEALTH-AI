target_farmer = r"crop-health-monitor\frontend\src\components\FarmerDashboard.tsx"
with open(target_farmer, "r", encoding="utf-8") as f:
    text = f.read()

# Update header
text = text.replace(
    'header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50"',
    'header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md"'
)

# Update hero image opacity and remove mix-blend-luminosity
text = text.replace(
    'opacity-30 mix-blend-luminosity scale-105',
    'opacity-55 scale-105'
)

# Add ambient backdrop
old_div = '<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-white">'
new_div = '''<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-white relative">
      {/* Ambient Aerial Backdrop */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none opacity-20 z-0"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      <div className="fixed inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/90 to-slate-950 pointer-events-none z-0" />'''

if old_div in text:
    text = text.replace(old_div, new_div)

# Ensure main is z-10
text = text.replace(
    '<main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 space-y-6 w-full">',
    '<main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 space-y-6 w-full relative z-10">'
)

with open(target_farmer, "w", encoding="utf-8") as f:
    f.write(text)

print("Farmer dashboard updated!")

target_history = r"crop-health-monitor\frontend\src\app\history\page.tsx"
with open(target_history, "r", encoding="utf-8") as f:
    htext = f.read()

htext = htext.replace(
    'header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50"',
    'header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md"'
)

old_hdiv = '<div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white">'
new_hdiv = '''<div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white relative">
      {/* Ambient Aerial Backdrop */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none opacity-20 z-0"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      <div className="fixed inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/90 to-slate-950 pointer-events-none z-0" />'''

if old_hdiv in htext:
    htext = htext.replace(old_hdiv, new_hdiv)

htext = htext.replace(
    '<main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">',
    '<main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8 relative z-10">'
)

with open(target_history, "w", encoding="utf-8") as f:
    f.write(htext)

print("History page updated!")

