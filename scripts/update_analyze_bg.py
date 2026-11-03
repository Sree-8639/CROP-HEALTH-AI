import re

target = r"crop-health-monitor\frontend\src\app\analyze\page.tsx"

with open(target, "r", encoding="utf-8") as f:
    text = f.read()

# Replace hero background with bg-paddy-aerial.jpg
text = text.replace("/bg-tea-landscape.jpg", "/bg-paddy-aerial.jpg")

# Replace header background with emerald styling
text = text.replace(
    'header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50"',
    'header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md"'
)

# Add full-page ambient aerial farmland backdrop if not present
old_div = '<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">'
new_div = '''<div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col relative">
      {/* Full-Page Ambient Aerial Farmland Backdrop */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none opacity-25 z-0"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      <div className="fixed inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/90 to-slate-950 pointer-events-none z-0" />'''

if old_div in text:
    text = text.replace(old_div, new_div)
    print("Added full-page ambient backdrop!")

# Ensure hero banner opacity is vibrant (opacity-55 or 60)
text = text.replace("opacity-45 scale-105", "opacity-60 scale-105")

# Ensure main container is z-10 relative
text = text.replace(
    '<main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 space-y-8">',
    '<main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 space-y-8 relative z-10">'
)

with open(target, "w", encoding="utf-8") as f:
    f.write(text)

print("Analyze page background script finished successfully!")

