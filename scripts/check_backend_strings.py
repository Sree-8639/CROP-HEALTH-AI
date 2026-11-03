import glob, os

files = glob.glob(r"crop-health-monitor\backend\services\*.py") + [r"crop-health-monitor\backend\main.py"]

for fpath in files:
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if "phase" in line_clean.lower() and not line_clean.startswith("#"):
            if '"' in line_clean or "'" in line_clean:
                print(f"{os.path.basename(fpath)}:{idx+1}: {line_clean[:120]}")

