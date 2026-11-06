"""
experiments/utils.py:
Shared helper utilities for reproducible experiments:
- Zero-dependency DataFrame to Markdown table converter
- Zero-dependency DataFrame to LaTeX table converter
- Seed synchronization across random, numpy, torch
- Metric JSON synchronization to backend and frontend
"""

import os
import json
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def set_all_seeds(seed: int = 42):
    """Pins all pseudo-random number generator seeds for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame into a GitHub Flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[h]).replace("\n", " ").replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def dataframe_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Converts a pandas DataFrame into a publication-ready LaTeX tabular environment."""
    headers = [str(c) for c in df.columns]
    col_align = "l" * len(headers)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\hline",
        " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = [str(row[h]).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&") for h in headers]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def sync_metrics_json(filename: str, data: dict):
    """Synchronizes metric JSON outputs across root results/, backend/results/, and frontend/public/."""
    targets = [
        os.path.join(BASE_DIR, "results", filename),
        os.path.join(BASE_DIR, "backend", "results", filename),
        os.path.join(BASE_DIR, "frontend", "public", filename),
    ]
    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    print(f"  Synchronized {filename} to root, backend, and frontend.")

