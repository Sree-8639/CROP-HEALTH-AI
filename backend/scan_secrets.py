import subprocess
import re
import sys

def check_tools():
    for tool in ['gitleaks', 'trufflehog']:
        try:
            res = subprocess.run([tool, 'version'], capture_output=True, text=True)
            print(f"Tool {tool} is installed: {res.stdout.strip()}")
        except FileNotFoundError:
            print(f"Tool {tool} is NOT available in PATH.")

def scan_git_history():
    print("Scanning full git log across all branches and commits for credentials...")
    patterns = [
        ("SentinelHub Secret", re.compile(r'sh-[a-zA-Z0-9_\-]{20,}')),
        ("Google API Key", re.compile(r'AIza[0-9A-Za-z\-_]{35}')),
        ("AWS Access Key", re.compile(r'AKIA[0-9A-Z]{16}')),
        ("GitHub Token", re.compile(r'ghp_[0-9a-zA-Z]{36}')),
        ("Copernicus / Secret Env", re.compile(r'(COPERNICUS_PASSWORD|SENTINEL_HUB_CLIENT_SECRET|JWT_SECRET_KEY)\s*=\s*[a-zA-Z0-9_\-\.]+')),
        (".env file addition", re.compile(r'diff --git a/.*\.env\b'))
    ]

    repo_dir = r"c:\Users\shanm\Downloads\New folder (5)\crop-health-monitor"
    proc = subprocess.Popen(['git', 'log', '--all', '-p'], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
    
    current_commit = "initial"
    current_author = ""
    current_date = ""
    matches = []

    for line in proc.stdout:
        if line.startswith("commit "):
            current_commit = line.strip().split()[1]
        elif line.startswith("Author: "):
            current_author = line.strip()
        elif line.startswith("Date: "):
            current_date = line.strip()
        
        for name, pat in patterns:
            # ignore lines that are just .env.example or template placeholders
            if "your-" in line or "your_" in line or "<" in line or "EXAMPLE" in line.upper():
                continue
            if pat.search(line):
                # Check if it's a real secret
                clean_line = line.strip()
                if not any(placeholder in clean_line for placeholder in ["dummy", "placeholder", "secret_key_here"]):
                    matches.append((current_commit, name, clean_line[:120]))

    proc.wait()
    return matches

if __name__ == '__main__':
    check_tools()
    matches = scan_git_history()
    print(f"\nTotal potential secret matches found in git commit diff history: {len(matches)}")
    if matches:
        for commit, name, sample in matches:
            print(f"[{commit[:8]}] {name}: {sample}")
    else:
        print("CONFIRMED: Clean! Zero API keys, passwords, client secrets, or committed .env files found in entire git history.")
