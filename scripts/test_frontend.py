import urllib.request

req = urllib.request.Request("http://localhost:3000", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode("utf-8")
    print("Frontend Status Code:", resp.status)
    print("HTML length:", len(html))
    print("Has bg-farm.jpg:", "bg-farm.jpg" in html)
    print("Has logo.png:", "logo.png" in html)
    print("Has favicon.ico:", "favicon.ico" in html)
    print("Has phase in HTML:", "phase" in html.lower())

analyze_req = urllib.request.Request("http://localhost:3000/analyze", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(analyze_req, timeout=10) as resp:
    a_html = resp.read().decode("utf-8")
    print("\nAnalyze Page Status Code:", resp.status)
    print("Analyze Page HTML length:", len(a_html))
    print("Has bg-tea-landscape.jpg:", "bg-tea-landscape.jpg" in a_html)
    print("Has phase in Analyze HTML:", "phase" in a_html.lower())

