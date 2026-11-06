import urllib.request
import re

def inspect_paper(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8')
            t_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
            title = t_match.group(1).strip() if t_match else 'N/A'
            # arXiv specific meta tags
            meta_title = re.search(r'<meta name="citation_title" content="(.*?)"', html)
            meta_authors = re.findall(r'<meta name="citation_author" content="(.*?)"', html)
            
            print(f"URL: {url}")
            print(f"HTML Title: {title}")
            if meta_title:
                print(f"Citation Title: {meta_title.group(1)}")
            if meta_authors:
                print(f"Authors: {', '.join(meta_authors)}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")

print("================================================================================")
print("VERIFYING PAPER 1: arXiv:2407.13596 (EarthMarker)")
print("================================================================================")
inspect_paper("https://arxiv.org/abs/2407.13596")

print("\n================================================================================")
print("VERIFYING PAPER 2: CropHarvest (NeurIPS Proceedings & arXiv:2102.08860)")
print("================================================================================")
inspect_paper("https://arxiv.org/abs/2102.08860")
inspect_paper("https://proceedings.neurips.cc/paper/2021/hash/2a4e286e74b5c7776b975e5330379201-Abstract.html")

