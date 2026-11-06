import json
import urllib.request

bbox = [83.15, 17.62, 83.35, 17.80]
stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
query = {
    "collections": ["landsat-c2-l2"],
    "bbox": bbox,
    "datetime": "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z",
    "query": {"eo:cloud_cover": {"lt": 20}},
    "limit": 10,
}
req = urllib.request.Request(
    stac_url,
    data=json.dumps(query).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

features = data.get("features", [])
print(f"Found {len(features)} Landsat scenes on Planetary Computer:")
for feat in features[:6]:
    dt = feat["properties"]["datetime"]
    cc = feat["properties"]["eo:cloud_cover"]
    assets = list(feat["assets"].keys())
    print(f"Scene {feat['id']}: date={dt}, cloud={cc:.1f}%, assets={len(assets)}")
    # Find surface temperature asset
    for k in ["lwir11", "ST_B10", "surface_temperature", "thermal"]:
        if k in assets:
            print(f"   Thermal asset '{k}': {feat['assets'][k]['href'][:90]}...")
            break

