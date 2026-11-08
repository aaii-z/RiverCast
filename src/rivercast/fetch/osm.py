from rivercast.fetch.http import request_json

MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
URBAN_LANDUSE = ("residential", "commercial", "industrial", "retail")


def overpass(query: str) -> dict:
    error = None
    for url in MIRRORS:
        try:
            return request_json("POST", url, data={"data": query}, retries=4)
        except RuntimeError as e:
            error = e
    raise error


def _bbox(bbox) -> str:
    return "({},{},{},{})".format(*bbox)


def river_ways(names: list[str], bbox) -> list[dict]:
    pattern = "|".join(names)
    q = (f'[out:json][timeout:120];way["waterway"="river"]["name"~"^({pattern})$"]'
         f"{_bbox(bbox)};out geom;")
    return [{"name": e["tags"]["name"], "coords": [(p["lon"], p["lat"]) for p in e["geometry"]]}
            for e in overpass(q)["elements"]]


def wastewater_plants(bbox) -> list[dict]:
    q = f'[out:json][timeout:120];nwr["man_made"="wastewater_plant"]{_bbox(bbox)};out center tags;'
    plants = []
    for e in overpass(q)["elements"]:
        c = e.get("center", e)
        plants.append({"name": e.get("tags", {}).get("name", ""), "lon": c["lon"], "lat": c["lat"]})
    return plants


def urban_areas(bbox) -> list[list[tuple[float, float]]]:
    """Closed landuse ways (outer rings only) for built-up land."""
    pattern = "|".join(URBAN_LANDUSE)
    q = f'[out:json][timeout:180];way["landuse"~"^({pattern})$"]{_bbox(bbox)};out geom;'
    rings = [[(p["lon"], p["lat"]) for p in e["geometry"]] for e in overpass(q)["elements"]]
    return [r for r in rings if len(r) >= 4 and r[0] == r[-1]]
