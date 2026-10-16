"""Download and cache the OSM pedestrian network for buffered Changwon."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/interim/network"
GRAPHML = CACHE / "changwon_walk_network.graphml"
NODES = CACHE / "changwon_walk_network_nodes.parquet"
EDGES = CACHE / "changwon_walk_network_edges.parquet"
META = CACHE / "changwon_walk_network_source.json"
BUFFER_M = 2000


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if GRAPHML.exists() and NODES.exists() and EDGES.exists() and META.exists():
        print(f"Using cached walk network: {GRAPHML}")
        return
    if GRAPHML.exists():
        graph = ox.io.load_graphml(GRAPHML)
    else:
        boundary = gpd.read_parquet(ROOT / "data/release/changwon_transit_demographic_context_by_dong.parquet")
        extent_metric = boundary.to_crs(5179).geometry.union_all().buffer(BUFFER_M)
        extent = gpd.GeoSeries([extent_metric], crs=5179).to_crs(4326).iloc[0]
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(ROOT / "data/raw/osm")
        ox.settings.requests_timeout = 300
        graph = ox.graph.graph_from_polygon(
            extent, network_type="walk", simplify=True, retain_all=True,
            truncate_by_edge=True,
        )
        graph = ox.projection.project_graph(graph, to_crs="EPSG:5179")
        ox.io.save_graphml(graph, GRAPHML)
    nodes, edges = ox.convert.graph_to_gdfs(graph)
    nodes = nodes.reset_index(); edges = edges.reset_index()
    for frame in (nodes, edges):
        for column in frame.select_dtypes(include="object"):
            if column != frame.geometry.name:
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, set, dict)) else value
                ).astype("string")
    nodes.to_parquet(NODES, index=False)
    edges.to_parquet(EDGES, index=False)
    metadata = {
        "source": "OpenStreetMap contributors via OSMnx/Overpass",
        "source_url": "https://www.openstreetmap.org/copyright",
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "network_type": "walk", "buffer_m": BUFFER_M,
        "network_crs": "EPSG:5179", "analysis_population": "2,926 official Changwon stops",
        "extent": "official Changwon administrative boundary union buffered by 2,000 metres",
        "osmnx_version": ox.__version__, "nodes": int(len(nodes)), "edges": int(len(edges)),
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cached walk network: nodes={len(nodes)}, edges={len(edges)}, graphml={GRAPHML}")


if __name__ == "__main__":
    main()
