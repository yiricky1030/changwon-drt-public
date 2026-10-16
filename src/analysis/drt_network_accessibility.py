"""Pedestrian-network accessibility features for official Changwon stops."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.strtree import STRtree


RADII = (300, 500, 800)


def load_walk_graph(path) -> nx.MultiGraph:
    directed = ox.io.load_graphml(path)
    for _, _, _, data in directed.edges(keys=True, data=True):
        data["length"] = float(data["length"])
    return ox.convert.to_undirected(directed)


def snap_stops(stops: gpd.GeoDataFrame, graph: nx.MultiGraph) -> pd.DataFrame:
    metric = stops.to_crs(5179).reset_index(drop=True)
    node_ids = np.array(list(graph.nodes), dtype=object)
    node_points = np.array([graph.nodes[node]["geometry"] if "geometry" in graph.nodes[node]
                            else None for node in node_ids], dtype=object)
    if any(point is None for point in node_points):
        from shapely.geometry import Point
        node_points = np.array([Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in node_ids])
    pairs, distances = STRtree(node_points).query_nearest(metric.geometry.to_numpy(), all_matches=False, return_distance=True)
    nearest = pairs[1]
    return pd.DataFrame({"stop_id": metric.stop_id.astype(str), "network_node_id": node_ids[nearest].astype(str),
                         "snap_distance_m": distances.astype(float)})


def _candidates(lengths: dict[Any, float], node_to_stops: dict[Any, list[int]],
                snap_distance: np.ndarray, source_index: int, limit: float | None = None) -> list[tuple[int, float]]:
    source_snap = snap_distance[source_index]
    result = []
    for node, graph_distance in lengths.items():
        for target in node_to_stops.get(node, []):
            if target == source_index:
                continue
            distance = source_snap + float(graph_distance) + snap_distance[target]
            if limit is None or distance <= limit:
                result.append((target, distance))
    return result


def build_accessibility(stops: gpd.GeoDataFrame, graph: nx.MultiGraph,
                        route_sets: dict[str, set[str]]) -> gpd.GeoDataFrame:
    """Use local/adaptive Dijkstra searches, never an all-pairs matrix."""
    frame = stops.reset_index(drop=True).copy()
    snapped = snap_stops(frame, graph)
    frame = frame.merge(snapped, on="stop_id", validate="one_to_one")
    node_lookup = {str(node): node for node in graph.nodes}
    snapped_nodes = np.array([node_lookup[value] for value in frame.network_node_id], dtype=object)
    snaps = frame.snap_distance_m.to_numpy(float)
    node_to_stops: dict[Any, list[int]] = defaultdict(list)
    for index, node in enumerate(snapped_nodes): node_to_stops[node].append(index)
    stop_ids = frame.stop_id.astype(str).to_numpy()
    observed = frame.has_observed_profile.to_numpy(bool)
    ready = frame.analysis_ready.to_numpy(bool)
    cache: dict[tuple[Any, float | None], dict[Any, float]] = {}

    def distances(node, cutoff):
        key = (node, cutoff)
        if key not in cache:
            cache[key] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight="length")
        return cache[key]

    rows = []
    for index, source_node in enumerate(snapped_nodes):
        local = _candidates(distances(source_node, 800), node_to_stops, snaps, index, 800)
        all_candidates = local
        cutoff = 800
        need = lambda candidates: (not candidates or not any(observed[j] for j, _ in candidates)
                                   or not any(ready[j] for j, _ in candidates))
        while need(all_candidates) and cutoff < 12800:
            cutoff *= 2
            all_candidates = _candidates(distances(source_node, cutoff), node_to_stops, snaps, index)
        if need(all_candidates):
            all_candidates = _candidates(distances(source_node, None), node_to_stops, snaps, index)

        def nearest(candidates, mask=None):
            selected = candidates if mask is None else [(j, d) for j, d in candidates if mask[j]]
            return min(selected, key=lambda item: (item[1], stop_ids[item[0]])) if selected else (None, np.nan)

        nearest_index, nearest_distance = nearest(all_candidates)
        _, observed_distance = nearest(all_candidates, observed)
        _, ready_distance = nearest(all_candidates, ready)
        radius_members = {radius: [j for j, distance in local if distance <= radius] for radius in RADII}
        routes500 = set().union(*(route_sets.get(stop_ids[j], set()) for j in radius_members[500])) if radius_members[500] else set()
        routes800 = set().union(*(route_sets.get(stop_ids[j], set()) for j in radius_members[800])) if radius_members[800] else set()
        rows.append({"stop_id": stop_ids[index],
            "nearest_stop_walk_stop_id": stop_ids[nearest_index] if nearest_index is not None else None,
            "nearest_stop_walk_distance_m": nearest_distance,
            "nearest_observed_stop_walk_distance_m": observed_distance,
            "nearest_analysis_ready_stop_walk_distance_m": ready_distance,
            "nearby_stops_walk_300m": len(radius_members[300]),
            "nearby_stops_walk_500m": len(radius_members[500]),
            "nearby_stops_walk_800m": len(radius_members[800]),
            "nearby_observed_stops_walk_500m": sum(observed[j] for j in radius_members[500]),
            "nearby_analysis_ready_stops_walk_500m": sum(ready[j] for j in radius_members[500]),
            "nearby_route_count_walk_500m": len(routes500),
            "nearby_route_count_walk_800m": len(routes800),
            "network_reachable_flag": bool(nearest_index is not None)})
    access = frame.merge(pd.DataFrame(rows), on="stop_id", validate="one_to_one")
    access["network_snap_flag"] = np.select(
        [access.snap_distance_m.le(100), access.snap_distance_m.le(200)],
        ["NETWORK_SNAP_OK", "NETWORK_SNAP_FAR"], default="NETWORK_SNAP_VERY_FAR")
    access["walk_detour_ratio"] = access.nearest_stop_walk_distance_m / access.nearest_stop_distance_m.where(access.nearest_stop_distance_m.gt(0))
    for raw, prefix in [("nearest_stop_walk_distance_m", "walk_isolation"),
                        ("nearby_stops_walk_500m", "local_stop_walk_supply"),
                        ("nearby_route_count_walk_500m", "local_route_walk_supply")]:
        access[f"{prefix}_pct_citywide"] = access[raw].rank(method="average", pct=True) * 100
        access[f"{prefix}_pct_within_dong"] = access.groupby("adm_cd")[raw].rank(method="average", pct=True) * 100
    for scope in ("citywide", "within_dong"):
        access[f"local_stop_walk_scarcity_pct_{scope}"] = 100 - access[f"local_stop_walk_supply_pct_{scope}"]
        access[f"local_route_walk_scarcity_pct_{scope}"] = 100 - access[f"local_route_walk_supply_pct_{scope}"]
    keep = ["stop_id", "stop_name", "adm_cd", "adm_nm", "anchor_data_status", "analysis_ready",
            "has_observed_profile", "network_node_id", "snap_distance_m", "nearest_stop_walk_stop_id",
            "nearest_stop_walk_distance_m", "nearest_observed_stop_walk_distance_m",
            "nearest_analysis_ready_stop_walk_distance_m", "nearby_stops_walk_300m", "nearby_stops_walk_500m",
            "nearby_stops_walk_800m", "nearby_observed_stops_walk_500m",
            "nearby_analysis_ready_stops_walk_500m", "nearby_route_count_walk_500m",
            "nearby_route_count_walk_800m", "walk_detour_ratio", "walk_isolation_pct_within_dong",
            "walk_isolation_pct_citywide", "local_stop_walk_scarcity_pct_within_dong",
            "local_stop_walk_scarcity_pct_citywide", "local_route_walk_scarcity_pct_within_dong",
            "local_route_walk_scarcity_pct_citywide", "network_snap_flag", "network_reachable_flag", "geometry"]
    return gpd.GeoDataFrame(access[keep], geometry="geometry", crs=stops.crs)


def validate_accessibility(access: gpd.GeoDataFrame, graph: nx.MultiGraph) -> None:
    assert len(access) == 2926 and access.stop_id.is_unique
    assert access.geometry.notna().all() and access.geometry.is_valid.all()
    assert access.snap_distance_m.ge(0).all()
    assert set(access.network_node_id).issubset({str(node) for node in graph.nodes})
    assert access.nearby_stops_walk_300m.le(access.nearby_stops_walk_500m).all()
    assert access.nearby_stops_walk_500m.le(access.nearby_stops_walk_800m).all()
    assert access.nearest_stop_walk_stop_id.dropna().ne(access.loc[access.nearest_stop_walk_stop_id.notna(), "stop_id"]).all()
    assert access.loc[~access.network_reachable_flag, "nearest_stop_walk_distance_m"].isna().all()
    percentile = [column for column in access if "_pct_" in column]
    assert access[percentile].min().min() >= 0 and access[percentile].max().max() <= 100


def network_qa(graph: nx.MultiGraph) -> dict[str, Any]:
    components = list(nx.connected_components(graph))
    edge_geometries = [data.get("geometry") for _, _, _, data in graph.edges(keys=True, data=True)]
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "components": len(components),
            "largest_component_nodes": max(map(len, components)),
            "largest_component_share": max(map(len, components)) / graph.number_of_nodes(),
            "isolated_components": sum(len(component) == 1 for component in components),
            "total_network_length_km": sum(float(data["length"]) for _, _, _, data in graph.edges(keys=True, data=True)) / 1000,
            "null_geometry": sum(geometry is None for geometry in edge_geometries),
            "invalid_geometry": sum(geometry is not None and not geometry.is_valid for geometry in edge_geometries)}
