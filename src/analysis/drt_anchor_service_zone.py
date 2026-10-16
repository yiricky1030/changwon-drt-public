"""Experimental walk-network access zones around robust-clean physical Anchors."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


DISTANCES = (400, 800, 1200)
BUFFER_M = 20.0
SIMPLIFY_M = 5.0
SOURCE = "OpenStreetMap via OSMnx; cached walk network"
METHOD = "network cutoff reachable edge/node buffer union"


def physical_anchors(demand: pd.DataFrame, supply: pd.DataFrame, access: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return one physical row per robust-clean stop without changing ranks or confidence."""
    dmask = demand.robust_clean & demand.get("anchor_eligible", True)
    smask = supply.robust_clean & supply.get("anchor_eligible", True)
    d = demand.loc[dmask, ["stop_id"]].assign(scenario_demand=True)
    s = supply.loc[smask, ["stop_id"]].assign(scenario_supply=True)
    flags = d.merge(s, on="stop_id", how="outer").fillna(False)
    flags[["scenario_demand", "scenario_supply"]] = flags[["scenario_demand", "scenario_supply"]].astype(bool)
    columns = ["stop_id", "stop_name", "adm_cd", "adm_nm", "network_node_id", "snap_distance_m", "geometry"]
    frame = access[columns].merge(flags, on="stop_id", validate="one_to_one")
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=access.crs)


def _edge_geometry(graph: nx.MultiGraph, u: Any, v: Any, data: dict) -> LineString:
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry
    return LineString([(float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                       (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))])


def _polygonize(graph: nx.MultiGraph, members: set[Any]) -> tuple[Any, int]:
    edges = []
    for u, v, _, data in graph.edges(keys=True, data=True):
        if u in members and v in members:
            edges.append(_edge_geometry(graph, u, v, data))
    nodes = [Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in members]
    pieces = [geometry.buffer(BUFFER_M, cap_style="round", join_style="round") for geometry in edges]
    pieces.extend(point.buffer(BUFFER_M) for point in nodes)
    geometry = unary_union(pieces).buffer(0).simplify(SIMPLIFY_M, preserve_topology=True)
    return geometry, len(edges)


def _node_membership(frame: pd.DataFrame) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(frame.network_node_id.astype(str)):
        result[node].append(index)
    return result


def build_service_zones(graph: nx.MultiGraph, anchors: gpd.GeoDataFrame,
                        stops: gpd.GeoDataFrame, route_bridge: pd.DataFrame,
                        poi: pd.DataFrame) -> gpd.GeoDataFrame:
    node_lookup = {str(node): node for node in graph.nodes}
    stop_frame = stops.reset_index(drop=True).copy()
    poi_frame = poi.loc[poi.network_reachable_flag & poi.network_node_id.notna()].reset_index(drop=True).copy()
    stop_nodes = _node_membership(stop_frame)
    poi_nodes = _node_membership(poi_frame)
    route_sets = (route_bridge.loc[route_bridge.is_matched & route_bridge.stop_id.notna()]
                  .assign(stop_id=lambda x: x.stop_id.astype(str))
                  .groupby("stop_id").route_id.agg(lambda values: set(values)).to_dict())
    rows = []
    for anchor in anchors.itertuples(index=False):
        source = node_lookup[str(anchor.network_node_id)]
        source_snap = float(anchor.snap_distance_m)
        graph_cutoff = max(DISTANCES) - source_snap
        lengths = nx.single_source_dijkstra_path_length(graph, source, cutoff=max(0, graph_cutoff), weight="length")
        for distance in DISTANCES:
            usable = max(0.0, distance - source_snap)
            members = {node for node, value in lengths.items() if value <= usable}
            geometry, edge_count = _polygonize(graph, members)
            stop_indices = set()
            for node, graph_distance in lengths.items():
                if graph_distance > usable:
                    continue
                for index in stop_nodes.get(str(node), []):
                    total = 0.0 if stop_frame.iloc[index].stop_id == anchor.stop_id else source_snap + graph_distance + float(stop_frame.iloc[index].snap_distance_m)
                    if total <= distance:
                        stop_indices.add(index)
            poi_indices = set()
            for node, graph_distance in lengths.items():
                if graph_distance > usable:
                    continue
                for index in poi_nodes.get(str(node), []):
                    if source_snap + graph_distance + float(poi_frame.iloc[index].network_snap_distance_m) <= distance:
                        poi_indices.add(index)
            stop_context = stop_frame.iloc[sorted(stop_indices)]
            poi_context = poi_frame.iloc[sorted(poi_indices)]
            routes = set().union(*(route_sets.get(str(stop_id), set()) for stop_id in stop_context.stop_id)) if len(stop_context) else set()
            categories = poi_context.poi_category.value_counts()
            rows.append({
                "stop_id": str(anchor.stop_id), "stop_name": anchor.stop_name, "adm_cd": str(anchor.adm_cd),
                "adm_nm": anchor.adm_nm, "distance_m": distance,
                "scenario_demand": bool(anchor.scenario_demand), "scenario_supply": bool(anchor.scenario_supply),
                "network_node_id": str(anchor.network_node_id), "reachable_node_count": len(members),
                "reachable_edge_count": edge_count, "polygon_area_m2": float(geometry.area),
                "official_stop_count": int(len(stop_context)),
                "observed_stop_count": int(stop_context.has_observed_profile.sum()),
                "analysis_ready_stop_count": int(stop_context.analysis_ready.sum()),
                "distinct_route_count": int(len(routes)),
                "healthcare_count": int(categories.get("HEALTHCARE", 0)),
                "administration_count": int(categories.get("ADMINISTRATION", 0)),
                "source": SOURCE, "method": METHOD,
                "status": "OK" if not geometry.is_empty else "UNREACHABLE", "geometry": geometry,
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=5179).to_crs(4326)


def validate_service_zones(zones: gpd.GeoDataFrame) -> None:
    assert zones.stop_id.nunique() * len(DISTANCES) == len(zones)
    assert set(zones.distance_m) == set(DISTANCES)
    assert not zones.duplicated(["stop_id", "distance_m"]).any()
    assert zones.geometry.notna().all() and zones.geometry.is_valid.all() and (~zones.geometry.is_empty).all()
    assert zones.to_crs(5179).geometry.area.gt(0).all()
    counts = [column for column in zones if column.endswith("_count")]
    assert zones[counts].ge(0).all().all()
    for _, group in zones.sort_values("distance_m").groupby("stop_id"):
        assert group.official_stop_count.is_monotonic_increasing
        assert group.observed_stop_count.is_monotonic_increasing
        assert group.analysis_ready_stop_count.is_monotonic_increasing
        assert group.polygon_area_m2.is_monotonic_increasing
