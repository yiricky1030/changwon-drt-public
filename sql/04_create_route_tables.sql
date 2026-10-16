CREATE TABLE IF NOT EXISTS transport.dim_transit_route (
 route_id text PRIMARY KEY, route_no text, route_type text, start_node_name text, end_node_name text,
 start_vehicle_time text, end_vehicle_time text, interval_time text, interval_sat_time text, interval_sun_time text,
 stop_count integer NOT NULL, valid_stop_count integer NOT NULL, geometry_method text,
 geometry geometry(LineString,4326), source text NOT NULL, retrieved_at timestamptz NOT NULL, dataset_version text NOT NULL
);
CREATE TABLE IF NOT EXISTS transport.bridge_route_stop (
 route_id text NOT NULL REFERENCES transport.dim_transit_route(route_id), stop_id text REFERENCES transport.dim_transit_stop(stop_id),
 node_id text, stop_name text, stop_order integer NOT NULL, match_method text NOT NULL, match_distance_m double precision,
 is_matched boolean NOT NULL, retrieved_at timestamptz NOT NULL, dataset_version text NOT NULL,
 UNIQUE(route_id,stop_order)
);
CREATE INDEX IF NOT EXISTS idx_dim_transit_route_geometry ON transport.dim_transit_route USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_dim_transit_route_no ON transport.dim_transit_route(route_no);
CREATE INDEX IF NOT EXISTS idx_bridge_route_stop_stop ON transport.bridge_route_stop(stop_id);
