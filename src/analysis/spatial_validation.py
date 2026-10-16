"""Spatial validation helpers that never mutate the canonical screening score.

The functions in this module are deliberately small and dependency-light.  They
support diagnostic analyses (Moran's I, OLS, observation propensity, synthetic
transport grids, and missingness scenarios), not causal or policy models.
"""
from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import optimize, stats
from shapely.geometry import box


def row_standardize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).copy()
    np.fill_diagonal(weights, 0.0)
    totals = weights.sum(axis=1)
    valid = totals > 0
    weights[valid] = weights[valid] / totals[valid, None]
    return weights


def queen_weights(frame: gpd.GeoDataFrame) -> tuple[np.ndarray, list[int]]:
    """Binary Queen contiguity, including vertex-only contact."""
    geometries = frame.geometry.reset_index(drop=True)
    n = len(geometries)
    weights = np.zeros((n, n), dtype=float)
    index = geometries.sindex
    for i, geometry in enumerate(geometries):
        for j in index.query(geometry, predicate="intersects"):
            if i != j:
                weights[i, int(j)] = 1.0
    weights = np.maximum(weights, weights.T)
    islands = np.flatnonzero(weights.sum(axis=1) == 0).astype(int).tolist()
    return row_standardize(weights), islands


def knn_weights(frame: gpd.GeoDataFrame, k: int = 4, metric_crs: int = 5186) -> np.ndarray:
    if not 1 <= k < len(frame):
        raise ValueError("k must be between 1 and n-1")
    points = frame.to_crs(metric_crs).geometry.centroid
    coordinates = np.column_stack([points.x, points.y])
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    weights = np.zeros_like(distance)
    for i in range(len(frame)):
        neighbors = np.argsort(distance[i], kind="stable")[:k]
        weights[i, neighbors] = 1.0
    # Symmetric union prevents directed-neighbor ambiguity in Moran diagnostics.
    weights = np.maximum(weights, weights.T)
    return row_standardize(weights)


def moran_i(values: Iterable[float], weights: np.ndarray, permutations: int = 999,
            seed: int = 20260912) -> dict:
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(values), len(values)):
        raise ValueError("weights shape does not match values")
    if not np.isfinite(values).all():
        raise ValueError("Moran input contains non-finite values")
    centered = values - values.mean()
    denominator = float(centered @ centered)
    s0 = float(weights.sum())
    if denominator == 0 or s0 == 0:
        raise ValueError("Moran input has zero variance or no links")

    def statistic(candidate: np.ndarray) -> float:
        return float(len(candidate) / s0 * (candidate @ weights @ candidate) / (candidate @ candidate))

    observed = statistic(centered)
    rng = np.random.default_rng(seed)
    simulated = np.array([statistic(rng.permutation(centered)) for _ in range(permutations)])
    mean = float(simulated.mean())
    std = float(simulated.std(ddof=1))
    p_value = float((np.count_nonzero(np.abs(simulated - mean) >= abs(observed - mean)) + 1)
                    / (permutations + 1))
    return {
        "I": observed,
        "expected_I": -1.0 / (len(values) - 1),
        "permutation_mean": mean,
        "z": (observed - mean) / std if std else None,
        "p_value": p_value,
        "permutations": int(permutations),
        "seed": int(seed),
        "n": int(len(values)),
    }


def fit_ols(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[dict, np.ndarray]:
    model = frame[[outcome, *predictors]].dropna().copy()
    y = model[outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(model)), model[predictors].to_numpy(float)])
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    dof = len(y) - x.shape[1]
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_value), dof)
    total = float(((y - y.mean()) ** 2).sum())
    sigma2_ml = float(residual @ residual / len(y))
    log_likelihood = float(-len(y) / 2 * (np.log(2 * np.pi * sigma2_ml) + 1))
    parameter_count = x.shape[1] + 1
    names = ["intercept", *predictors]
    coefficients = [
        {"term": name, "estimate": float(b), "std_error": float(se),
         "t": float(t), "p_value": float(p)}
        for name, b, se, t, p in zip(names, beta, standard_error, t_value, p_value)
    ]
    return ({"n": int(len(y)), "dof": int(dof), "r_squared": 1 - float(residual @ residual) / total,
             "log_likelihood": log_likelihood, "aic": 2 * parameter_count - 2 * log_likelihood,
             "sigma2_ml": sigma2_ml,
             "coefficients": coefficients, "row_index": model.index.astype(int).tolist()}, residual)


def _spatial_bounds(weights: np.ndarray) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvals(weights).real
    lower = max(-0.99, 1 / eigenvalues.min() + 1e-6) if eigenvalues.min() < 0 else -0.99
    upper = min(0.99, 1 / eigenvalues.max() - 1e-6) if eigenvalues.max() > 0 else 0.99
    return float(lower), float(upper)


def _coefficient_rows(names: list[str], beta: np.ndarray) -> list[dict]:
    return [{"term": name, "estimate": float(value)} for name, value in zip(names, beta)]


def fit_spatial_lag(frame: pd.DataFrame, outcome: str, predictors: list[str],
                    weights: np.ndarray) -> tuple[dict, np.ndarray]:
    """Concentrated Gaussian maximum-likelihood spatial lag model."""
    model = frame[[outcome, *predictors]].dropna().copy()
    y = model[outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(model)), model[predictors].to_numpy(float)])
    identity = np.eye(len(y))

    def components(rho: float):
        transform = identity - rho * weights
        transformed_y = transform @ y
        beta = np.linalg.lstsq(x, transformed_y, rcond=None)[0]
        innovations = transformed_y - x @ beta
        sigma2 = float(innovations @ innovations / len(y))
        sign, logdet = np.linalg.slogdet(transform)
        likelihood = (-np.inf if sign <= 0 or sigma2 <= 0 else
                      float(logdet - len(y) / 2 * (np.log(2 * np.pi * sigma2) + 1)))
        return likelihood, beta, innovations, sigma2

    lower, upper = _spatial_bounds(weights)
    optimized = optimize.minimize_scalar(lambda rho: -components(float(rho))[0], bounds=(lower, upper),
                                         method="bounded", options={"xatol": 1e-10})
    rho = float(optimized.x)
    likelihood, beta, innovations, sigma2 = components(rho)
    parameters = x.shape[1] + 2
    return ({"model": "spatial_lag_ml", "n": len(y), "rho": rho, "log_likelihood": likelihood,
             "aic": 2 * parameters - 2 * likelihood, "sigma2_ml": sigma2,
             "coefficients": _coefficient_rows(["intercept", *predictors], beta),
             "converged": bool(optimized.success), "rho_bounds": [lower, upper]}, innovations)


def fit_spatial_error(frame: pd.DataFrame, outcome: str, predictors: list[str],
                      weights: np.ndarray) -> tuple[dict, np.ndarray]:
    """Concentrated Gaussian maximum-likelihood spatial error model."""
    model = frame[[outcome, *predictors]].dropna().copy()
    y = model[outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(model)), model[predictors].to_numpy(float)])
    identity = np.eye(len(y))

    def components(lam: float):
        transform = identity - lam * weights
        transformed_y = transform @ y
        transformed_x = transform @ x
        beta = np.linalg.lstsq(transformed_x, transformed_y, rcond=None)[0]
        structural_residual = y - x @ beta
        innovations = transform @ structural_residual
        sigma2 = float(innovations @ innovations / len(y))
        sign, logdet = np.linalg.slogdet(transform)
        likelihood = (-np.inf if sign <= 0 or sigma2 <= 0 else
                      float(logdet - len(y) / 2 * (np.log(2 * np.pi * sigma2) + 1)))
        return likelihood, beta, innovations, sigma2

    lower, upper = _spatial_bounds(weights)
    optimized = optimize.minimize_scalar(lambda lam: -components(float(lam))[0], bounds=(lower, upper),
                                         method="bounded", options={"xatol": 1e-10})
    lam = float(optimized.x)
    likelihood, beta, innovations, sigma2 = components(lam)
    parameters = x.shape[1] + 2
    return ({"model": "spatial_error_ml", "n": len(y), "lambda": lam, "log_likelihood": likelihood,
             "aic": 2 * parameters - 2 * likelihood, "sigma2_ml": sigma2,
             "coefficients": _coefficient_rows(["intercept", *predictors], beta),
             "converged": bool(optimized.success), "lambda_bounds": [lower, upper]}, innovations)


def fit_logistic(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    """Maximum-likelihood logistic regression with Wald intervals."""
    model = frame[[outcome, *predictors]].dropna().copy()
    y = model[outcome].to_numpy(float)
    raw = model[predictors].to_numpy(float)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=0)
    if np.any(scales == 0):
        raise ValueError("constant logistic predictor")
    standardized = (raw - means) / scales
    x = np.column_stack([np.ones(len(model)), standardized])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        probability = np.clip(1 / (1 + np.exp(-np.clip(linear, -35, 35))), 1e-12, 1 - 1e-12)
        loss = -float((y * np.log(probability) + (1 - y) * np.log(1 - probability)).sum())
        gradient = x.T @ (probability - y)
        return loss, gradient

    result = optimize.minimize(lambda b: objective(b), np.zeros(x.shape[1]), jac=True, method="BFGS")
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError(f"logistic regression did not converge: {result.message}")
    beta = result.x
    probability = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
    hessian = x.T @ ((probability * (1 - probability))[:, None] * x)
    covariance = np.linalg.pinv(hessian)
    standard_error = np.sqrt(np.diag(covariance))
    z_value = beta / standard_error
    p_value = 2 * stats.norm.sf(np.abs(z_value))
    names = ["intercept", *predictors]
    coefficients = []
    for name, estimate, se, z, p in zip(names, beta, standard_error, z_value, p_value):
        coefficients.append({
            "term": name,
            "standardized_estimate": float(estimate),
            "odds_ratio_per_1sd": float(np.exp(estimate)),
            "ci95_low": float(np.exp(estimate - 1.96 * se)),
            "ci95_high": float(np.exp(estimate + 1.96 * se)),
            "z": float(z),
            "p_value": float(p),
        })
    return {
        "n": int(len(model)), "observed_n": int(y.sum()), "unobserved_n": int((1 - y).sum()),
        "converged": bool(result.success or np.linalg.norm(result.jac) <= 1e-4),
        "log_likelihood": -float(result.fun), "coefficients": coefficients,
        "standardization": {name: {"mean": float(mean), "sd": float(scale)}
                            for name, mean, scale in zip(predictors, means, scales)},
        "row_index": model.index.astype(int).tolist(),
        "predicted_probability": probability.tolist(),
    }


def synthetic_transport_grid(dongs: gpd.GeoDataFrame, stops: gpd.GeoDataFrame,
                             route_bridge: pd.DataFrame, schedules: pd.DataFrame,
                             cell_size_m: int = 500, metric_crs: int = 5186) -> gpd.GeoDataFrame:
    """Build a transport-only synthetic grid; no population is allocated to cells."""
    dong_metric = dongs.to_crs(metric_crs)
    city = dong_metric.geometry.union_all()
    minx, miny, maxx, maxy = city.bounds
    x0 = np.floor(minx / cell_size_m) * cell_size_m
    y0 = np.floor(miny / cell_size_m) * cell_size_m
    rows = []
    for x in np.arange(x0, maxx, cell_size_m):
        for y in np.arange(y0, maxy, cell_size_m):
            geometry = box(x, y, x + cell_size_m, y + cell_size_m)
            if geometry.intersects(city):
                rows.append({"grid_id": f"G{int(x)}_{int(y)}", "geometry": geometry.intersection(city)})
    grid = gpd.GeoDataFrame(rows, crs=metric_crs)
    centroids = gpd.GeoDataFrame(grid[["grid_id"]], geometry=grid.geometry.centroid, crs=metric_crs)
    assigned_dong = gpd.sjoin(centroids, dong_metric[["adm_cd", "adm_nm", "geometry"]],
                              how="left", predicate="within")
    assigned_dong = assigned_dong.sort_values(["grid_id", "adm_cd"]).drop_duplicates("grid_id")
    grid = grid.merge(assigned_dong[["grid_id", "adm_cd", "adm_nm"]], on="grid_id", how="left")

    stop_metric = stops.to_crs(metric_crs).copy()
    joined = gpd.sjoin(stop_metric, grid[["grid_id", "geometry"]], how="left", predicate="within")
    if joined.grid_id.isna().any():
        # Boundary points use intersects deterministically; no name-based matching.
        missing = joined.grid_id.isna()
        fallback = gpd.sjoin(stop_metric.loc[missing], grid[["grid_id", "geometry"]],
                             how="left", predicate="intersects").sort_values(["stop_id", "grid_id"])
        fallback = fallback.drop_duplicates("stop_id").set_index("stop_id")["grid_id"]
        joined.loc[missing, "grid_id"] = joined.loc[missing, "stop_id"].map(fallback)
    bridge = route_bridge.loc[route_bridge.is_matched & route_bridge.stop_id.notna(), ["stop_id", "route_id"]].copy()
    bridge["stop_id"] = bridge.stop_id.astype(str)
    bridge = bridge.drop_duplicates()
    joined["stop_id"] = joined.stop_id.astype(str)
    stop_grid = joined[["stop_id", "grid_id"]].dropna().drop_duplicates()
    route_grid = stop_grid.merge(bridge, on="stop_id", how="left")
    schedule = schedules[["route_id", "estimated_scheduled_trips_weekday"]].drop_duplicates("route_id")
    route_grid = route_grid.merge(schedule, on="route_id", how="left")
    route_summary = route_grid.groupby("grid_id").agg(
        route_count=("route_id", "nunique"),
        service_intensity=("estimated_scheduled_trips_weekday", lambda s: s.dropna().sum(min_count=1)),
    )
    point_summary = joined.groupby("grid_id").agg(
        stop_count=("stop_id", "nunique"),
        observed_stop_count=("has_observed_profile", "sum"),
        observed_use=("ridership_total", lambda s: s.sum(min_count=1)),
    )
    grid = grid.merge(point_summary, on="grid_id", how="left").merge(route_summary, on="grid_id", how="left")
    for column in ["stop_count", "observed_stop_count", "route_count"]:
        grid[column] = grid[column].fillna(0).astype(int)
    # observed_use/service_intensity remain NULL when no relevant observation exists.
    grid["population_total"] = pd.NA
    grid["population_65_plus"] = pd.NA
    grid["grid_type"] = "synthetic analytical grid"
    grid["cell_size_m"] = int(cell_size_m)
    return grid


def grid_dong_summary(grid: gpd.GeoDataFrame, representative_names: list[str]) -> pd.DataFrame:
    metrics = ["stop_count", "route_count", "observed_use", "service_intensity"]
    city_percentiles = {metric: grid[metric].rank(method="average", pct=True) * 100 for metric in metrics}
    rows = []
    for name in representative_names:
        subset = grid.loc[grid.adm_nm.eq(name)].copy()
        row = {"adm_nm": name, "grid_count": int(len(subset))}
        for metric in metrics:
            values = subset[metric].dropna().astype(float)
            row[f"{metric}_within_variance"] = float(values.var(ddof=0)) if len(values) else None
            pct = city_percentiles[metric].loc[values.index]
            row[f"{metric}_percentile_min"] = float(pct.min()) if len(pct) else None
            row[f"{metric}_percentile_max"] = float(pct.max()) if len(pct) else None
            total = float(values.sum()) if len(values) else 0.0
            row[f"{metric}_top_grid_concentration"] = float(values.max() / total) if total > 0 else None
            row[f"{metric}_nonzero_grid_share"] = float(values.gt(0).mean()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_kappa(rater_a: Iterable[str], rater_b: Iterable[str]) -> float | None:
    """Cohen's kappa for two raters over matched observations.

    Returns None when fewer than two paired observations exist or when both
    raters assign every pair to a single identical category (kappa is
    undefined because there is no disagreement to model against chance).
    """
    a = list(rater_a)
    b = list(rater_b)
    if len(a) != len(b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(a)
    if n == 0:
        return None
    categories = sorted(set(a) | set(b))
    if len(categories) < 2:
        return None
    index = {category: i for i, category in enumerate(categories)}
    confusion = np.zeros((len(categories), len(categories)), dtype=float)
    for observed_a, observed_b in zip(a, b):
        confusion[index[observed_a], index[observed_b]] += 1
    observed_agreement = float(np.trace(confusion)) / n
    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    expected_agreement = float((row_marginals * col_marginals).sum())
    if expected_agreement >= 1.0:
        return None
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def field_inter_rater_agreement(frame: pd.DataFrame, item_columns: list[str],
                                pair_columns: list[str] = ("stop_id", "visit_round"),
                                rater_column: str = "rater_id",
                                rater_values: tuple[str, str] = ("A", "B"),
                                not_observed_label: str = "NOT_OBSERVED") -> dict:
    """Percent agreement and Cohen's kappa between two independent field raters.

    Designed for `evaluation/field_inter_rater_template.csv`. Every item column
    is compared per (stop_id, visit_round) pair. `not_observed_label` rows are
    counted in the "including NOT_OBSERVED" agreement figure and excluded from
    the "valid pairs only" figure and from kappa, since NOT_OBSERVED reflects a
    missed visit rather than a substantive rating. Returns None for any metric
    that cannot be computed from the rows actually present (this function does
    not fabricate or assume field observations).
    """
    pivot = frame.pivot_table(index=list(pair_columns), columns=rater_column,
                              values=item_columns, aggfunc="first")
    per_item = {}
    total_pairs = 0
    total_agree_all = 0
    total_valid_pairs = 0
    total_agree_valid = 0
    for item in item_columns:
        if item not in pivot.columns.get_level_values(0):
            per_item[item] = {"n_pairs": 0, "percent_agreement_all": None,
                              "n_valid_pairs": 0, "percent_agreement_valid": None,
                              "cohens_kappa": None}
            continue
        sub = pivot[item].reindex(columns=list(rater_values)).dropna(how="any")
        a = sub[rater_values[0]].tolist()
        b = sub[rater_values[1]].tolist()
        n_pairs = len(a)
        agree_all = sum(1 for x, y in zip(a, b) if x == y)
        valid = [(x, y) for x, y in zip(a, b) if x != not_observed_label and y != not_observed_label]
        n_valid = len(valid)
        agree_valid = sum(1 for x, y in valid if x == y)
        kappa = cohens_kappa([x for x, _ in valid], [y for _, y in valid]) if n_valid else None
        per_item[item] = {
            "n_pairs": n_pairs,
            "percent_agreement_all": (agree_all / n_pairs) if n_pairs else None,
            "n_valid_pairs": n_valid,
            "percent_agreement_valid": (agree_valid / n_valid) if n_valid else None,
            "cohens_kappa": kappa,
        }
        total_pairs += n_pairs
        total_agree_all += agree_all
        total_valid_pairs += n_valid
        total_agree_valid += agree_valid
    overall = {
        "n_pairs": total_pairs,
        "percent_agreement_all": (total_agree_all / total_pairs) if total_pairs else None,
        "n_valid_pairs": total_valid_pairs,
        "percent_agreement_valid": (total_agree_valid / total_valid_pairs) if total_valid_pairs else None,
    }
    return {"per_item": per_item, "overall": overall}


def missingness_scenarios(dongs: pd.DataFrame, stops: pd.DataFrame, iterations: int = 500,
                          seed: int = 20260912) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conditional empirical scenarios; values are not imputed true demand."""
    frame = stops[["stop_id", "adm_cd", "route_count", "ridership_total", "has_observed_profile"]].copy()
    observed = frame.loc[frame.has_observed_profile & frame.ridership_total.notna()].copy()
    missing = frame.loc[~frame.has_observed_profile].copy()
    observed["route_stratum"] = pd.cut(observed.route_count, [-np.inf, 0, 2, 5, 10, np.inf], labels=False)
    missing["route_stratum"] = pd.cut(missing.route_count, [-np.inf, 0, 2, 5, 10, np.inf], labels=False)
    pools = {int(key): group.ridership_total.to_numpy(float) for key, group in observed.groupby("route_stratum")}
    fallback = observed.ridership_total.to_numpy(float)
    canonical = dongs.set_index("adm_cd")

    def score(extra: pd.Series) -> pd.DataFrame:
        result = canonical.copy()
        result["scenario_ridership_total"] = result.ridership_total.add(extra, fill_value=0)
        result["scenario_ridership_per_1000"] = result.scenario_ridership_total / result.population_total * 1000
        result["scenario_ridership_pct"] = result.scenario_ridership_per_1000.rank(method="average", pct=True) * 100
        result["scenario_score"] = (result.scenario_ridership_pct * .40
                                    + result.stop_supply_scarcity_pct * .20
                                    + result.route_supply_scarcity_pct * .20
                                    + result.top3_hour_share_pct * .20)
        result["scenario_rank"] = result.scenario_score.rank(method="min", ascending=False)
        return result

    summaries = []
    for label, quantile in [("LOW", .25), ("MEDIAN", .50), ("HIGH", .75)]:
        values = missing.route_stratum.map(lambda key: float(np.quantile(pools.get(int(key), fallback), quantile)))
        extra = values.groupby(missing.adm_cd).sum(min_count=1)
        ranked = score(extra)
        for adm_cd, row in ranked.iterrows():
            summaries.append({"scenario": label, "adm_cd": adm_cd, "adm_nm": row.adm_nm,
                              "scenario_score": row.scenario_score, "scenario_rank": row.scenario_rank,
                              "additional_ridership": float(extra.get(adm_cd, 0.0))})

    rng = np.random.default_rng(seed)
    rank_rows = []
    for iteration in range(iterations):
        sampled = []
        for row in missing.itertuples():
            pool = pools.get(int(row.route_stratum), fallback)
            sampled.append(float(rng.choice(pool)))
        extra = pd.Series(sampled, index=missing.index).groupby(missing.adm_cd).sum(min_count=1)
        ranked = score(extra)
        for adm_cd, row in ranked.iterrows():
            rank_rows.append({"iteration": iteration, "adm_cd": adm_cd, "adm_nm": row.adm_nm,
                              "scenario_rank": row.scenario_rank,
                              "top10_inclusion": (pd.NA if pd.isna(row.scenario_rank)
                                                  else bool(row.scenario_rank <= 10))})
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame["top10_inclusion"] = rank_frame.top10_inclusion.astype("boolean")
    distribution = rank_frame.groupby(["adm_cd", "adm_nm"], as_index=False).agg(
        rank_min=("scenario_rank", "min"), rank_median=("scenario_rank", "median"),
        rank_max=("scenario_rank", "max"), top10_inclusion_share=("top10_inclusion", "mean"),
    )
    return pd.DataFrame(summaries), distribution


def read_official_population_grid(shp_path, source_crs: int = 5179) -> gpd.GeoDataFrame:
    """Read one NGII population-grid shapefile (single `val` variable per file).

    NGII ships these grids with a `.cst` sidecar declaring UTF-8; the default
    OGR/pyogrio encoding guess is wrong (mis-decodes Korean grid-id strings),
    so encoding is forced explicitly. Three distinct missingness states are
    preserved rather than collapsed to a number:

    - KNOWN: a real reported value.
    - MASKED_SUPPRESSED: NGII's own disclosure-control marks the cell "N/A"
      (small-count privacy masking) while still writing a placeholder `val`
      of 0 — that 0 is not a real observation and is discarded here.
    - NO_DATA: both the label and value fields are empty; the cell was not
      part of the attribute join at all.

    `value` is NaN for both non-KNOWN states so no downstream aggregation can
    silently treat suppressed-or-absent population as zero.
    """
    frame = gpd.read_file(shp_path, encoding="utf-8")
    if frame.crs is None or frame.crs.to_epsg() != source_crs:
        raise ValueError(f"Expected source CRS EPSG:{source_crs}, got {frame.crs}")
    label = frame["lbl"].astype("string")
    raw_value = pd.to_numeric(frame["val"], errors="coerce")
    status = pd.Series("KNOWN", index=frame.index, dtype="string")
    status = status.mask(label.isna() & raw_value.isna(), "NO_DATA")
    # fillna(False): a nullable-string `==` against a genuinely missing label must
    # not accidentally read as "masked" (NA is not truthy, but must be forced here
    # so pandas' .mask() never has to guess).
    status = status.mask(label.eq("N/A").fillna(False), "MASKED_SUPPRESSED")
    value = raw_value.where(status.eq("KNOWN"))
    return gpd.GeoDataFrame(
        {"grid_id": frame["gid"].astype("string"), "value": value, "status": status},
        geometry=frame.geometry, crs=frame.crs)


def merge_population_variables(total_grid: gpd.GeoDataFrame, elderly_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Outer-merge two single-variable official grids (total, elderly) on grid_id.

    The two source files can be different vintages with slightly different
    cell membership (statistical disclosure control can vary by year), so this
    is an outer join, not an assumption that both files cover the same cells.
    """
    total = total_grid.rename(columns={"value": "population_total", "status": "population_total_status"})
    elderly = elderly_grid.rename(columns={"value": "population_65_plus", "status": "population_65_plus_status"})
    merged = total.merge(
        elderly.drop(columns="geometry"), on="grid_id", how="outer", validate="one_to_one")
    missing_geometry = merged.geometry.isna()
    if missing_geometry.any():
        fallback = elderly.set_index("grid_id").geometry
        merged.loc[missing_geometry, "geometry"] = merged.loc[missing_geometry, "grid_id"].map(fallback)
    merged["population_total_status"] = merged.population_total_status.fillna("NO_DATA")
    merged["population_65_plus_status"] = merged.population_65_plus_status.fillna("NO_DATA")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=total_grid.crs)


def assign_grid_cells_to_dongs(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame,
                               method: str = "centroid") -> pd.DataFrame:
    """One grid_id -> one adm_cd/adm_nm assignment, by one of two explicit rules.

    - "centroid" (default): the dong containing the cell's centroid. Deterministic,
      cheap, but a large cell whose centroid lands just inside a small dong can pull
      that dong's grid-sum toward a neighboring cell's true population.
    - "largest_overlap": the dong with the largest intersection AREA with the cell
      polygon (winner-take-all by area, not a fractional population split). This is
      an alternative assignment rule for sensitivity comparison only, never the
      default, and it does not allocate a cell's population across multiple dongs.

    Cells that intersect no dong at all are absent from the result (not assigned).
    """
    dong_metric = dongs.to_crs(grid.crs)[["adm_cd", "adm_nm", "geometry"]]
    if method == "centroid":
        centroids = gpd.GeoDataFrame(grid[["grid_id"]], geometry=grid.geometry.centroid, crs=grid.crs)
        joined = gpd.sjoin(centroids, dong_metric, how="inner", predicate="within")
        return joined[["grid_id", "adm_cd", "adm_nm"]].drop_duplicates("grid_id").reset_index(drop=True)
    if method == "largest_overlap":
        cells = grid[["grid_id", "geometry"]]
        overlay = gpd.overlay(cells, dong_metric, how="intersection", keep_geom_type=False)
        if overlay.empty:
            return pd.DataFrame(columns=["grid_id", "adm_cd", "adm_nm"])
        overlay["overlap_area"] = overlay.geometry.area
        winners = overlay.sort_values("overlap_area", ascending=False).drop_duplicates("grid_id")
        return winners[["grid_id", "adm_cd", "adm_nm"]].reset_index(drop=True)
    raise ValueError(f"Unknown assignment method: {method}")


def boundary_crossing_cell_counts(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per grid_id, how many distinct dong polygons its geometry intersects.

    A count of 2+ means the cell straddles an administrative boundary; centroid
    assignment sends the whole cell to just one of those dongs regardless.
    """
    dong_metric = dongs.to_crs(grid.crs)[["adm_cd", "geometry"]]
    overlay = gpd.overlay(grid[["grid_id", "geometry"]], dong_metric, how="intersection", keep_geom_type=False)
    counts = overlay.groupby("grid_id").adm_cd.nunique().rename("intersecting_dong_count").reset_index()
    return counts


def official_grid_dong_comparison(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame,
                                  method: str = "centroid") -> pd.DataFrame:
    """Compare official-grid-summed population against the canonical per-dong value.

    This is a MAUP-adjacent diagnostic: administrative dong totals average away
    within-dong spatial concentration that the 500m/100m grid can show, and the
    cell-to-dong assignment rule itself materially affects the result for small
    dongs (see assign_grid_cells_to_dongs). Grid sums use only KNOWN cells;
    MASKED/NO_DATA cell counts are reported alongside so a large gap can be
    explained by suppression or assignment-rule sensitivity rather than silently
    treated as a real shortfall. `method="largest_overlap"` requires the strong
    assumption that population is uniform within a cell to make area a population
    proxy, so it is offered only as an explicit sensitivity option, not the default.
    """
    dong_metric = dongs.to_crs(grid.crs)[["adm_cd", "adm_nm", "population_total", "population_65_plus"]]
    assignment = assign_grid_cells_to_dongs(grid, dongs, method=method)
    joined = assignment.merge(grid.drop(columns="geometry"), on="grid_id", how="left")

    def summarize(column: str, status_column: str) -> pd.DataFrame:
        known = joined.loc[joined[status_column].eq("KNOWN")]
        sums = known.groupby("adm_cd")[column].sum(min_count=1).rename(f"grid_sum_{column}")
        counts = joined.groupby(["adm_cd", status_column]).size().unstack(fill_value=0)
        counts.columns = [f"{status_column.split('_status')[0]}_cells_{status.lower()}" for status in counts.columns]
        return pd.concat([sums, counts], axis=1)

    total_summary = summarize("population_total", "population_total_status")
    elderly_summary = summarize("population_65_plus", "population_65_plus_status")
    combined = dong_metric.set_index("adm_cd").join([total_summary, elderly_summary])
    combined["population_total_ratio_grid_over_canonical"] = (
        combined.grid_sum_population_total / combined.population_total)
    combined["population_65_plus_ratio_grid_over_canonical"] = (
        combined.grid_sum_population_65_plus / combined.population_65_plus)
    return combined.reset_index()


def dong_ratio_decomposition(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame,
                             dong_names: list[str]) -> pd.DataFrame:
    """Per-dong breakdown behind a grid-vs-canonical population ratio.

    Reports, for each named dong: official (canonical) population, grid KNOWN sum,
    masked/no-data cell counts, how many cells were centroid-assigned to it, and how
    many of those assigned cells actually straddle a dong boundary (intersect more
    than one dong). This is the decomposition a bare ratio number cannot show.
    """
    dong_metric = dongs.to_crs(grid.crs)[["adm_cd", "adm_nm", "population_total"]]
    assignment = assign_grid_cells_to_dongs(grid, dongs, method="centroid")
    crossing = boundary_crossing_cell_counts(grid, dongs)
    joined = assignment.merge(grid.drop(columns="geometry"), on="grid_id", how="left").merge(
        crossing, on="grid_id", how="left")
    rows = []
    for name in dong_names:
        subset = joined.loc[joined.adm_nm.eq(name)]
        known = subset.loc[subset.population_total_status.eq("KNOWN")]
        canonical = dong_metric.loc[dong_metric.adm_nm.eq(name), "population_total"]
        rows.append({
            "adm_nm": name,
            "official_dong_population": float(canonical.iloc[0]) if len(canonical) else None,
            "grid_known_population_sum": float(known.population_total.sum()) if len(known) else None,
            "centroid_assigned_cells": int(len(subset)),
            "known_cells": int(known.shape[0]),
            "masked_cells": int(subset.population_total_status.eq("MASKED_SUPPRESSED").sum()),
            "no_data_cells": int(subset.population_total_status.eq("NO_DATA").sum()),
            "boundary_crossing_cells": int(subset.intersecting_dong_count.gt(1).sum()),
        })
    return pd.DataFrame(rows)


def citywide_population_reconciliation(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame,
                                       column: str, status_column: str) -> dict:
    """Citywide KNOWN/MASKED/NO_DATA totals vs. the sum actually assigned to a dong.

    `known_total` is explicitly NOT called a population estimate: MASKED cells have
    unknown (not zero) population, so `known_total` is a lower bound, not the true
    citywide total. `assignment_gap` cells intersect no dong (outside the 55-dong
    boundary) and are excluded from any dong-level sum by construction.
    """
    status_counts = grid[status_column].value_counts().to_dict()
    known = grid.loc[grid[status_column].eq("KNOWN")]
    known_total = float(known[column].sum()) if len(known) else 0.0
    assignment = assign_grid_cells_to_dongs(grid, dongs, method="centroid")
    assigned_known = assignment.merge(known[["grid_id", column]], on="grid_id", how="inner")
    assigned_known_total = float(assigned_known[column].sum()) if len(assigned_known) else 0.0
    outside_known = known.loc[~known.grid_id.isin(assigned_known.grid_id)]
    outside_known_total = float(outside_known[column].sum()) if len(outside_known) else 0.0
    # True partition check: every KNOWN cell is either assigned to exactly one dong
    # (centroid method, deduplicated) or entirely outside all 55 dongs — never both,
    # never neither. This should hold exactly by construction; it is not a lenient
    # "explained by an excuse" check.
    conserved = abs(known_total - (assigned_known_total + outside_known_total)) < 1e-6 * max(known_total, 1)
    return {
        "known_cell_count": int(status_counts.get("KNOWN", 0)),
        "masked_suppressed_cell_count": int(status_counts.get("MASKED_SUPPRESSED", 0)),
        "no_data_cell_count": int(status_counts.get("NO_DATA", 0)),
        "citywide_known_total": known_total,
        "dong_assigned_known_total": assigned_known_total,
        "cells_outside_any_dong": int(len(known) - len(assigned_known)),
        "population_outside_any_dong": outside_known_total,
        "conserved": bool(conserved),
    }


def population_weighted_resident_origin_summary(
        origins: pd.DataFrame, weight_column: str,
        value_columns: tuple[str, ...] = ("accessible_stop_count", "accessible_route_count",
                                          "estimated_route_trips_accessible"),
        group_cols: tuple[str, ...] = ("adm_cd", "adm_nm", "radius_m")) -> pd.DataFrame:
    """Population-weighted (not simple-mean) resident-origin summary.

    Origins with a missing/NaN weight are excluded from the weighted average
    entirely (never coerced to a weight of 0, which would silently discard their
    contribution while pretending they were real zero-population points). This
    does not claim individual-level accessibility; it only changes how the
    per-dong average is computed across the same origin points already selected
    for the unweighted diagnostic.
    """
    valid = origins.loc[origins[weight_column].notna() & (origins[weight_column] > 0)].copy()
    excluded = int(len(origins) - len(valid))
    rows = []
    for keys, group in valid.groupby(list(group_cols), sort=True):
        weights = group[weight_column].to_numpy(float)
        row = dict(zip(group_cols, keys))
        row["origin_count"] = int(len(group))
        row["weight_total_population"] = float(weights.sum())
        for column in value_columns:
            row[f"{column}_population_weighted_mean"] = float(np.average(group[column].to_numpy(float), weights=weights))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["excluded_missing_weight_rows"] = excluded
    return result


def select_population_weighted_origins(grid: gpd.GeoDataFrame, dongs: gpd.GeoDataFrame,
                                       count: int = 5) -> gpd.GeoDataFrame:
    """Up to `count` spatially dispersed KNOWN, populated grid-cell centroids per dong.

    Mirrors the dispersion logic used for the stop-origin diagnostic (farthest-
    point sampling), but the candidate pool is now residential grid cells
    instead of official stops, so these are genuine resident origins rather
    than transport-supply origins.
    """
    dong_metric = dongs.to_crs(grid.crs)[["adm_cd", "adm_nm", "geometry"]]
    candidates = grid.loc[grid.population_total_status.eq("KNOWN") & grid.population_total.gt(0)].copy()
    carry_columns = ["grid_id", "population_total"]
    if "population_65_plus" in candidates.columns:
        carry_columns += ["population_65_plus", "population_65_plus_status"]
    centroids = gpd.GeoDataFrame(
        candidates[carry_columns], geometry=candidates.geometry.centroid, crs=grid.crs)
    joined = gpd.sjoin(centroids, dong_metric, how="inner", predicate="within").reset_index(drop=True)
    selected_rows = []
    for _, group in joined.groupby("adm_cd", sort=True):
        coordinates = np.column_stack([group.geometry.x, group.geometry.y])
        centroid = coordinates.mean(axis=0)
        selected = [int(np.argmin(np.linalg.norm(coordinates - centroid, axis=1)))]
        while len(selected) < min(count, len(group)):
            distance = np.min(np.linalg.norm(coordinates[:, None, :] - coordinates[selected][None, :, :], axis=2), axis=1)
            distance[selected] = -1
            selected.append(int(np.argmax(distance)))
        selected_rows.append(group.iloc[selected])
    return gpd.GeoDataFrame(pd.concat(selected_rows, ignore_index=True), geometry="geometry", crs=grid.crs)


def anchor_population_context(anchors: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                              radii_m: tuple[int, ...] = (400, 800, 1200),
                              resolution_m: int | None = None,
                              reference_dates: dict[str, str] | None = None) -> pd.DataFrame:
    """Grid-centroid-based population CONTEXT around each Anchor — not a resident-count claim.

    Inclusion rule is explicit: a grid cell counts toward an Anchor's radius when
    that cell's CENTROID falls within a straight-line (Euclidean, not walk-network)
    buffer of the given radius. This is population context, not "actual radius-m
    resident population" or "walking-catchment population" — the cell can extend
    beyond the buffer edge, and a cell's population is not observed at the
    centroid point. KNOWN-only sums are reported alongside MASKED/NO_DATA cell
    counts within the same buffer so a low sum caused by suppression is visible.
    `resolution_m` and `reference_dates` are attached to every row purely as
    metadata so a reader never has to guess which grid vintage/resolution produced
    a given number.
    """
    anchor_metric = anchors.to_crs(grid.crs).reset_index(drop=True)
    centroids = grid.geometry.centroid
    rows = []
    for _, anchor in anchor_metric.iterrows():
        for radius in radii_m:
            within = grid.loc[centroids.distance(anchor.geometry) <= radius]
            row = {"stop_id": str(anchor.stop_id), "stop_name": anchor.stop_name,
                  "adm_cd": anchor.adm_cd, "adm_nm": anchor.adm_nm, "radius_m": radius,
                  "resolution_m": resolution_m,
                  "total_reference_date": (reference_dates or {}).get("total"),
                  "elderly_reference_date": (reference_dates or {}).get("elderly")}
            for column, status_column, prefix in [
                ("population_total", "population_total_status", "population_total"),
                ("population_65_plus", "population_65_plus_status", "population_65_plus"),
            ]:
                known = within.loc[within[status_column].eq("KNOWN")]
                row[f"{prefix}_context_sum"] = float(known[column].sum()) if len(known) else None
                for status in ("KNOWN", "MASKED_SUPPRESSED", "NO_DATA"):
                    row[f"{prefix}_cells_{status.lower()}"] = int(within[status_column].eq(status).sum())
            rows.append(row)
    return pd.DataFrame(rows)
