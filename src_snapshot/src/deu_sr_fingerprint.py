"""
Flat SR / Minkowski interval fingerprint tools for the DEU triangular foam.

Run after:
    %run -i src/deu_exp456_minimal.py
    %run -i src/deu_exp456_overflow_patch.py

The functions intentionally compare foam diamonds to controls using only
order-invariant interval statistics: MM dimension, ordering fraction,
V~h^D, A~h^(D-1), and normalized layer profiles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def sr_loglog_fit(x, y, min_points: int = 3) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if len(x) < min_points:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n_fit": int(len(x))}
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = intercept + slope * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n_fit": int(len(x))}


def _sample_spatial_ball(rng: np.random.Generator, spatial_dim: int, radius: float) -> np.ndarray:
    if spatial_dim <= 0:
        return np.empty(0)
    direction = rng.normal(size=spatial_dim)
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        direction[0] = 1.0
        norm = 1.0
    direction = direction / norm
    r = radius * (rng.random() ** (1.0 / spatial_dim))
    return direction * r


def make_minkowski_interval_causet(n_interior: int = 250, d: int = 3, seed: int = 0, T: float = 2.0):
    """
    Build a fixed-endpoint d-dimensional Minkowski Alexandrov interval.

    Events 0 and n+1 are the bottom/top endpoints after topological sorting.
    Interior points are uniformly sprinkled in the continuum diamond.
    """
    rng = np.random.default_rng(seed)
    spatial_dim = int(d) - 1
    R = T / 2.0

    times = [-R]
    positions = [np.zeros(spatial_dim)]

    for _ in range(int(n_interior)):
        side = -1.0 if rng.random() < 0.5 else 1.0
        u = rng.random()
        s = 1.0 - (1.0 - u) ** (1.0 / float(d))
        t = side * R * s
        cross_section_radius = max(0.0, R - abs(t))
        x = _sample_spatial_ball(rng, spatial_dim, cross_section_radius)
        times.append(float(t))
        positions.append(x)

    times.append(R)
    positions.append(np.zeros(spatial_dim))

    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    order = np.argsort(times, kind="mergesort")

    # Re-index by time so build_bitset_causet can compute closure safely.
    old_to_new = {int(old): int(new) for new, old in enumerate(order)}
    sorted_times = times[order]
    sorted_positions = positions[order]
    n_total = len(order)

    elements = set(range(n_total))
    children = {i: set() for i in range(n_total)}

    for i in range(n_total):
        future = np.arange(i + 1, n_total)
        if len(future) == 0:
            continue
        dt = sorted_times[future] - sorted_times[i]
        if spatial_dim > 0:
            dx = sorted_positions[future] - sorted_positions[i]
            ds = np.linalg.norm(dx, axis=1)
        else:
            ds = np.zeros_like(dt)
        related = future[dt >= ds - 1e-12]
        children[i].update(int(j) for j in related)

    roles = {i: "endpoint" if i in (0, n_total - 1) else "sprinkle" for i in elements}
    kinds = {i: f"Minkowski_{d}D_interval" for i in elements}
    basins = {i: -1 for i in elements}
    epochs = {i: float(sorted_times[i]) for i in elements}

    return build_bitset_causet(elements, children, roles, kinds, basins, epochs), 0, n_total - 1


def make_chain_interval_causet(n_interior: int = 250):
    n_total = int(n_interior) + 2
    elements = set(range(n_total))
    children = {i: {i + 1} for i in range(n_total - 1)}
    children[n_total - 1] = set()
    roles = {i: "endpoint" if i in (0, n_total - 1) else "chain" for i in elements}
    kinds = {i: "chain_interval" for i in elements}
    basins = {i: -1 for i in elements}
    epochs = {i: float(i) for i in elements}
    return build_bitset_causet(elements, children, roles, kinds, basins, epochs), 0, n_total - 1


def make_random_dag_interval_causet(n_interior: int = 250, p: float = 0.025, seed: int = 0):
    """
    Random DAG interval with fixed endpoints. This is a non-Lorentzian foil.
    Endpoint 0 precedes all interior nodes; all interiors precede endpoint n+1.
    """
    rng = np.random.default_rng(seed)
    n_total = int(n_interior) + 2
    elements = set(range(n_total))
    children = {i: set() for i in range(n_total)}
    # Endpoints ensure the whole set is a causal interval.
    for i in range(1, n_total - 1):
        children[0].add(i)
        children[i].add(n_total - 1)
    # Sparse random direct edges among interior nodes.
    for i in range(1, n_total - 1):
        for j in range(i + 1, n_total - 1):
            if rng.random() < p:
                children[i].add(j)
    roles = {i: "endpoint" if i in (0, n_total - 1) else "random" for i in elements}
    kinds = {i: "random_DAG_interval" for i in elements}
    basins = {i: -1 for i in elements}
    epochs = {i: float(i) for i in elements}
    return build_bitset_causet(elements, children, roles, kinds, basins, epochs), 0, n_total - 1


def _profile_from_layers(layer_counts: Dict[int, int], h_inclusive: int, n_bins: int = 21, normalize: str = "area"):
    """
    Convert strict layer counts, keyed by inclusive rank, into fixed normalized bins.
    """
    bins = np.linspace(0.0, 1.0, int(n_bins))
    vals = np.zeros(len(bins), dtype=float)
    if h_inclusive <= 1 or not layer_counts:
        return bins, vals
    for rank, count in layer_counts.items():
        # rank=1 and rank=h are endpoints; strict layers live mostly in between.
        x = (float(rank) - 1.0) / max(1.0, float(h_inclusive) - 1.0)
        bi = int(np.clip(np.round(x * (len(bins) - 1)), 0, len(bins) - 1))
        vals[bi] += float(count)
    if normalize == "max":
        mx = float(np.max(vals))
        if mx > 0:
            vals = vals / mx
    elif normalize == "area":
        sm = float(np.sum(vals))
        if sm > 0:
            vals = vals / sm
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'area', 'max', or 'none'")
    return bins, vals


def fingerprint_interval(
    causet,
    strict_bits: int,
    endpoint_indices: Optional[Tuple[int, int]],
    model: str,
    interval_id: str,
    n_profile_bins: int = 21,
    meta: Optional[Dict] = None,
):
    """
    Compute one interval fingerprint and normalized layer profile.
    """
    strict_bits = int(strict_bits)
    if endpoint_indices is not None:
        a_index, b_index = endpoint_indices
        inclusive_bits = strict_bits | bit_at(int(a_index)) | bit_at(int(b_index))
    else:
        a_index = b_index = np.nan
        inclusive_bits = strict_bits

    V_strict = strict_bits.bit_count()
    V_inclusive = inclusive_bits.bit_count()

    incl_ranks, incl_layers, h_inclusive = longest_chain_layers_bits(causet, inclusive_bits)
    strict_layer_counts = {}
    for i in iter_bits(strict_bits):
        r = int(incl_ranks.get(i, 0))
        strict_layer_counts[r] = strict_layer_counts.get(r, 0) + 1

    if h_inclusive > 0:
        mid_rank = int(round((h_inclusive + 1) / 2.0))
        mids = [mid_rank]
        if h_inclusive % 2 == 0:
            mids = [h_inclusive // 2, h_inclusive // 2 + 1]
        mid_layer = max(strict_layer_counts.get(r, 0) for r in mids) if strict_layer_counts else 0
        mid_band = sum(strict_layer_counts.get(r, 0) for r in sorted(set(mids))) if strict_layer_counts else 0
    else:
        mid_rank = 0
        mid_layer = 0
        mid_band = 0

    max_layer = max(strict_layer_counts.values(), default=0)
    rel = count_related_pairs(strict_bits, causet.descendant_bits) if V_strict >= 2 else 0
    total = V_strict * (V_strict - 1) // 2
    r_frac = rel / total if total > 0 else np.nan
    d_mm = mm_dim(r_frac) if total > 0 and np.isfinite(r_frac) else np.nan

    xbins, prof = _profile_from_layers(strict_layer_counts, h_inclusive, n_bins=n_profile_bins, normalize="area")

    row = {
        "model": model,
        "interval_id": interval_id,
        "a_index": a_index,
        "b_index": b_index,
        "V_strict": V_strict,
        "V_inclusive": V_inclusive,
        "h_inclusive": h_inclusive,
        "mid_rank": mid_rank,
        "mid_layer_size": mid_layer,
        "mid_band_size": mid_band,
        "max_layer_size": max_layer,
        "n_nonempty_strict_layers": int(len(strict_layer_counts)),
        "r_strict": float(r_frac) if np.isfinite(r_frac) else np.nan,
        "MM_dim_strict": float(d_mm) if np.isfinite(d_mm) else np.nan,
    }
    if meta:
        row.update(meta)

    profile_rows = []
    for k, (x, y) in enumerate(zip(xbins, prof)):
        profile_rows.append({
            "model": model,
            "interval_id": interval_id,
            "profile_bin": int(k),
            "x_norm": float(x),
            "profile_area_norm": float(y),
        })
    return row, profile_rows


def fingerprint_fixed_endpoint_causet(causet, a_index: int, b_index: int, model: str, interval_id: str, n_profile_bins: int = 21, meta: Optional[Dict] = None):
    strict_bits = int(causet.descendant_bits[int(a_index)] & causet.ancestor_bits[int(b_index)])
    return fingerprint_interval(causet, strict_bits, (int(a_index), int(b_index)), model, interval_id, n_profile_bins=n_profile_bins, meta=meta)


def build_control_fingerprints(
    sizes: Sequence[int] = (40, 60, 90, 130, 190, 280, 420),
    reps_per_size: int = 4,
    seed: int = 123,
    n_profile_bins: int = 21,
    random_dag_p: float = 0.025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate chain, 2+1D Minkowski, 3+1D Minkowski, and random-DAG controls.
    """
    rows = []
    profile_rows = []
    for n in sizes:
        # Chain once per size.
        c, a, b = make_chain_interval_causet(int(n))
        row, prof = fingerprint_fixed_endpoint_causet(c, a, b, "chain_1D", f"chain_n{n}", n_profile_bins, {"n_interior_target": int(n), "rep": 0})
        rows.append(row); profile_rows.extend(prof)

        for rep in range(int(reps_per_size)):
            base_seed = int(seed + 100000 * rep + n)
            for d, label in [(3, "minkowski_2p1D"), (4, "minkowski_3p1D")]:
                c, a, b = make_minkowski_interval_causet(int(n), d=d, seed=base_seed + 10 * d)
                row, prof = fingerprint_fixed_endpoint_causet(c, a, b, label, f"{label}_n{n}_rep{rep}", n_profile_bins, {"n_interior_target": int(n), "rep": rep, "control_d": d})
                rows.append(row); profile_rows.extend(prof)
            c, a, b = make_random_dag_interval_causet(int(n), p=random_dag_p, seed=base_seed + 999)
            row, prof = fingerprint_fixed_endpoint_causet(c, a, b, "random_DAG", f"random_n{n}_rep{rep}", n_profile_bins, {"n_interior_target": int(n), "rep": rep, "random_dag_p": random_dag_p})
            rows.append(row); profile_rows.extend(prof)
    return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def extract_diamond_fingerprints(
    causet,
    diamond_df: pd.DataFrame,
    model: str = "foam",
    n_profile_bins: int = 21,
    max_diamonds: Optional[int] = None,
    gap_bin_filter: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if diamond_df is None or len(diamond_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    df = diamond_df.copy()
    if gap_bin_filter is not None and "gap_bin" in df.columns:
        df = df[df["gap_bin"].isin(list(gap_bin_filter))].copy()
    if max_diamonds is not None:
        df = df.head(int(max_diamonds)).copy()

    rows = []
    profile_rows = []
    for k, row0 in enumerate(df.itertuples(index=False)):
        strict_bits = int(getattr(row0, "interval_bits"))
        a_index = int(getattr(row0, "a_index"))
        b_index = int(getattr(row0, "b_index"))
        meta = {}
        for col in ["gap_bin", "gap_lo", "gap_hi", "gap_mid", "ep_a", "ep_b", "epoch_gap", "V_strict_prescan"]:
            if hasattr(row0, col):
                meta[col] = getattr(row0, col)
        rid = f"{model}_{k}"
        frow, prof = fingerprint_interval(causet, strict_bits, (a_index, b_index), model, rid, n_profile_bins=n_profile_bins, meta=meta)
        rows.append(frow); profile_rows.extend(prof)
    return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def summarize_interval_fingerprints(interval_df: pd.DataFrame, target_spacetime_dim: float = 3.0) -> pd.DataFrame:
    rows = []
    if interval_df is None or len(interval_df) == 0:
        return pd.DataFrame()
    for model, sub in interval_df.groupby("model", dropna=False):
        h = sub["h_inclusive"].to_numpy(dtype=float)
        V = sub["V_strict"].to_numpy(dtype=float)
        mid = sub["mid_layer_size"].to_numpy(dtype=float)
        mx = sub["max_layer_size"].to_numpy(dtype=float)
        vol = sr_loglog_fit(h, V)
        midfit = sr_loglog_fit(h, mid)
        maxfit = sr_loglog_fit(h, mx)
        rows.append({
            "model": model,
            "n_intervals": int(len(sub)),
            "V_med": float(np.nanmedian(sub["V_strict"])),
            "h_med": float(np.nanmedian(sub["h_inclusive"])),
            "h_min": float(np.nanmin(sub["h_inclusive"])),
            "h_max": float(np.nanmax(sub["h_inclusive"])),
            "r_med": float(np.nanmedian(sub["r_strict"])),
            "MM_med": float(np.nanmedian(sub["MM_dim_strict"])),
            "MM_mean": float(np.nanmean(sub["MM_dim_strict"])),
            "volume_slope": vol["slope"],
            "volume_r2": vol["r2"],
            "mid_layer_slope": midfit["slope"],
            "mid_layer_r2": midfit["r2"],
            "max_layer_slope": maxfit["slope"],
            "max_layer_r2": maxfit["r2"],
            "target_dim": float(target_spacetime_dim),
            "target_r": float(r_of_d(target_spacetime_dim)),
            "abs_MM_err_to_3": abs(float(np.nanmedian(sub["MM_dim_strict"])) - 3.0),
            "abs_volume_err_to_3": abs(vol["slope"] - 3.0) if np.isfinite(vol["slope"]) else np.nan,
            "abs_mid_err_to_2": abs(midfit["slope"] - 2.0) if np.isfinite(midfit["slope"]) else np.nan,
            "abs_max_err_to_2": abs(maxfit["slope"] - 2.0) if np.isfinite(maxfit["slope"]) else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["sr_score_2p1D"] = (
            out["abs_MM_err_to_3"]
            + out["abs_volume_err_to_3"]
            + 0.75 * out["abs_mid_err_to_2"]
            + 0.75 * out["abs_max_err_to_2"]
            + np.maximum(0.0, 0.95 - out["volume_r2"].fillna(0.0))
            + np.maximum(0.0, 0.90 - out["mid_layer_r2"].fillna(0.0))
            + np.maximum(0.0, 0.90 - out["max_layer_r2"].fillna(0.0))
        )
        out = out.sort_values("sr_score_2p1D").reset_index(drop=True)
    return out


def average_profiles(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df is None or len(profile_df) == 0:
        return pd.DataFrame()
    avg = (
        profile_df.groupby(["model", "profile_bin", "x_norm"], as_index=False)
        .agg(profile_mean=("profile_area_norm", "mean"), profile_std=("profile_area_norm", "std"), n=("profile_area_norm", "size"))
    )
    # Normalize the mean profile again per model to remove small numerical differences.
    parts = []
    for model, sub in avg.groupby("model"):
        sub = sub.copy()
        sm = float(sub["profile_mean"].sum())
        if sm > 0:
            sub["profile_mean_area_norm"] = sub["profile_mean"] / sm
        else:
            sub["profile_mean_area_norm"] = sub["profile_mean"]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else avg


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    ps = p.sum(); qs = q.sum()
    if ps <= 0 or qs <= 0:
        return np.nan
    p = p / ps; q = q / qs
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return float(math.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def profile_distance_table(avg_profiles: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    if avg_profiles is None or len(avg_profiles) == 0:
        return pd.DataFrame()
    pivot = avg_profiles.pivot_table(index="profile_bin", columns="model", values="profile_mean_area_norm", fill_value=0.0)
    if reference_model not in pivot.columns:
        raise ValueError(f"reference_model {reference_model!r} not in profile table")
    ref = pivot[reference_model].to_numpy(dtype=float)
    rows = []
    for model in pivot.columns:
        arr = pivot[model].to_numpy(dtype=float)
        rows.append({
            "reference_model": reference_model,
            "model": model,
            "l1_distance": float(np.sum(np.abs(arr - ref))),
            "l2_distance": float(np.sqrt(np.sum((arr - ref) ** 2))),
            "js_distance": _js_distance(arr, ref),
        })
    return pd.DataFrame(rows).sort_values("js_distance").reset_index(drop=True)


def choose_local_gap_window(
    binned: pd.DataFrame,
    window_sizes: Sequence[int] = (3, 4, 5, 6),
    target_dim: float = 3.0,
    target_spatial_dim: float = 2.0,
) -> pd.DataFrame:
    """
    Choose contiguous gap-bin windows using only binned scaling diagnostics.
    This is meant to avoid hand-picking the SR shoulder.
    """
    if binned is None or len(binned) == 0:
        return pd.DataFrame()
    df = binned.reset_index(drop=True).copy()
    rows = []
    for w in window_sizes:
        if w > len(df):
            continue
        for i in range(0, len(df) - w + 1):
            sub = df.iloc[i:i+w]
            h = sub["h_med"].to_numpy(dtype=float)
            vol = sr_loglog_fit(h, sub["V_med"].to_numpy(dtype=float))
            mid = sr_loglog_fit(h, sub["mid_layer_med"].to_numpy(dtype=float))
            mx = sr_loglog_fit(h, sub["max_layer_med"].to_numpy(dtype=float))
            mm_med = float(np.nanmedian(sub["MM_med"]))
            r_med = float(np.nanmedian(sub["r_med"]))
            score = (
                abs(mm_med - target_dim)
                + abs(vol["slope"] - target_dim)
                + 0.75 * abs(mid["slope"] - target_spatial_dim)
                + 0.75 * abs(mx["slope"] - target_spatial_dim)
                + max(0.0, 0.95 - (vol["r2"] if np.isfinite(vol["r2"]) else 0.0))
                + max(0.0, 0.90 - (mid["r2"] if np.isfinite(mid["r2"]) else 0.0))
                + max(0.0, 0.90 - (mx["r2"] if np.isfinite(mx["r2"]) else 0.0))
            )
            rows.append({
                "start_bin": int(i),
                "end_bin": int(i + w - 1),
                "n_bins": int(w),
                "gap_lo": float(sub["gap_lo"].iloc[0]) if "gap_lo" in sub else np.nan,
                "gap_hi": float(sub["gap_hi"].iloc[-1]) if "gap_hi" in sub else np.nan,
                "h_min": float(np.nanmin(h)),
                "h_max": float(np.nanmax(h)),
                "MM_med": mm_med,
                "r_med": r_med,
                "volume_slope": vol["slope"],
                "volume_r2": vol["r2"],
                "mid_layer_slope": mid["slope"],
                "mid_layer_r2": mid["r2"],
                "max_layer_slope": mx["slope"],
                "max_layer_r2": mx["r2"],
                "score_to_2p1D_SR": float(score),
                "gap_bin_filter": tuple(int(x) for x in sub["gap_bin"].tolist()) if "gap_bin" in sub else tuple(range(i, i+w)),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("score_to_2p1D_SR").reset_index(drop=True)
    return out


def run_foam_sr_sample(
    cap: int = 256,
    seed: int = 101,
    target_basin_splits: int = 10000,
    ep_lo: Optional[int] = 9,
    ep_hi: Optional[int] = 33,
    gap_bins: Sequence[Tuple[int, int]] = ((2, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 20), (20, 24)),
    n_endpoint_samples_per_bin: int = 10000,
    keep_top_per_bin: int = 100,
    n_profile_bins: int = 21,
    choose_window: bool = True,
    explicit_gap_bin_filter: Optional[Sequence[int]] = None,
):
    """
    Grow one foam run, sample broad multi-gap diamonds, choose an SR-like local window,
    and extract interval fingerprints for both all sampled diamonds and the selected window.
    """
    run = grow_typed_foam_causet_spatial(
        target_basin_splits=int(target_basin_splits),
        seed=int(seed),
        scheduler="capped",
        max_splits_per_epoch=int(cap),
        max_ticks_per_epoch=int(cap),
        snapshot_every=10,
    )
    summary, binned, scaling, diamonds, rg2, basin = run_broad_multigap_causal_scaling(
        run,
        ep_lo=ep_lo,
        ep_hi=ep_hi,
        gap_bins=gap_bins,
        n_endpoint_samples_per_bin=int(n_endpoint_samples_per_bin),
        keep_top_per_bin=int(keep_top_per_bin),
        min_size=2,
        seed=int(seed),
        target_dim=3.0,
    )
    local = choose_local_gap_window(binned)
    if explicit_gap_bin_filter is not None:
        selected_bins = tuple(int(x) for x in explicit_gap_bin_filter)
    elif choose_window and len(local):
        selected_bins = tuple(int(x) for x in local.iloc[0]["gap_bin_filter"])
    else:
        selected_bins = None

    all_fp, all_prof = extract_diamond_fingerprints(rg2, diamonds, model=f"foam_cap{cap}_all", n_profile_bins=n_profile_bins)
    if selected_bins is not None:
        win_fp, win_prof = extract_diamond_fingerprints(rg2, diamonds, model=f"foam_cap{cap}_SR_window", n_profile_bins=n_profile_bins, gap_bin_filter=selected_bins)
    else:
        win_fp, win_prof = pd.DataFrame(), pd.DataFrame()
    return {
        "run": run,
        "rg2": rg2,
        "basin": basin,
        "summary": summary,
        "binned": binned,
        "scaling": scaling,
        "diamonds": diamonds,
        "local_windows": local,
        "selected_gap_bins": selected_bins,
        "foam_all_intervals": all_fp,
        "foam_all_profiles": all_prof,
        "foam_window_intervals": win_fp,
        "foam_window_profiles": win_prof,
    }


def plot_layer_profiles(avg_profiles: pd.DataFrame, outfile: Optional[str] = None, models: Optional[Sequence[str]] = None):
    import matplotlib.pyplot as plt
    df = avg_profiles.copy()
    if models is not None:
        df = df[df["model"].isin(list(models))].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("x_norm")
        ax.plot(sub["x_norm"], sub["profile_mean_area_norm"], marker="o", label=model)
    ax.set_xlabel("normalized rank in interval")
    ax.set_ylabel("mean layer profile, area-normalized")
    ax.set_title("SR interval fingerprint: normalized layer profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=180)
    return fig, ax


def plot_scaling(interval_df: pd.DataFrame, outfile: Optional[str] = None, models: Optional[Sequence[str]] = None):
    import matplotlib.pyplot as plt
    df = interval_df.copy()
    if models is not None:
        df = df[df["model"].isin(list(models))].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        # Bin by height to reduce overplotting.
        b = sub.groupby("h_inclusive", as_index=False).agg(V_med=("V_strict", "median"))
        ax.plot(b["h_inclusive"], b["V_med"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("proper-time proxy h (longest-chain height)")
    ax.set_ylabel("strict interval volume V")
    ax.set_title("SR interval volume scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=180)
    return fig, ax


def plot_antichain_scaling(interval_df: pd.DataFrame, outfile: Optional[str] = None, models: Optional[Sequence[str]] = None):
    import matplotlib.pyplot as plt
    df = interval_df.copy()
    if models is not None:
        df = df[df["model"].isin(list(models))].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        b = sub.groupby("h_inclusive", as_index=False).agg(A_mid=("mid_layer_size", "median"), A_max=("max_layer_size", "median"))
        ax.plot(b["h_inclusive"], b["A_mid"], marker="o", label=f"{model}: mid")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("proper-time proxy h (longest-chain height)")
    ax.set_ylabel("mid-layer antichain size")
    ax.set_title("SR spatial cross-section scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=180)
    return fig, ax


def write_sr_report(summary_df: pd.DataFrame, profile_dist: pd.DataFrame, outfile: str):
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# DEU triangular foam SR fingerprint audit\n")
    lines.append("\n## Summary table\n")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("\n\n## Layer-profile distances\n")
    lines.append(profile_dist.to_markdown(index=False))
    lines.append("\n\nInterpretation rule: the SR-compatible result is strongest if the selected foam window is closer to `minkowski_2p1D` than to chain, random-DAG, and 3+1D controls, while also showing MM≈3, V~h^3, and A~h^2.\n")
    Path(outfile).write_text("\n".join(lines), encoding="utf-8")


def interpret_sr_fingerprint(summary_df: pd.DataFrame, dist_to_foam: pd.DataFrame, foam_model: str) -> str:
    """Conservative pass/fail text for the selected foam SR window."""
    if summary_df is None or len(summary_df) == 0 or foam_model not in set(summary_df["model"]):
        return "NO FOAM RESULT: selected foam model is missing."
    row = summary_df[summary_df["model"] == foam_model].iloc[0]
    nearest = None
    if dist_to_foam is not None and len(dist_to_foam):
        nonself = dist_to_foam[dist_to_foam["model"] != foam_model].copy()
        if len(nonself):
            nearest = str(nonself.iloc[0]["model"])
    mm_ok = abs(float(row["MM_med"]) - 3.0) <= 0.35
    vol_ok = abs(float(row["volume_slope"]) - 3.0) <= 0.35 and float(row["volume_r2"]) >= 0.95
    mid_ok = abs(float(row["mid_layer_slope"]) - 2.0) <= 0.40 and float(row["mid_layer_r2"]) >= 0.90
    max_ok = abs(float(row["max_layer_slope"]) - 2.0) <= 0.40 and float(row["max_layer_r2"]) >= 0.90
    nearest_ok = nearest == "minkowski_2p1D"
    if mm_ok and vol_ok and (mid_ok or max_ok) and nearest_ok:
        return "PASS / PROMISING: selected foam window matches flat 2+1D SR interval fingerprints better than the tested controls."
    if mm_ok and vol_ok and (mid_ok or max_ok):
        return f"MIXED: scaling exponents are SR-like, but nearest layer-profile control is {nearest!r}, not minkowski_2p1D."
    if nearest_ok and (mm_ok or vol_ok):
        return "MIXED: layer profile is closest to 2+1D Minkowski, but one or more scaling exponents fail."
    return "FAIL / NOT YET SR-LIKE: selected foam window does not pass the compact flat-SR fingerprint screen."


def interpret_sr_fingerprint_v2(
    summary_df: pd.DataFrame,
    dist_to_foam: pd.DataFrame,
    foam_model: str,
    selected_local_window: Optional[pd.Series] = None,
) -> str:
    """
    Conservative interpretation that uses binned local-window scaling when available
    and ignores other foam-family labels when ranking external controls.
    """
    if summary_df is None or len(summary_df) == 0 or foam_model not in set(summary_df["model"]):
        return "NO FOAM RESULT: selected foam model is missing."

    nearest_control = None
    if dist_to_foam is not None and len(dist_to_foam):
        nonself = dist_to_foam[(dist_to_foam["model"] != foam_model) & (~dist_to_foam["model"].astype(str).str.startswith("foam_"))].copy()
        if len(nonself):
            nearest_control = str(nonself.iloc[0]["model"])

    # Prefer binned local-window diagnostics; individual-diamond fits are noisier.
    if selected_local_window is not None:
        mm = float(selected_local_window.get("MM_med", np.nan))
        vol = float(selected_local_window.get("volume_slope", np.nan))
        vol_r2 = float(selected_local_window.get("volume_r2", np.nan))
        mid = float(selected_local_window.get("mid_layer_slope", np.nan))
        mid_r2 = float(selected_local_window.get("mid_layer_r2", np.nan))
        mx = float(selected_local_window.get("max_layer_slope", np.nan))
        mx_r2 = float(selected_local_window.get("max_layer_r2", np.nan))
        source = "binned selected local window"
    else:
        row = summary_df[summary_df["model"] == foam_model].iloc[0]
        mm = float(row["MM_med"])
        vol = float(row["volume_slope"])
        vol_r2 = float(row["volume_r2"])
        mid = float(row["mid_layer_slope"])
        mid_r2 = float(row["mid_layer_r2"])
        mx = float(row["max_layer_slope"])
        mx_r2 = float(row["max_layer_r2"])
        source = "individual-diamond summary"

    mm_ok = np.isfinite(mm) and abs(mm - 3.0) <= 0.35
    vol_ok = np.isfinite(vol) and abs(vol - 3.0) <= 0.35 and (not np.isfinite(vol_r2) or vol_r2 >= 0.95)
    mid_ok = np.isfinite(mid) and abs(mid - 2.0) <= 0.40 and (not np.isfinite(mid_r2) or mid_r2 >= 0.90)
    max_ok = np.isfinite(mx) and abs(mx - 2.0) <= 0.40 and (not np.isfinite(mx_r2) or mx_r2 >= 0.90)
    nearest_ok = nearest_control == "minkowski_2p1D"

    details = (
        f"[{source}] MM={mm:.3f}, V-slope={vol:.3f}, "
        f"mid-slope={mid:.3f}, max-slope={mx:.3f}; nearest external profile control={nearest_control!r}."
    )

    if mm_ok and vol_ok and (mid_ok or max_ok) and nearest_ok:
        return "PASS / PROMISING: selected foam window passes the compact flat-2+1D SR fingerprint screen. " + details
    if mm_ok and vol_ok and (mid_ok or max_ok):
        return "MIXED: scaling is SR-like, but layer-profile nearest-control test is not clean. " + details
    if nearest_ok and (mm_ok or vol_ok):
        return "MIXED: layer profile is closest to 2+1D Minkowski, but one or more scaling diagnostics fail. " + details
    return "FAIL / NOT YET SR-LIKE: selected foam window does not pass the compact flat-SR fingerprint screen. " + details
