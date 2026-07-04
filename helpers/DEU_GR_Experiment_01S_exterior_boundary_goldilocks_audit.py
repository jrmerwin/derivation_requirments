"""
DEU GR Experiment 01S: Exterior-Boundary Goldilocks Stitch Audit

Purpose
-------
The 01Q coherent stitch run produced a large positive circumference deficit in
auto source mode, but the 01R anchor-locked reanalysis collapsed to exactly zero
Delta C because center/landmark sources can become detached, saturated, or fail
to sample the exterior annuli when the stitched graph is strongly starved.

This helper measures circles from an intrinsic, persistent source: the exterior
boundary of the tagged defect region. Source faces are active non-defect faces
adjacent to one or more tagged defect faces. This is the "throat mouth" of the
stitched defect: it persists even when the center node is swallowed.

It also records support diagnostics for every radial band, so zero C is treated
as "no supported annulus" unless the annulus actually contains faces.

Main functions
--------------
reanalyze_stitch_exterior_boundary(result, OUT, label="...", ...)
    Reanalyze an existing 01Q coherent_stitch result object.

run_goldilocks_stitch_sweep(...)
    Run several fixed-epoch ensembles and reanalyze them with the exterior
    boundary source. Each final_epoch is compared only within itself, preserving
    coordinate-time snapshots while searching for a supported Goldilocks regime.

Dependencies
------------
Execute/load DEU_GR_Experiment_01Q_coherent_topology_stitch_layer3.py first.
Requires _weighted_face_adj and grow_fixed_epoch_coherent_topology_stitch_defect.
"""

from pathlib import Path
import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sem_01S(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _frac_pos_01S(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) == 0:
        return np.nan
    return float((vals > 0).mean())


def _ols_01S(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or len(np.unique(x)) < 2:
        return {"n": int(len(x)), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
    }


def _largest_component_nodes_01S(wadj):
    unseen = set(wadj)
    comps = []
    while unseen:
        s = next(iter(unseen))
        unseen.remove(s)
        q = [s]
        comp = {s}
        while q:
            u = q.pop()
            for v, _w in wadj.get(u, ()): 
                if v in unseen:
                    unseen.remove(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps[0] if comps else set()


def _source_faces_exterior_boundary_01S(snapshot, wadj, nodes):
    """Non-defect faces touching tagged defect faces in the weighted component."""
    face_defect = getattr(snapshot, "face_defect", {})
    tagged = {f for f in nodes if bool(face_defect.get(f, False))}
    exterior = set()
    for f in tagged:
        for n, _w in wadj.get(f, ()): 
            if n in nodes and not bool(face_defect.get(n, False)):
                exterior.add(n)
    return sorted(exterior, key=lambda x: repr(x)), tagged


def _source_faces_tagged_boundary_01S(snapshot, wadj, nodes):
    """Tagged faces adjacent to non-defect exterior; diagnostic inner throat source."""
    face_defect = getattr(snapshot, "face_defect", {})
    tagged_boundary = set()
    for f in nodes:
        if not bool(face_defect.get(f, False)):
            continue
        if any((n in nodes and not bool(face_defect.get(n, False))) for n, _w in wadj.get(f, ())):
            tagged_boundary.add(f)
    return sorted(tagged_boundary, key=lambda x: repr(x))


def _source_faces_for_mode_01S(snapshot, wadj, nodes, source_mode):
    mode = str(source_mode)
    exterior, tagged = _source_faces_exterior_boundary_01S(snapshot, wadj, nodes)
    tagged_boundary = _source_faces_tagged_boundary_01S(snapshot, wadj, nodes)

    if mode == "exterior_boundary":
        source = exterior
    elif mode == "tagged_boundary":
        source = tagged_boundary
    elif mode == "boundary_union":
        source = sorted(set(exterior) | set(tagged_boundary), key=lambda x: repr(x))
    elif mode == "tagged_all":
        source = sorted(tagged, key=lambda x: repr(x))
    else:
        raise ValueError(
            "source_mode must be one of: exterior_boundary, tagged_boundary, boundary_union, tagged_all"
        )

    return source, {
        "source_mode": mode,
        "exterior_boundary_faces": int(len(exterior)),
        "tagged_boundary_faces": int(len(tagged_boundary)),
        "tagged_faces_total": int(len(tagged)),
    }


def exterior_boundary_weighted_curve_01S(
    snapshot,
    *,
    radius_edges=None,
    source_mode="exterior_boundary",
    min_source_faces=1,
):
    """Weighted ball/circumference curve from the defect exterior boundary."""
    if "_weighted_face_adj" not in globals():
        raise RuntimeError("_weighted_face_adj not found. Execute/load 01Q and src/deu_exp456_minimal.py first.")
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    wadj, areas, depths = _weighted_face_adj(snapshot, component="largest")
    nodes = _largest_component_nodes_01S(wadj)
    if not nodes:
        raise RuntimeError("No weighted largest component found.")
    # Restrict adjacency to largest component.
    wadj = {u: [(v, w) for v, w in wadj.get(u, ()) if v in nodes] for u in nodes}
    areas = {f: float(areas[f]) for f in nodes if f in areas}
    depths = {f: int(depths[f]) for f in nodes if f in depths}
    lengths = {f: float((3.0 ** -0.5) ** depths[f]) for f in nodes if f in depths}

    source_faces, source_info = _source_faces_for_mode_01S(snapshot, wadj, nodes, source_mode)
    source_faces = [f for f in source_faces if f in nodes and f in lengths]

    if len(source_faces) < int(min_source_faces):
        raise RuntimeError(
            f"Not enough source faces for source_mode={source_mode}: {len(source_faces)} < {min_source_faces}. "
            f"source_info={source_info}"
        )

    dist = {s: 0.5 * lengths[s] for s in source_faces}
    heap = [(d, s) for s, d in dist.items()]
    heapq.heapify(heap)
    max_dist = float(np.max(radius_edges))

    while heap:
        du, u = heapq.heappop(heap)
        if du != dist.get(u):
            continue
        if du > max_dist:
            continue
        for v, w in wadj.get(u, ()): 
            nd = du + float(w)
            if nd <= max_dist and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    faces = list(dist.keys())
    dists = np.array([dist[f] for f in faces], dtype=float)
    area_vals = np.array([areas[f] for f in faces], dtype=float)

    component_area = float(sum(areas.values()))
    tagged_area = float(sum(areas[f] for f in nodes if bool(getattr(snapshot, "face_defect", {}).get(f, False)) and f in areas))

    rows = []
    for i in range(1, len(radius_edges)):
        R0 = float(radius_edges[i - 1])
        R1 = float(radius_edges[i])
        if R1 <= R0:
            continue
        dR = R1 - R0
        R_mid = 0.5 * (R0 + R1)
        in_ball = dists <= R1
        in_ann = (dists > R0) & (dists <= R1)
        A_ball = float(area_vals[in_ball].sum())
        A_ann = float(area_vals[in_ann].sum())
        C_est = A_ann / dR if dR > 0 else np.nan
        rows.append({
            "bin": int(i),
            "R0": R0,
            "R1": R1,
            "R_mid": R_mid,
            "dR": dR,
            "A_ball": A_ball,
            "A_annulus": A_ann,
            "C_est": C_est,
            "C_over_R": C_est / R_mid if R_mid > 0 else np.nan,
            "A_over_R2": A_ball / (R1 ** 2) if R1 > 0 else np.nan,
            "n_faces_ball": int(in_ball.sum()),
            "n_faces_annulus": int(in_ann.sum()),
            "annulus_supported": bool(in_ann.sum() > 0 and A_ann > 0),
            "source_mode": str(source_mode),
            "source_faces_count": int(len(source_faces)),
            **source_info,
            "component_weighted_area": component_area,
            "tagged_defect_area_total": tagged_area,
            "largest_component_faces": int(len(nodes)),
            "depth_min": int(min(depths.values())) if depths else np.nan,
            "depth_med": float(np.median(list(depths.values()))) if depths else np.nan,
            "depth_max": int(max(depths.values())) if depths else np.nan,
        })
    return pd.DataFrame(rows)


def _summarize_exterior_paired_01S(
    paired,
    run_df,
    *,
    radial_windows=None,
    require_supported=True,
):
    if radial_windows is None:
        radial_windows = {
            "inner_0p25_0p60": (0.25, 0.60),
            "mid_0p60_1p00": (0.60, 1.00),
            "outer_1p00_1p60": (1.00, 1.60),
            "wide_0p25_1p60": (0.25, 1.60),
        }
    rows = []
    for wname, (rlo, rhi) in radial_windows.items():
        d0 = paired[(paired["R_mid"] >= rlo) & (paired["R_mid"] <= rhi)].copy()
        if d0.empty:
            continue
        for (seed, m), g0 in d0.groupby(["seed", "m_defects"]):
            g = g0.copy()
            if require_supported:
                # Keep bins where both the vacuum and the m-run have actual annulus support.
                g = g[(g["annulus_supported"]) & (g["annulus_supported_m0"])]
            support_total = int(len(g0))
            support_used = int(len(g))
            if support_used == 0:
                rows.append({
                    "radial_window": wname,
                    "seed": int(seed),
                    "m_defects": int(m),
                    "median_delta_A": np.nan,
                    "median_delta_A_over_R2": np.nan,
                    "median_delta_C": np.nan,
                    "median_delta_C_over_R": np.nan,
                    "mean_delta_C": np.nan,
                    "support_bins_total": support_total,
                    "support_bins_used": 0,
                    "support_fraction_used": 0.0,
                    "median_n_faces_annulus": 0.0,
                    "median_n_faces_annulus_m0": 0.0,
                    "source_faces_count": float(g0["source_faces_count"].median()) if len(g0) else np.nan,
                    "source_mode_used": str(g0["source_mode"].mode().iloc[0]) if len(g0["source_mode"].mode()) else "",
                })
                continue
            rows.append({
                "radial_window": wname,
                "seed": int(seed),
                "m_defects": int(m),
                "median_delta_A": float(g["delta_A_vs_m0"].median()),
                "median_delta_A_over_R2": float(g["delta_A_over_R2_vs_m0"].median()),
                "median_delta_C": float(g["delta_C_vs_m0"].median()),
                "median_delta_C_over_R": float(g["delta_C_over_R_vs_m0"].median()),
                "mean_delta_C": float(g["delta_C_vs_m0"].mean()),
                "support_bins_total": support_total,
                "support_bins_used": support_used,
                "support_fraction_used": float(support_used / support_total) if support_total else np.nan,
                "median_n_faces_annulus": float(g["n_faces_annulus"].median()),
                "median_n_faces_annulus_m0": float(g["n_faces_annulus_m0"].median()),
                "source_faces_count": float(g["source_faces_count"].median()),
                "source_mode_used": str(g["source_mode"].mode().iloc[0]) if len(g["source_mode"].mode()) else "",
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    meta = [c for c in [
        "seed", "m_defects", "forced_defect_splits", "forced_stitch_splits",
        "omitted_child_faces", "omitted_weighted_area", "defect_active_faces_final",
        "basin_splits", "final_active_faces", "final_nodes", "epochs", "final_epoch",
        "stitch_omit_index",
    ] if c in run_df.columns]
    per_seed = per_seed.merge(run_df[meta], on=["seed", "m_defects"], how="left")

    ens_rows = []
    for (wname, m), g in per_seed.groupby(["radial_window", "m_defects"]):
        row = {"radial_window": wname, "m_defects": int(m), "seeds": int(g["seed"].nunique())}
        for col in [
            "median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R", "mean_delta_C",
            "support_fraction_used", "median_n_faces_annulus", "median_n_faces_annulus_m0", "source_faces_count",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_mean_over_seeds"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sem_over_seeds"] = _sem_01S(vals)
            if col.startswith("median_delta") or col.startswith("mean_delta"):
                row[f"{col}_seed_frac_positive"] = _frac_pos_01S(vals)
        for col in ["forced_stitch_splits", "omitted_weighted_area", "defect_active_faces_final", "basin_splits", "final_active_faces"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        row["source_modes"] = ",".join(sorted(set(str(x) for x in g["source_mode_used"].dropna())))
        ens_rows.append(row)
    ensemble_summary = pd.DataFrame(ens_rows)

    fit_rows = []
    nz = per_seed[per_seed["m_defects"] > 0].copy()
    nz = nz[nz["support_fraction_used"] > 0]
    for wname, gd in nz.groupby("radial_window"):
        for min_m in [1, 2, 4, 6, 8, 10, 12, 16]:
            h = gd[gd["m_defects"] >= min_m].copy()
            if h.empty:
                continue
            for x in ["m_defects", "forced_stitch_splits", "omitted_weighted_area", "defect_active_faces_final"]:
                if x not in h.columns:
                    continue
                for y in ["median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R", "mean_delta_C"]:
                    if y not in h.columns:
                        continue
                    fit_rows.append({"radial_window": wname, "min_m": int(min_m), "x": x, "y": y, **_ols_01S(h[x], h[y])})
    fit_table = pd.DataFrame(fit_rows)

    verdict_rows = []
    core = ensemble_summary[
        ensemble_summary["radial_window"].isin(["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"])
        & (ensemble_summary["m_defects"] > 0)
    ].copy()
    if not core.empty:
        core["supported"] = core["support_fraction_used_mean_over_seeds"] >= 0.5
        core["C_pos"] = core["median_delta_C_mean_over_seeds"] > 0
        core["A_pos"] = core["median_delta_A_over_R2_mean_over_seeds"] > 0
        core["C_seed_good"] = core.get("median_delta_C_seed_frac_positive", 0) >= (2 / 3)
        core["A_seed_good"] = core.get("median_delta_A_over_R2_seed_frac_positive", 0) >= (2 / 3)
        core["cone_row"] = core["supported"] & core["C_pos"] & core["A_pos"] & core["C_seed_good"] & core["A_seed_good"]
        core["throat_row"] = core["supported"] & core["C_pos"] & (~core["A_pos"]) & core["C_seed_good"]
        core["unsupported_row"] = ~core["supported"]
        high = core[core["m_defects"] >= 4]
        n_high = int(len(high))
        cone_high = int(high["cone_row"].sum()) if n_high else 0
        throat_high = int(high["throat_row"].sum()) if n_high else 0
        unsupported_high = int(high["unsupported_row"].sum()) if n_high else 0
        if n_high and cone_high >= max(1, math.ceil(0.5 * n_high)):
            verdict = "EXTERIOR_BOUNDARY_CONE_SIGNAL"
        elif n_high and throat_high >= max(1, math.ceil(0.5 * n_high)):
            verdict = "EXTERIOR_BOUNDARY_THROAT_SIGNAL"
        elif n_high and unsupported_high >= max(1, math.ceil(0.5 * n_high)):
            verdict = "SUPPORT_LIMITED__NO_LAYER3_VERDICT"
        else:
            verdict = "EXTERIOR_BOUNDARY_WEAK_OR_NULL"
        verdict_rows.append({
            "verdict": verdict,
            "n_core_rows": int(len(core)),
            "n_high_rows": n_high,
            "n_cone_high_rows": cone_high,
            "n_throat_high_rows": throat_high,
            "n_unsupported_high_rows": unsupported_high,
        })
    verdict = pd.DataFrame(verdict_rows)
    return per_seed, ensemble_summary, fit_table, verdict


def reanalyze_stitch_exterior_boundary(
    result,
    *,
    OUT=None,
    label="exterior_boundary_stitch",
    source_mode="exterior_boundary",
    radius_edges=None,
    radial_windows=None,
    require_supported=True,
    make_plots=True,
):
    """Reanalyze an existing 01Q result with an intrinsic defect-boundary source."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    if "runs" not in result:
        raise KeyError("Expected result['runs'] from 01Q.")

    curve_rows = []
    run_rows = []
    errors = []
    for (seed, m), run in sorted(result["runs"].items()):
        ep = max(run.spatial_snapshots)
        snap = run.spatial_snapshots[ep]
        try:
            curve = exterior_boundary_weighted_curve_01S(
                snap,
                radius_edges=radius_edges,
                source_mode=source_mode,
            )
            curve["seed"] = int(seed)
            curve["m_defects"] = int(m)
            curve["epoch"] = int(ep)
            curve_rows.append(curve)
        except Exception as exc:
            errors.append({"seed": int(seed), "m_defects": int(m), "error": repr(exc)})
            continue
        row = dict(run.stats)
        row["seed"] = int(seed)
        row["m_defects"] = int(m)
        row["final_epoch"] = int(ep)
        row["defect_active_faces_final"] = int(sum(1 for f in snap.active_faces if snap.face_defect.get(f, False)))
        run_rows.append(row)

    if not curve_rows:
        raise RuntimeError(f"No curves computed. errors={errors[:5]}")
    curves = pd.concat(curve_rows, ignore_index=True)
    run_df = pd.DataFrame(run_rows)
    error_df = pd.DataFrame(errors)

    base_cols = [
        "seed", "bin", "R_mid", "A_ball", "A_over_R2", "C_est", "C_over_R", "component_weighted_area",
        "n_faces_annulus", "annulus_supported",
    ]
    base = curves[curves["m_defects"] == 0][base_cols].rename(columns={
        "A_ball": "A_ball_m0",
        "A_over_R2": "A_over_R2_m0",
        "C_est": "C_est_m0",
        "C_over_R": "C_over_R_m0",
        "component_weighted_area": "component_weighted_area_m0",
        "n_faces_annulus": "n_faces_annulus_m0",
        "annulus_supported": "annulus_supported_m0",
    })
    paired = curves.merge(base, on=["seed", "bin", "R_mid"], how="left")
    paired["delta_A_vs_m0"] = paired["A_ball_m0"] - paired["A_ball"]
    paired["delta_A_over_R2_vs_m0"] = paired["A_over_R2_m0"] - paired["A_over_R2"]
    paired["delta_C_vs_m0"] = paired["C_est_m0"] - paired["C_est"]
    paired["delta_C_over_R_vs_m0"] = paired["C_over_R_m0"] - paired["C_over_R"]
    paired["component_area_delta_vs_m0"] = paired["component_weighted_area_m0"] - paired["component_weighted_area"]

    per_seed, ensemble_summary, fit_table, verdict = _summarize_exterior_paired_01S(
        paired,
        run_df,
        radial_windows=radial_windows,
        require_supported=require_supported,
    )

    stem = f"{label}_{source_mode}".replace(" ", "_")
    paths = {
        "curves": OUT / f"{stem}_curves.csv",
        "paired": OUT / f"{stem}_paired.csv",
        "run_summary": OUT / f"{stem}_run_summary.csv",
        "errors": OUT / f"{stem}_errors.csv",
        "per_seed_summary": OUT / f"{stem}_per_seed_summary.csv",
        "ensemble_summary": OUT / f"{stem}_ensemble_summary.csv",
        "fit_table": OUT / f"{stem}_fit_table.csv",
        "verdict": OUT / f"{stem}_verdict.csv",
    }
    curves.to_csv(paths["curves"], index=False)
    paired.to_csv(paths["paired"], index=False)
    run_df.to_csv(paths["run_summary"], index=False)
    error_df.to_csv(paths["errors"], index=False)
    per_seed.to_csv(paths["per_seed_summary"], index=False)
    ensemble_summary.to_csv(paths["ensemble_summary"], index=False)
    fit_table.to_csv(paths["fit_table"], index=False)
    verdict.to_csv(paths["verdict"], index=False)

    if make_plots and not ensemble_summary.empty:
        for metric, ylabel in [
            ("median_delta_C_mean_over_seeds", "Delta C"),
            ("median_delta_A_over_R2_mean_over_seeds", "Delta A/R^2"),
            ("support_fraction_used_mean_over_seeds", "supported annulus fraction"),
        ]:
            if metric not in ensemble_summary.columns:
                continue
            plt.figure(figsize=(7, 4))
            for w in ["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"]:
                d = ensemble_summary[ensemble_summary["radial_window"] == w].copy()
                if d.empty:
                    continue
                plt.plot(d["m_defects"], d[metric], marker="o", linewidth=1, label=w)
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel(ylabel)
            plt.title(f"01S exterior-boundary stitch audit: {ylabel}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            p = OUT / f"{stem}_{metric}.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths[f"plot_{metric}"] = p

    print("\n01S exterior-boundary ensemble summary:")
    try:
        display(ensemble_summary)
    except NameError:
        print(ensemble_summary)
    print("\n01S verdict:")
    try:
        display(verdict)
    except NameError:
        print(verdict)
    print("\nErrors:")
    try:
        display(error_df)
    except NameError:
        print(error_df)
    print("\nWrote:")
    for p in paths.values():
        print(" ", p)

    return {
        "curves": curves,
        "paired": paired,
        "run_summary": run_df,
        "errors": error_df,
        "per_seed_summary": per_seed,
        "ensemble_summary": ensemble_summary,
        "fit_table": fit_table,
        "verdict": verdict,
        "paths": paths,
        "source_mode": source_mode,
    }


def run_goldilocks_stitch_sweep(
    *,
    OUT=None,
    final_epochs=(52, 56, 60),
    cap=512,
    seeds=(101, 202, 303),
    m_values=(0, 4, 6, 8, 10, 12, 16),
    defect_inject_epoch=25,
    stitch_omit_index=1,
    source_mode="exterior_boundary",
    radius_edges=None,
    label="goldilocks_stitch_sweep",
    make_plots=False,
):
    """
    Run fixed-epoch ensembles across a small epoch/mass sweep, then reanalyze
    from the defect exterior boundary. Each final_epoch is internally paired
    against its own m=0 vacuum, preserving coordinate-time comparisons.
    """
    if "grow_fixed_epoch_coherent_topology_stitch_defect" not in globals():
        raise RuntimeError("Load/execute 01Q first; grow_fixed_epoch_coherent_topology_stitch_defect is missing.")
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)

    all_results = {}
    verdict_rows = []
    for ep in final_epochs:
        runs = {}
        print("\n" + "=" * 100)
        print(f"Goldilocks stitch sweep epoch={ep}, cap={cap}, omit={stitch_omit_index}")
        print("=" * 100)
        for seed in seeds:
            for m in m_values:
                print(f"running seed={seed}, m={m}, final_epoch={ep}")
                r = grow_fixed_epoch_coherent_topology_stitch_defect(
                    final_epoch=int(ep),
                    seed=int(seed),
                    max_splits_per_epoch=int(cap),
                    max_ticks_per_epoch=int(cap),
                    m_defects=int(m),
                    defect_inject_epoch=int(defect_inject_epoch),
                    stitch_omit_index=int(stitch_omit_index),
                    snapshot_every=10,
                    record_final=True,
                )
                runs[(int(seed), int(m))] = r
        fake_result = {"runs": runs}
        ep_label = f"{label}_cap{cap}_epoch{ep}_omit{stitch_omit_index}"
        res = reanalyze_stitch_exterior_boundary(
            fake_result,
            OUT=OUT,
            label=ep_label,
            source_mode=source_mode,
            radius_edges=radius_edges,
            make_plots=make_plots,
        )
        all_results[int(ep)] = res
        if not res["verdict"].empty:
            vr = res["verdict"].copy()
            vr["final_epoch"] = int(ep)
            verdict_rows.append(vr)
    sweep_verdict = pd.concat(verdict_rows, ignore_index=True) if verdict_rows else pd.DataFrame()
    sweep_path = OUT / f"{label}_cap{cap}_omit{stitch_omit_index}_{source_mode}_sweep_verdict.csv"
    sweep_verdict.to_csv(sweep_path, index=False)
    print("\nGoldilocks sweep verdicts:")
    try:
        display(sweep_verdict)
    except NameError:
        print(sweep_verdict)
    print("Wrote:", sweep_path)
    return {"by_epoch": all_results, "sweep_verdict": sweep_verdict, "path": sweep_path}
