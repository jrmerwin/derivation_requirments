"""
DEU GR Experiment 01R: Anchor-Locked Stitch Snap Audit

Purpose
-------
Reanalyze an existing coherent topology-stitch ensemble using a fixed source
mode for all masses. This controls the main confound discovered in 01Q:
large-m runs can lose center-node incident active faces and fall back from
center_node to landmark_nodes, exactly where the apparent circumference snap
appears.

Use after running/executing DEU_GR_Experiment_01Q_coherent_topology_stitch_layer3.py
and after creating a result object such as coherent_stitch_cap512.

Main functions
--------------
reanalyze_coherent_stitch_source_mode(result, OUT, label, source_mode="landmark_only")
    Recomputes weighted ball/circumference curves from the stored run snapshots
    with a fixed measurement-source strategy.

summarize_stitch_snap_signal(result, windows=None)
    Summarizes cone/throat/snap evidence and anchor strategy consistency.
"""

from pathlib import Path
import ast
import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_anchor_nodes_01R(stats):
    if not isinstance(stats, dict):
        return []
    if "defect_anchor_nodes" in stats:
        try:
            return [int(x) for x in stats["defect_anchor_nodes"]]
        except Exception:
            pass
    if "defect_anchor_nodes_repr" in stats:
        try:
            val = ast.literal_eval(str(stats["defect_anchor_nodes_repr"]))
            return [int(x) for x in val]
        except Exception:
            pass
    return []


def _sem_01R(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _frac_pos_01R(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) == 0:
        return np.nan
    return float((vals > 0).mean())


def _ols_01R(x, y):
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
        "r2": np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot),
    }


def coherent_stitch_weighted_anchor_curve_locked(
    snapshot,
    *,
    center_node=None,
    radius_edges=None,
    source_mode="landmark_only",
):
    """
    Recompute weighted ball/circumference curve around a fixed anchor source.

    source_mode options:
      - "center_only": active faces incident to the virtual center node only;
        returns empty/error if center is detached.
      - "landmark_only": active faces incident to original anchor triangle nodes
        only. This is usually the best cross-mass control.
      - "center_plus_landmark": union of center-node and landmark-node faces.
      - "auto_center_then_landmark": previous 01Q behavior; useful only for
        diagnosing the confound, not for final evidence.
      - "tagged_only": all tagged defect-lineage faces. Diagnostic only; source
        size changes strongly with m.
    """
    if "_weighted_face_adj" not in globals():
        raise RuntimeError("_weighted_face_adj not found. Load src/deu_exp456_minimal.py or execute 01Q first.")
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)
    if center_node is None:
        center_node = snapshot.stats.get("defect_center_node")
    if center_node is None:
        raise ValueError("No center_node supplied and snapshot.stats has no defect_center_node")
    center_node = int(center_node)
    anchor_nodes = _parse_anchor_nodes_01R(snapshot.stats)

    wadj, areas, depths = _weighted_face_adj(snapshot, component="largest")
    nodes = set(wadj)
    lengths = {f: float((3.0 ** -0.5) ** int(depths[f])) for f in nodes}

    def incident_faces_for_node_set(node_set):
        ns = set(int(x) for x in node_set)
        return [f for f in nodes if snapshot.face_nodes.get(f, frozenset()) & ns]

    center_faces = incident_faces_for_node_set([center_node])
    landmark_faces = incident_faces_for_node_set(anchor_nodes)
    tagged_faces = [f for f in nodes if getattr(snapshot, "face_defect", {}).get(f, False)]

    mode = str(source_mode)
    if mode == "center_only":
        source_faces = list(center_faces)
        strategy = "center_only"
    elif mode == "landmark_only":
        source_faces = list(landmark_faces)
        strategy = "landmark_only"
    elif mode == "center_plus_landmark":
        source_faces = sorted(set(center_faces) | set(landmark_faces), key=lambda x: repr(x))
        strategy = "center_plus_landmark"
    elif mode == "tagged_only":
        source_faces = list(tagged_faces)
        strategy = "tagged_only"
    elif mode == "auto_center_then_landmark":
        if center_faces:
            source_faces = list(center_faces)
            strategy = "auto_center_node"
        elif landmark_faces:
            source_faces = list(landmark_faces)
            strategy = "auto_landmark_nodes"
        elif tagged_faces:
            source_faces = list(tagged_faces)
            strategy = "auto_tagged_fallback"
        else:
            source_faces = []
            strategy = "auto_none"
    else:
        raise ValueError(f"Unknown source_mode: {source_mode}")

    if not source_faces:
        raise RuntimeError(
            f"No usable source faces for source_mode={source_mode}, center_node={center_node}, "
            f"anchor_nodes={anchor_nodes}. center={len(center_faces)}, landmark={len(landmark_faces)}, tagged={len(tagged_faces)}"
        )

    dist = {s: 0.5 * lengths[s] for s in source_faces if s in lengths}
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

    tagged_area = float(sum(areas[f] for f in tagged_faces if f in areas))
    component_area = float(sum(areas.values()))

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
        C_est = A_ann / dR
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
            "source_strategy": strategy,
            "source_mode_requested": mode,
            "source_faces_count": int(len(source_faces)),
            "center_faces_count": int(len(center_faces)),
            "landmark_faces_count": int(len(landmark_faces)),
            "tagged_defect_faces_total": int(len(tagged_faces)),
            "tagged_defect_area_total": tagged_area,
            "component_weighted_area": component_area,
            "center_node": int(center_node),
            "anchor_nodes_repr": repr(tuple(int(x) for x in anchor_nodes)),
            "depth_min": int(min(depths.values())) if depths else np.nan,
            "depth_med": float(np.median(list(depths.values()))) if depths else np.nan,
            "depth_max": int(max(depths.values())) if depths else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_locked_paired_01R(paired, run_df, radial_windows=None):
    if radial_windows is None:
        radial_windows = {
            "inner_0p25_0p60": (0.25, 0.60),
            "mid_0p60_1p00": (0.60, 1.00),
            "outer_1p00_1p60": (1.00, 1.60),
            "wide_0p25_1p60": (0.25, 1.60),
        }
    rows = []
    for wname, (rlo, rhi) in radial_windows.items():
        d = paired[(paired["R_mid"] >= rlo) & (paired["R_mid"] <= rhi)].copy()
        if d.empty:
            continue
        for (seed, m), g in d.groupby(["seed", "m_defects"]):
            rows.append({
                "radial_window": wname,
                "seed": int(seed),
                "m_defects": int(m),
                "median_delta_A": float(g["delta_A_vs_m0"].median()),
                "median_delta_A_over_R2": float(g["delta_A_over_R2_vs_m0"].median()),
                "median_delta_C": float(g["delta_C_vs_m0"].median()),
                "median_delta_C_over_R": float(g["delta_C_over_R_vs_m0"].median()),
                "median_component_area_delta": float(g["component_area_delta_vs_m0"].median()),
                "median_source_faces_count": float(g["source_faces_count"].median()),
                "median_center_faces_count": float(g["center_faces_count"].median()),
                "median_landmark_faces_count": float(g["landmark_faces_count"].median()),
                "median_tagged_defect_faces_total": float(g["tagged_defect_faces_total"].median()),
                "median_tagged_defect_area_total": float(g["tagged_defect_area_total"].median()),
                "source_strategy_mode": str(g["source_strategy"].mode().iloc[0]) if len(g["source_strategy"].mode()) else "",
                "n_annuli": int(len(g)),
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(), pd.DataFrame()

    available_meta = [c for c in [
        "seed", "m_defects", "forced_defect_splits", "forced_stitch_splits",
        "omitted_child_faces", "omitted_weighted_area", "defect_active_faces_final",
        "basin_splits", "final_active_faces", "final_nodes", "epochs", "final_epoch",
        "stitch_omit_index",
    ] if c in run_df.columns]
    per_seed = per_seed.merge(run_df[available_meta], on=["seed", "m_defects"], how="left")

    ens_rows = []
    for (wname, m), g in per_seed.groupby(["radial_window", "m_defects"]):
        row = {"radial_window": wname, "m_defects": int(m), "seeds": int(g["seed"].nunique())}
        for col in [
            "median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R",
            "median_component_area_delta", "median_tagged_defect_area_total",
        ]:
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            row[f"{col}_mean_over_seeds"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sem_over_seeds"] = _sem_01R(vals)
            row[f"{col}_seed_frac_positive"] = _frac_pos_01R(vals)
        for col in ["forced_defect_splits", "forced_stitch_splits", "omitted_child_faces", "omitted_weighted_area", "defect_active_faces_final", "basin_splits", "final_active_faces"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        row["source_strategy_modes"] = ",".join(sorted(set(str(x) for x in g["source_strategy_mode"].dropna())))
        ens_rows.append(row)
    ensemble_summary = pd.DataFrame(ens_rows)

    fit_rows = []
    nonzero = per_seed[per_seed["m_defects"] > 0].copy()
    for wname, gd in nonzero.groupby("radial_window"):
        for min_m in [1, 2, 4, 8, 16]:
            h = gd[gd["m_defects"] >= min_m].copy()
            if h.empty:
                continue
            for x in ["m_defects", "forced_stitch_splits", "omitted_weighted_area", "defect_active_faces_final"]:
                if x not in h.columns:
                    continue
                for y in ["median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R", "median_component_area_delta"]:
                    if y not in h.columns:
                        continue
                    fit_rows.append({"radial_window": wname, "min_m": int(min_m), "x": x, "y": y, **_ols_01R(h[x], h[y])})
    fit_table = pd.DataFrame(fit_rows)
    return per_seed, ensemble_summary, fit_table


def reanalyze_coherent_stitch_source_mode(
    result,
    *,
    OUT=None,
    label="anchor_locked_stitch",
    source_mode="landmark_only",
    radius_edges=None,
    radial_windows=None,
    make_plots=True,
):
    """Recompute curves/deficits from a 01Q result using one fixed source mode."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    if "runs" not in result:
        raise KeyError("result must contain a 'runs' dictionary as returned by 01Q")

    curve_rows = []
    run_rows = []
    for (seed, m), run in sorted(result["runs"].items()):
        ep = max(run.spatial_snapshots)
        snap = run.spatial_snapshots[ep]
        center_node = int(snap.stats["defect_center_node"])
        curve = coherent_stitch_weighted_anchor_curve_locked(
            snap,
            center_node=center_node,
            radius_edges=radius_edges,
            source_mode=source_mode,
        )
        curve["seed"] = int(seed)
        curve["m_defects"] = int(m)
        curve["epoch"] = int(ep)
        curve_rows.append(curve)

        row = dict(run.stats)
        row["seed"] = int(seed)
        row["m_defects"] = int(m)
        row["final_epoch"] = int(ep)
        row["defect_active_faces_final"] = int(sum(1 for f in snap.active_faces if snap.face_defect.get(f, False)))
        row["source_strategy_final_locked"] = str(curve["source_strategy"].iloc[0])
        row["source_faces_count_final_locked"] = int(curve["source_faces_count"].iloc[0])
        row.setdefault("forced_defect_splits", 0)
        row.setdefault("forced_stitch_splits", 0)
        row.setdefault("omitted_weighted_area", 0.0)
        run_rows.append(row)

    curves = pd.concat(curve_rows, ignore_index=True)
    run_df = pd.DataFrame(run_rows)

    base_cols = ["seed", "bin", "R_mid", "A_ball", "A_over_R2", "C_est", "C_over_R", "component_weighted_area"]
    base = curves[curves["m_defects"] == 0][base_cols].rename(columns={
        "A_ball": "A_ball_m0",
        "A_over_R2": "A_over_R2_m0",
        "C_est": "C_est_m0",
        "C_over_R": "C_over_R_m0",
        "component_weighted_area": "component_weighted_area_m0",
    })
    paired = curves.merge(base, on=["seed", "bin", "R_mid"], how="left")
    paired["delta_A_vs_m0"] = paired["A_ball_m0"] - paired["A_ball"]
    paired["delta_A_over_R2_vs_m0"] = paired["A_over_R2_m0"] - paired["A_over_R2"]
    paired["delta_C_vs_m0"] = paired["C_est_m0"] - paired["C_est"]
    paired["delta_C_over_R_vs_m0"] = paired["C_over_R_m0"] - paired["C_over_R"]
    paired["component_area_delta_vs_m0"] = paired["component_weighted_area_m0"] - paired["component_weighted_area"]

    per_seed, ensemble_summary, fit_table = summarize_locked_paired_01R(paired, run_df, radial_windows=radial_windows)
    snap_summary = summarize_stitch_snap_signal({"ensemble_summary": ensemble_summary, "fit_table": fit_table})

    stem = f"{label}_{source_mode}".replace(" ", "_")
    paths = {
        "curves": OUT / f"{stem}_curves.csv",
        "paired": OUT / f"{stem}_paired.csv",
        "run_summary": OUT / f"{stem}_run_summary.csv",
        "per_seed_summary": OUT / f"{stem}_per_seed_summary.csv",
        "ensemble_summary": OUT / f"{stem}_ensemble_summary.csv",
        "fit_table": OUT / f"{stem}_fit_table.csv",
        "snap_summary": OUT / f"{stem}_snap_summary.csv",
    }
    curves.to_csv(paths["curves"], index=False)
    paired.to_csv(paths["paired"], index=False)
    run_df.to_csv(paths["run_summary"], index=False)
    per_seed.to_csv(paths["per_seed_summary"], index=False)
    ensemble_summary.to_csv(paths["ensemble_summary"], index=False)
    fit_table.to_csv(paths["fit_table"], index=False)
    snap_summary.to_csv(paths["snap_summary"], index=False)

    if make_plots and not ensemble_summary.empty:
        for metric, ylabel in [("median_delta_C_mean_over_seeds", "Delta C"), ("median_delta_A_over_R2_mean_over_seeds", "Delta A/R^2")]:
            plt.figure(figsize=(7, 4))
            for w in ["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"]:
                d = ensemble_summary[ensemble_summary["radial_window"] == w].copy()
                if d.empty or metric not in d.columns:
                    continue
                plt.plot(d["m_defects"], d[metric], marker="o", linewidth=1, label=w)
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel(ylabel)
            plt.title(f"Anchor-locked stitch reanalysis: {source_mode}, {ylabel}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            p = OUT / f"{stem}_{metric}.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths[f"plot_{metric}"] = p

    print("\nAnchor-locked ensemble summary:")
    try:
        display(ensemble_summary)
    except NameError:
        print(ensemble_summary)
    print("\nSnap / cone / throat summary:")
    try:
        display(snap_summary)
    except NameError:
        print(snap_summary)
    print("\nWrote:")
    for p in paths.values():
        print(" ", p)

    return {
        "source_mode": source_mode,
        "curves": curves,
        "paired": paired,
        "run_summary": run_df,
        "per_seed_summary": per_seed,
        "ensemble_summary": ensemble_summary,
        "fit_table": fit_table,
        "snap_summary": snap_summary,
        "paths": paths,
    }


def summarize_stitch_snap_signal(result, windows=None):
    """Classify cone/throat/snap evidence from an ensemble summary."""
    if windows is None:
        windows = ["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"]
    ens = result["ensemble_summary"].copy()
    rows = []
    if ens.empty:
        return pd.DataFrame()
    for w in windows:
        d = ens[ens["radial_window"] == w].copy().sort_values("m_defects")
        if d.empty:
            continue
        modes = sorted(set(str(x) for x in d.get("source_strategy_modes", pd.Series(dtype=str)).dropna()))
        strategy_constant = len(modes) <= 1
        nonzero = d[d["m_defects"] > 0].copy()
        high = d[d["m_defects"].isin([8, 16])].copy()
        low = d[d["m_defects"].isin([2, 4])].copy()
        c_high = high["median_delta_C_mean_over_seeds"].mean() if not high.empty else np.nan
        c_low_abs = low["median_delta_C_mean_over_seeds"].abs().mean() if not low.empty else np.nan
        a_high = high["median_delta_A_over_R2_mean_over_seeds"].mean() if not high.empty else np.nan
        c_high_frac = high["median_delta_C_seed_frac_positive"].mean() if not high.empty else np.nan
        a_high_frac = high["median_delta_A_over_R2_seed_frac_positive"].mean() if not high.empty else np.nan
        throat = bool(np.isfinite(c_high) and np.isfinite(a_high) and c_high > 0 and a_high < 0 and (c_high_frac >= 2/3 if np.isfinite(c_high_frac) else False))
        cone = bool(np.isfinite(c_high) and np.isfinite(a_high) and c_high > 0 and a_high > 0 and (c_high_frac >= 2/3 if np.isfinite(c_high_frac) else False) and (a_high_frac >= 2/3 if np.isfinite(a_high_frac) else False))
        snap_ratio = np.nan
        if np.isfinite(c_high) and np.isfinite(c_low_abs) and c_low_abs > 0:
            snap_ratio = float(c_high / c_low_abs)
        snap = bool(np.isfinite(snap_ratio) and snap_ratio >= 10 and c_high > 0 and (c_high_frac >= 2/3 if np.isfinite(c_high_frac) else False))
        plateau_8_16 = np.nan
        v8 = d.loc[d["m_defects"] == 8, "median_delta_C_mean_over_seeds"]
        v16 = d.loc[d["m_defects"] == 16, "median_delta_C_mean_over_seeds"]
        if len(v8) and len(v16):
            plateau_8_16 = float(abs(v16.iloc[0] - v8.iloc[0]))
        rows.append({
            "radial_window": w,
            "source_strategy_modes": ",".join(modes),
            "source_strategy_constant": strategy_constant,
            "C_high_mean_m8_m16": c_high,
            "AoverR2_high_mean_m8_m16": a_high,
            "C_high_frac_positive_mean": c_high_frac,
            "A_high_frac_positive_mean": a_high_frac,
            "C_low_abs_mean_m2_m4": c_low_abs,
            "snap_ratio_highC_to_lowAbsC": snap_ratio,
            "plateau_abs_C_m16_minus_m8": plateau_8_16,
            "pure_cone_candidate": cone,
            "throat_candidate": throat,
            "stair_step_candidate": snap,
            "anchor_confounded": not strategy_constant,
        })
    return pd.DataFrame(rows)
