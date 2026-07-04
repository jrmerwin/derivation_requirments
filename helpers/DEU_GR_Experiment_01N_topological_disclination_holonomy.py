"""
DEU GR Experiment 01N: Topological Disclination / Holonomy Test

Purpose
-------
The controlled split-sink experiments showed a robust lapse/useful-bandwidth drain but
failed to produce a robust conical spatial deficit in the refinement-weighted metric.
This helper tests the alternative mechanism: mass as a true topological holonomy defect.

In a triangular 2D surface, a flat interior vertex has six equilateral sectors.  A
five-sector vertex is a positive disclination: it removes one 60-degree wedge and is
the discrete analogue of a 2+1D conical mass.  A seven-sector vertex is the negative
curvature/excess-angle control.

This file evolves closed triangular fans with q sectors around a fixed center vertex:
    q=6 : flat local topology control
    q=5 : one missing sector, positive conical deficit
    q=4 : two missing sectors, stronger positive deficit
    q=7 : one extra sector, negative-curvature control

It then measures refinement-weighted ball area and annular circumference around the
fixed center vertex and pairs every q against the q=6 control for the same seed.

This is not yet a native S-G registry defect.  It is the measurement and mechanism
calibration for the proposed "topology stitch" pivot.
"""

from dataclasses import dataclass
from collections import defaultdict, Counter, deque
from pathlib import Path
import itertools
import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class FanDisclinationSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_depth: dict
    face_neighbors: dict
    stats: dict


@dataclass
class FanDisclinationRun:
    stats: dict
    spatial_snapshots: dict
    epoch_log: pd.DataFrame


def _fd_adj_from_state(faces, edge_to_faces, active):
    def get_neighbors(fid):
        ns = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            ns |= edge_to_faces[frozenset(e)]
        ns.discard(fid)
        return ns & active
    return {fid: get_neighbors(fid) for fid in active}


def _fd_components(adj):
    unseen = set(adj)
    comps = []
    while unseen:
        s = next(iter(unseen))
        unseen.remove(s)
        q = deque([s])
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj.get(u, set()):
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
                    comp.append(v)
        comps.append(set(comp))
    comps.sort(key=len, reverse=True)
    return comps


def grow_fan_disclination_depth(
    *,
    q_sectors=6,
    final_epoch=52,
    seed=101,
    max_splits_per_epoch=512,
    max_ticks_per_epoch=None,
    snapshot_every=10,
    record_initial=True,
    record_final=True,
    initial_type_pattern=None,
):
    """
    Depth-only triangular foam replay from a closed q-sector fan around center vertex 0.

    A flat triangular lattice has q=6 sectors around an interior vertex.  q<6 is a
    positive angular deficit; q>6 is an excess-angle control.

    The same S/G/I typed split/tick rule is used as the native depth-only replay:
        split: S touching G and no I
        tick: I->S, G->I, S->G with the native candidate priorities
    """
    q_sectors = int(q_sectors)
    if q_sectors < 3:
        raise ValueError("q_sectors must be >= 3")
    if max_ticks_per_epoch is None:
        max_ticks_per_epoch = max_splits_per_epoch

    rng = np.random.default_rng(seed)
    faces = {}
    face_types = {}
    face_depth = {}
    edge_to_faces = defaultdict(set)
    active = set()
    next_face = 0
    next_node = q_sectors + 1  # center node 0; ring nodes 1..q
    stats = Counter()
    epoch_log = []
    spatial_snapshots = {}

    def add_face(nodes, ftype, depth):
        nonlocal next_face
        fid = next_face
        next_face += 1
        nodes = frozenset(int(x) for x in nodes)
        faces[fid] = nodes
        face_types[fid] = str(ftype)
        face_depth[fid] = int(depth)
        for e in itertools.combinations(sorted(nodes), 2):
            edge_to_faces[frozenset(e)].add(fid)
        active.add(fid)
        return fid

    def remove_face(fid):
        for e in itertools.combinations(sorted(faces[fid]), 2):
            key = frozenset(e)
            edge_to_faces[key].discard(fid)
            if not edge_to_faces[key]:
                del edge_to_faces[key]
        active.discard(fid)
        del faces[fid]
        del face_types[fid]
        del face_depth[fid]

    def snapshot_raw():
        active0 = set(active)
        neigh0 = _fd_adj_from_state(faces, edge_to_faces, active0)
        return active0, dict(face_types), dict(face_depth), dict(faces), neigh0

    def record_snapshot(ep):
        active0, types0, depth0, faces0, neigh0 = snapshot_raw()
        sdict = dict(stats)
        sdict["q_sectors"] = int(q_sectors)
        sdict["topological_deficit_units"] = int(6 - q_sectors)
        sdict["expected_circumference_ratio_vs_q6"] = float(q_sectors / 6.0)
        sdict["center_node"] = 0
        spatial_snapshots[int(ep)] = FanDisclinationSnapshot(
            epoch=int(ep),
            active_faces=active0,
            face_nodes=faces0,
            face_types=types0,
            face_depth=depth0,
            face_neighbors=neigh0,
            stats=sdict,
        )

    def should_record(ep):
        return snapshot_every is not None and snapshot_every > 0 and int(ep) % int(snapshot_every) == 0

    def is_frustrated0(fid, types0, neigh0):
        if types0[fid] != "S":
            return False
        nts = {types0[n] for n in neigh0[fid]}
        return ("G" in nts) and ("I" not in nts)

    def split_face(fid):
        nonlocal next_node
        if fid not in active:
            return False
        old_nodes = sorted(faces[fid])
        old_depth = int(face_depth[fid])
        a_node, b_node, c_node = old_nodes
        new_node = next_node
        next_node += 1
        remove_face(fid)
        add_face((new_node, a_node, b_node), "S", old_depth + 1)
        add_face((new_node, a_node, c_node), "I", old_depth + 1)
        add_face((new_node, b_node, c_node), "G", old_depth + 1)
        stats["basin_splits"] += 1
        return True

    # Closed fan: q triangles around center node 0 and boundary cycle 1..q.
    if initial_type_pattern is None:
        # Alternating S/G, no I in the seed, so the fan is live but not screened.
        initial_type_pattern = ["S" if i % 2 == 0 else "G" for i in range(q_sectors)]
    if len(initial_type_pattern) < q_sectors:
        reps = int(np.ceil(q_sectors / len(initial_type_pattern)))
        initial_type_pattern = list(initial_type_pattern) * reps
    initial_type_pattern = list(initial_type_pattern)[:q_sectors]

    for i in range(q_sectors):
        a = 1 + i
        b = 1 + ((i + 1) % q_sectors)
        add_face((0, a, b), initial_type_pattern[i], 0)

    if record_initial:
        record_snapshot(0)

    for epoch in range(1, int(final_epoch) + 1):
        active0, types0, depth0, faces0, neigh0 = snapshot_raw()
        frustrated = [fid for fid in active0 if is_frustrated0(fid, types0, neigh0)]
        frontier_size = len(frustrated)
        stats["frontier_max"] = max(stats.get("frontier_max", 0), frontier_size)
        actual_splits = 0
        actual_ticks = 0
        kind = "idle"

        if frustrated:
            rng.shuffle(frustrated)
            selected = frustrated[: min(len(frustrated), int(max_splits_per_epoch))]
            for fid in selected:
                if split_face(fid):
                    actual_splits += 1
            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(stats.get("max_splits_in_epoch", 0), actual_splits)
            kind = "split"
        else:
            screening_I = []
            adjacent_I = []
            g_faces = []
            s_faces = []
            for fid in active0:
                nts = {types0[n] for n in neigh0[fid]}
                if types0[fid] == "I" and "S" in nts and "G" in nts:
                    screening_I.append(fid)
                if types0[fid] == "I" and "S" in nts:
                    adjacent_I.append(fid)
                if types0[fid] == "G":
                    g_faces.append(fid)
                if types0[fid] == "S":
                    s_faces.append(fid)

            if screening_I:
                candidates = screening_I
            elif adjacent_I:
                candidates = adjacent_I
            elif g_faces:
                candidates = g_faces
            elif s_faces:
                candidates = s_faces
            else:
                candidates = list(active0)

            rng.shuffle(candidates)
            selected = candidates[: min(len(candidates), int(max_ticks_per_epoch))]
            for fid in selected:
                if fid not in active:
                    continue
                old = face_types[fid]
                if old == "I":
                    face_types[fid] = "S"
                elif old == "G":
                    face_types[fid] = "I"
                else:
                    face_types[fid] = "G"
                stats["sterile_ticks"] += 1
                stats[f"sterile_{old}_to_{face_types[fid]}"] += 1
                actual_ticks += 1
            stats["tick_epochs"] += 1
            stats["max_ticks_in_epoch"] = max(stats.get("max_ticks_in_epoch", 0), actual_ticks)
            kind = "tick"
            if actual_ticks == 0:
                stats["sterile_starved"] += 1

        epoch_log.append({
            "epoch": int(epoch),
            "kind": kind,
            "frontier_size": int(frontier_size),
            "actual_splits": int(actual_splits),
            "ticks": int(actual_ticks),
            "active_faces": int(len(active)),
            "basin_splits_total": int(stats["basin_splits"]),
        })

        if should_record(epoch):
            record_snapshot(epoch)

    stats["epochs"] = int(final_epoch)
    stats["final_active_faces"] = int(len(active))
    stats["final_nodes"] = int(next_node)
    stats["q_sectors"] = int(q_sectors)
    stats["topological_deficit_units"] = int(6 - q_sectors)
    stats["expected_circumference_ratio_vs_q6"] = float(q_sectors / 6.0)
    stats["final_epoch"] = int(final_epoch)

    if record_final and int(final_epoch) not in spatial_snapshots:
        record_snapshot(int(final_epoch))

    return FanDisclinationRun(
        stats=dict(stats),
        spatial_snapshots=dict(sorted(spatial_snapshots.items())),
        epoch_log=pd.DataFrame(epoch_log),
    )


def fan_weighted_vertex_curve(snapshot, center_node=0, radius_edges=None, component="largest"):
    """Weighted ball/circumference curve around a fixed vertex in a depth snapshot."""
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    # Build face adjacency from snapshot.
    adj0 = {f: set(ns) for f, ns in snapshot.face_neighbors.items()}
    if component == "largest":
        comps = _fd_components(adj0)
        keep = comps[0] if comps else set(adj0)
        adj0 = {f: set(ns) & keep for f, ns in adj0.items() if f in keep}
    else:
        keep = set(adj0)

    depths = {f: int(snapshot.face_depth.get(f, 0)) for f in adj0}
    lengths = {f: float((3.0 ** -0.5) ** depths[f]) for f in adj0}
    areas = {f: float(3.0 ** (-depths[f])) for f in adj0}
    wadj = {
        f: [(g, 0.5 * (lengths[f] + lengths[g])) for g in ns if g in adj0]
        for f, ns in adj0.items()
    }

    source_faces = [f for f in adj0 if int(center_node) in snapshot.face_nodes.get(f, frozenset())]
    if not source_faces:
        raise RuntimeError(f"No active faces incident to center_node={center_node}")

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
            "source_faces_incident_to_center_node": int(len(source_faces)),
            "center_node": int(center_node),
            "component_faces": int(len(adj0)),
            "total_weighted_area_component": float(sum(areas.values())),
            "depth_min": int(min(depths.values())) if depths else np.nan,
            "depth_med": float(np.median(list(depths.values()))) if depths else np.nan,
            "depth_max": int(max(depths.values())) if depths else np.nan,
        })
    return pd.DataFrame(rows)


def _linear_fit_df(df, x, y):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return {"x": x, "y": y, "n": int(len(d)), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    xv = d[x].to_numpy(dtype=float)
    yv = d[y].to_numpy(dtype=float)
    slope, intercept = np.polyfit(xv, yv, 1)
    pred = slope * xv + intercept
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {"x": x, "y": y, "n": int(len(d)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def summarize_fan_disclination_paired(paired, run_df, radial_windows=None):
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
        for (seed, q), g in d.groupby(["seed", "q_sectors"]):
            rows.append({
                "radial_window": wname,
                "seed": int(seed),
                "q_sectors": int(q),
                "topological_deficit_units": int(6 - int(q)),
                "expected_ratio_vs_q6": float(int(q) / 6.0),
                "median_delta_A": float(g["delta_A_vs_q6"].median()),
                "mean_delta_A": float(g["delta_A_vs_q6"].mean()),
                "median_delta_A_over_R2": float(g["delta_A_over_R2_vs_q6"].median()),
                "mean_delta_A_over_R2": float(g["delta_A_over_R2_vs_q6"].mean()),
                "median_delta_C": float(g["delta_C_vs_q6"].median()),
                "mean_delta_C": float(g["delta_C_vs_q6"].mean()),
                "median_delta_C_over_R": float(g["delta_C_over_R_vs_q6"].median()),
                "mean_delta_C_over_R": float(g["delta_C_over_R_vs_q6"].mean()),
                "median_C_over_R": float(g["C_over_R"].median()),
                "median_C_ratio_vs_q6": float(g["C_ratio_vs_q6"].median()),
                "median_A_ratio_vs_q6": float(g["A_ratio_vs_q6"].median()),
                "n_annuli": int(len(g)),
                "median_source_faces": float(g["source_faces_incident_to_center_node"].median()),
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(), pd.DataFrame()

    meta_cols = [
        "seed", "q_sectors", "basin_splits", "final_active_faces", "final_nodes", "epochs",
        "final_epoch", "frontier_max", "split_epochs", "tick_epochs",
        "total_weighted_area_final",
    ]
    per_seed = per_seed.merge(run_df[[c for c in meta_cols if c in run_df.columns]], on=["seed", "q_sectors"], how="left")

    ens_rows = []
    for (wname, q), g in per_seed.groupby(["radial_window", "q_sectors"]):
        row = {
            "radial_window": wname,
            "q_sectors": int(q),
            "topological_deficit_units": int(6 - int(q)),
            "expected_ratio_vs_q6": float(int(q) / 6.0),
            "seeds": int(g["seed"].nunique()),
        }
        for col in [
            "median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R",
            "median_C_over_R", "median_C_ratio_vs_q6", "median_A_ratio_vs_q6", "median_source_faces",
        ]:
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(vals):
                row[f"{col}_mean_over_seeds"] = float(np.mean(vals))
                row[f"{col}_sem_over_seeds"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                row[f"{col}_seed_frac_positive"] = float(np.mean(vals > 0))
            else:
                row[f"{col}_mean_over_seeds"] = np.nan
                row[f"{col}_sem_over_seeds"] = np.nan
                row[f"{col}_seed_frac_positive"] = np.nan
        if "basin_splits" in g.columns:
            row["basin_splits_mean"] = float(pd.to_numeric(g["basin_splits"], errors="coerce").mean())
        if "final_active_faces" in g.columns:
            row["final_active_faces_mean"] = float(pd.to_numeric(g["final_active_faces"], errors="coerce").mean())
        ens_rows.append(row)
    ensemble_summary = pd.DataFrame(ens_rows)

    fit_rows = []
    pos_def = per_seed[per_seed["topological_deficit_units"] > 0].copy()
    for wname, gd in pos_def.groupby("radial_window"):
        for x in ["topological_deficit_units", "q_sectors"]:
            for y in ["median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R", "median_C_ratio_vs_q6"]:
                fr = _linear_fit_df(gd, x, y)
                fr["radial_window"] = wname
                fit_rows.append(fr)
    fit_table = pd.DataFrame(fit_rows)
    if not fit_table.empty:
        fit_table = fit_table[["radial_window", "x", "y", "n", "slope", "intercept", "r2"]]

    return per_seed, ensemble_summary, fit_table


def run_fan_disclination_ensemble(
    *,
    OUT=None,
    final_epoch=52,
    cap=512,
    seeds=(101, 202, 303),
    q_values=(6, 5, 4, 7),
    radius_edges=None,
    radial_windows=None,
    label=None,
    make_plots=True,
):
    """Run q-sector disclination ensemble and pair all q values against q=6 by seed."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    if label is None:
        label = f"fan_disclination_cap{cap}_epoch{final_epoch}_{len(seeds)}seeds"

    if 6 not in set(q_values):
        raise ValueError("q_values must include q=6 flat control")

    runs = {}
    curve_rows = []
    run_rows = []

    for seed in seeds:
        for q in q_values:
            print(f"\nRunning fan disclination: seed={seed}, q={q}")
            r = grow_fan_disclination_depth(
                q_sectors=int(q),
                final_epoch=int(final_epoch),
                seed=int(seed),
                max_splits_per_epoch=int(cap),
                max_ticks_per_epoch=int(cap),
                snapshot_every=10,
                record_final=True,
            )
            runs[(int(seed), int(q))] = r
            ep = max(r.spatial_snapshots)
            snap = r.spatial_snapshots[ep]
            curve = fan_weighted_vertex_curve(snap, center_node=0, radius_edges=radius_edges)
            curve["seed"] = int(seed)
            curve["q_sectors"] = int(q)
            curve["topological_deficit_units"] = int(6 - int(q))
            curve["expected_ratio_vs_q6"] = float(int(q) / 6.0)
            curve["epoch"] = int(ep)
            curve["cap"] = int(cap)
            curve_rows.append(curve)

            total_area = float(sum(3.0 ** (-int(d)) for d in snap.face_depth.values()))
            row = dict(r.stats)
            row["seed"] = int(seed)
            row["q_sectors"] = int(q)
            row["final_epoch"] = int(ep)
            row["total_weighted_area_final"] = total_area
            run_rows.append(row)
            print({
                k: row.get(k)
                for k in [
                    "seed", "q_sectors", "topological_deficit_units", "final_epoch",
                    "basin_splits", "final_active_faces", "frontier_max", "total_weighted_area_final",
                ]
            })

    curves = pd.concat(curve_rows, ignore_index=True)
    run_df = pd.DataFrame(run_rows)

    base = curves[curves["q_sectors"] == 6][[
        "seed", "bin", "R_mid", "A_ball", "A_over_R2", "C_est", "C_over_R"
    ]].rename(columns={
        "A_ball": "A_ball_q6",
        "A_over_R2": "A_over_R2_q6",
        "C_est": "C_est_q6",
        "C_over_R": "C_over_R_q6",
    })

    paired = curves.merge(base, on=["seed", "bin", "R_mid"], how="left")
    paired["delta_A_vs_q6"] = paired["A_ball_q6"] - paired["A_ball"]
    paired["delta_A_over_R2_vs_q6"] = paired["A_over_R2_q6"] - paired["A_over_R2"]
    paired["delta_C_vs_q6"] = paired["C_est_q6"] - paired["C_est"]
    paired["delta_C_over_R_vs_q6"] = paired["C_over_R_q6"] - paired["C_over_R"]
    paired["C_ratio_vs_q6"] = paired["C_est"] / paired["C_est_q6"].replace(0, np.nan)
    paired["A_ratio_vs_q6"] = paired["A_ball"] / paired["A_ball_q6"].replace(0, np.nan)

    per_seed, ensemble_summary, fit_table = summarize_fan_disclination_paired(
        paired, run_df, radial_windows=radial_windows
    )

    stem = str(label).replace(" ", "_")
    paths = {
        "curves": OUT / f"{stem}_vertex_curves.csv",
        "paired": OUT / f"{stem}_paired_vs_q6.csv",
        "run_summary": OUT / f"{stem}_run_summary.csv",
        "per_seed_summary": OUT / f"{stem}_per_seed_window_summary.csv",
        "ensemble_summary": OUT / f"{stem}_ensemble_window_summary.csv",
        "fit_table": OUT / f"{stem}_fit_table.csv",
    }
    curves.to_csv(paths["curves"], index=False)
    paired.to_csv(paths["paired"], index=False)
    run_df.to_csv(paths["run_summary"], index=False)
    per_seed.to_csv(paths["per_seed_summary"], index=False)
    ensemble_summary.to_csv(paths["ensemble_summary"], index=False)
    fit_table.to_csv(paths["fit_table"], index=False)

    print("\nRun summary:")
    try:
        display(run_df)
    except NameError:
        print(run_df)

    print("\nEnsemble window summary:")
    try:
        display(ensemble_summary)
    except NameError:
        print(ensemble_summary)

    print("\nFit table:")
    try:
        display(fit_table)
    except NameError:
        print(fit_table)

    if make_plots:
        for wname in ["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"]:
            plot_df = ensemble_summary[ensemble_summary["radial_window"] == wname].copy()
            if plot_df.empty:
                continue
            plot_df = plot_df.sort_values("q_sectors")

            plt.figure(figsize=(7, 4))
            plt.errorbar(
                plot_df["q_sectors"],
                plot_df["median_C_ratio_vs_q6_mean_over_seeds"],
                yerr=plot_df["median_C_ratio_vs_q6_sem_over_seeds"],
                marker="o",
                linewidth=1,
                capsize=3,
            )
            # q/6 reference line.
            xs = np.array(sorted(plot_df["q_sectors"].unique()), dtype=float)
            plt.plot(xs, xs / 6.0, linestyle="--", linewidth=1)
            plt.axhline(1.0, linewidth=1)
            plt.xlabel("central sector count q")
            plt.ylabel("median C(q)/C(q=6)")
            plt.title(f"Topological disclination circumference ratio: {wname}")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_{wname}_C_ratio_vs_q6.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths[f"plot_C_ratio_{wname}"] = p

            plt.figure(figsize=(7, 4))
            pos_df = plot_df[plot_df["topological_deficit_units"] >= 0].copy()
            plt.errorbar(
                pos_df["topological_deficit_units"],
                pos_df["median_delta_A_over_R2_mean_over_seeds"],
                yerr=pos_df["median_delta_A_over_R2_sem_over_seeds"],
                marker="o",
                linewidth=1,
                capsize=3,
            )
            plt.axhline(0, linewidth=1)
            plt.xlabel("missing sector units, 6 - q")
            plt.ylabel("median ΔA/R² vs q=6")
            plt.title(f"Topological disclination area deficit: {wname}")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_{wname}_area_deficit_vs_missing_sector.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths[f"plot_area_deficit_{wname}"] = p

    print("\nWrote:")
    for p in paths.values():
        print(" ", p)

    return {
        "runs": runs,
        "curves": curves,
        "paired": paired,
        "run_summary": run_df,
        "per_seed_summary": per_seed,
        "ensemble_summary": ensemble_summary,
        "fit_table": fit_table,
        "paths": paths,
    }
