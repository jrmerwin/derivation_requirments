"""
DEU GR Experiment 01L: Fixed-Epoch / Fixed-Anchor Defect Test

Purpose
-------
The 01I/01J/01K controlled defect-sink pilot used a growing tagged defect lineage
as the distance source and compared runs stopped at equal target_basin_splits. That
is not the clean GR/bandwidth question. A bandwidth sink should be compared at a
fixed global epoch, and the metric origin should be a fixed point/anchor rather
than a growing source set.

This helper implements a corrected pilot:
    1. all m values run to the same global final_epoch;
    2. all runs receive the same neutral marker split at the same bulk face, creating
       a stable anchor vertex / point-source coordinate;
    3. m_defects controls additional local forced splits after injection;
    4. distances are measured from the fixed anchor vertex, not from all tagged faces;
    5. both ball-area deficit and annular circumference deficit are paired against m=0.

This is still a controlled perturbation, not a native S-G registry defect proof.
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
class FixedAnchorDefectSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_depth: dict
    face_neighbors: dict
    face_defect: dict
    stats: dict


@dataclass
class FixedAnchorDefectRun:
    stats: dict
    spatial_snapshots: dict
    epoch_log: pd.DataFrame


def _fad_adj_from_state(faces, edge_to_faces, active):
    def get_neighbors(fid):
        ns = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            ns |= edge_to_faces[frozenset(e)]
        ns.discard(fid)
        return ns & active
    return {fid: get_neighbors(fid) for fid in active}


def _fad_components(adj):
    unseen = set(adj)
    comps = []
    while unseen:
        s = next(iter(unseen))
        unseen.remove(s)
        q = deque([s])
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
                    comp.append(v)
        comps.append(set(comp))
    comps.sort(key=len, reverse=True)
    return comps


def _fad_raw_bulk_center(adj):
    comps = _fad_components(adj)
    if not comps:
        raise RuntimeError("No component in adjacency")
    comp = comps[0]
    maxdeg = max(len(adj[n] & comp) for n in comp)
    boundary = [n for n in comp if len(adj[n] & comp) < maxdeg]
    if not boundary:
        boundary = list(comp)
    dist = {b: 0 for b in boundary}
    q = deque(boundary)
    while q:
        u = q.popleft()
        for v in adj[u] & comp:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    far = max(dist.values())
    candidates = [n for n, d in dist.items() if d == far]
    center = sorted(candidates, key=lambda x: repr(x))[0]
    return center, far, len(boundary)


def grow_fixed_epoch_fixed_anchor_defect(
    *,
    final_epoch=40,
    seed=101,
    scheduler="capped",
    max_splits_per_epoch=256,
    max_ticks_per_epoch=None,
    m_defects=0,
    defect_inject_epoch=15,
    snapshot_every=10,
    record_initial=True,
    record_final=True,
):
    """
    Spatial-depth replay with a fixed coordinate-time endpoint and fixed anchor vertex.

    Difference from the older 01I pilot:
        * The run does NOT stop at a fixed target_basin_splits.
        * Every m value is evolved to the same final_epoch.
        * A neutral marker split is applied at defect_inject_epoch for all m values.
          The new node from that marker split is the fixed metric origin.
        * Additional m_defects forced local splits consume the same per-epoch update
          budget as ordinary splits/ticks.

    m_defects is a controlled local sink strength, not yet native registry mass.
    """
    if scheduler != "capped":
        raise ValueError("This helper currently supports scheduler='capped' only.")

    if max_ticks_per_epoch is None:
        max_ticks_per_epoch = max_splits_per_epoch

    rng = np.random.default_rng(seed)

    faces = {}
    face_types = {}
    face_depth = {}
    face_defect = {}
    edge_to_faces = defaultdict(set)
    active = set()

    next_face = 0
    next_node = 6
    stats = Counter()
    epoch_log = []
    spatial_snapshots = {}

    anchor_face = None
    anchor_nodes = None
    anchor_center_node = None

    def add_face(nodes, ftype, depth, defect=False):
        nonlocal next_face
        fid = next_face
        next_face += 1
        nodes = frozenset(int(x) for x in nodes)
        faces[fid] = nodes
        face_types[fid] = str(ftype)
        face_depth[fid] = int(depth)
        face_defect[fid] = bool(defect)
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
        del face_defect[fid]

    def snapshot_raw():
        active0 = set(active)
        neigh0 = _fad_adj_from_state(faces, edge_to_faces, active0)
        return active0, dict(face_types), dict(face_depth), dict(faces), neigh0, dict(face_defect)

    def record_snapshot(ep):
        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()
        sdict = dict(stats)
        if anchor_face is not None:
            sdict["defect_anchor_face"] = int(anchor_face)
        if anchor_center_node is not None:
            sdict["defect_center_node"] = int(anchor_center_node)
        if anchor_nodes is not None:
            sdict["defect_anchor_nodes_repr"] = repr(tuple(int(x) for x in anchor_nodes))
        spatial_snapshots[int(ep)] = FixedAnchorDefectSnapshot(
            epoch=int(ep),
            active_faces=active0,
            face_nodes=faces0,
            face_types=types0,
            face_depth=depth0,
            face_neighbors=neigh0,
            face_defect=defect0,
            stats=sdict,
        )

    def should_record(ep):
        return snapshot_every is not None and snapshot_every > 0 and int(ep) % int(snapshot_every) == 0

    def is_frustrated0(fid, types0, neigh0):
        if types0[fid] != "S":
            return False
        nts = {types0[n] for n in neigh0[fid]}
        return ("G" in nts) and ("I" not in nts)

    def split_face(fid, *, forced=False, force_defect_children=None):
        nonlocal next_node
        if fid not in active:
            return None
        old_nodes = sorted(faces[fid])
        old_depth = int(face_depth[fid])
        old_defect = bool(face_defect.get(fid, False))
        child_defect = old_defect if force_defect_children is None else bool(force_defect_children)
        a_node, b_node, c_node = old_nodes
        new_node = next_node
        next_node += 1
        remove_face(fid)
        add_face((new_node, a_node, b_node), "S", old_depth + 1, child_defect)
        add_face((new_node, a_node, c_node), "I", old_depth + 1, child_defect)
        add_face((new_node, b_node, c_node), "G", old_depth + 1, child_defect)
        stats["basin_splits"] += 1
        if forced:
            stats["forced_defect_splits"] += 1
        return new_node

    def inject_marker_if_needed(ep):
        nonlocal anchor_face, anchor_nodes, anchor_center_node
        if anchor_center_node is not None or ep < defect_inject_epoch:
            return 0
        adj = _fad_adj_from_state(faces, edge_to_faces, active)
        center, raw_clearance, boundary_faces = _fad_raw_bulk_center(adj)
        anchor_face = int(center)
        anchor_nodes = tuple(sorted(int(x) for x in faces[center]))
        anchor_center_node = split_face(center, forced=False, force_defect_children=True)
        if anchor_center_node is None:
            raise RuntimeError("Marker split failed unexpectedly")
        stats["marker_anchor_split"] += 1
        stats["defect_anchor_face"] = int(anchor_face)
        stats["defect_center_node"] = int(anchor_center_node)
        stats["defect_anchor_raw_clearance"] = int(raw_clearance)
        stats["defect_boundary_faces_at_injection"] = int(boundary_faces)
        stats["defect_inject_epoch"] = int(ep)
        return 1

    # Same open seed patch as the native depth replay.
    add_face((0, 1, 2), "S", 0, False)
    add_face((0, 1, 3), "G", 0, False)
    add_face((2, 4, 5), "I", 0, False)
    add_face((3, 4, 5), "S", 0, False)

    if record_initial:
        record_snapshot(0)

    for epoch in range(1, int(final_epoch) + 1):
        marker_cost = inject_marker_if_needed(epoch)
        budget_remaining = max(0, int(max_splits_per_epoch) - int(marker_cost))

        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()

        # Persistent local sink: spend up to m_defects updates on deepest tagged faces.
        actual_forced = 0
        if int(m_defects) > 0 and anchor_center_node is not None and budget_remaining > 0:
            tagged = [fid for fid in active0 if defect0.get(fid, False)]
            rng.shuffle(tagged)
            tagged = sorted(tagged, key=lambda f: depth0[f], reverse=True)
            forced_selected = tagged[: min(int(m_defects), int(budget_remaining))]
            for fid in forced_selected:
                if budget_remaining <= 0:
                    break
                if split_face(fid, forced=True, force_defect_children=True) is not None:
                    actual_forced += 1
                    budget_remaining -= 1

        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()
        frustrated = [fid for fid in active0 if is_frustrated0(fid, types0, neigh0)]
        frontier_size = len(frustrated)
        stats["frontier_max"] = max(stats.get("frontier_max", 0), frontier_size)

        actual_normal = 0
        actual_ticks = 0
        kind = "idle"

        if frustrated and budget_remaining > 0:
            rng.shuffle(frustrated)
            selected = frustrated[: min(len(frustrated), budget_remaining)]
            for fid in selected:
                if budget_remaining <= 0:
                    break
                if split_face(fid, forced=False, force_defect_children=None) is not None:
                    actual_normal += 1
                    budget_remaining -= 1
            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(
                stats.get("max_splits_in_epoch", 0),
                int(marker_cost + actual_forced + actual_normal),
            )
            kind = "split"
            if actual_forced:
                kind = "forced_plus_split"
            if marker_cost:
                kind = "marker_plus_" + kind
        else:
            # Sterile ticks use the same remaining global budget. This differs from the older
            # pilot, where forced splits could be followed by a full tick budget.
            active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()
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
            selected = candidates[: min(len(candidates), int(budget_remaining), int(max_ticks_per_epoch))]

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
            if actual_forced:
                kind = "forced_plus_tick"
            if marker_cost:
                kind = "marker_plus_" + kind
            if actual_ticks == 0 and actual_forced == 0 and marker_cost == 0:
                stats["sterile_starved"] += 1

        epoch_log.append({
            "epoch": int(epoch),
            "kind": kind,
            "frontier_size": int(frontier_size),
            "marker_cost": int(marker_cost),
            "forced_splits": int(actual_forced),
            "normal_splits": int(actual_normal),
            "actual_split_updates": int(marker_cost + actual_forced + actual_normal),
            "ticks": int(actual_ticks),
            "active_faces": int(len(active)),
            "basin_splits_total": int(stats["basin_splits"]),
            "defect_active_faces": int(sum(1 for f in active if face_defect.get(f, False))),
            "budget_unused": int(budget_remaining),
        })

        if should_record(epoch):
            record_snapshot(epoch)

    stats["epochs"] = int(final_epoch)
    stats["final_active_faces"] = int(len(active))
    stats["final_nodes"] = int(next_node)
    stats["m_defects"] = int(m_defects)
    stats["scheduler"] = scheduler
    stats["max_splits_per_epoch"] = int(max_splits_per_epoch)
    stats["final_epoch"] = int(final_epoch)

    if record_final and int(final_epoch) not in spatial_snapshots:
        record_snapshot(int(final_epoch))

    return FixedAnchorDefectRun(
        stats=dict(stats),
        spatial_snapshots=dict(sorted(spatial_snapshots.items())),
        epoch_log=pd.DataFrame(epoch_log),
    )


def weighted_anchor_vertex_curve(snapshot, center_node=None, radius_edges=None):
    """
    Weighted ball/circumference curve around a fixed anchor vertex.

    This avoids the older source-size bias. The initial Dijkstra sources are active
    faces incident to the fixed center_node, with initial distance 0.5 * face_length.
    """
    if "_weighted_face_adj" not in globals():
        raise RuntimeError("_weighted_face_adj not found. Load src/deu_exp456_minimal.py first.")

    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    if center_node is None:
        center_node = snapshot.stats.get("defect_center_node")
    if center_node is None:
        raise ValueError("No center_node supplied and snapshot.stats has no defect_center_node")
    center_node = int(center_node)

    wadj, areas, depths = _weighted_face_adj(snapshot, component="largest")
    nodes = set(wadj)
    lengths = {f: float((3.0 ** -0.5) ** int(depths[f])) for f in nodes}

    source_faces = [
        f for f in nodes
        if center_node in snapshot.face_nodes.get(f, frozenset())
    ]
    if not source_faces:
        raise RuntimeError(f"No active faces incident to center_node={center_node}")

    # Multi-source Dijkstra from the vertex. A face incident on the vertex is reached
    # after about half a local face length, not at zero radius.
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

    tagged_faces = [f for f in nodes if getattr(snapshot, "face_defect", {}).get(f, False)]
    tagged_area = float(sum(areas[f] for f in tagged_faces if f in areas))

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
            "tagged_defect_faces_total": int(len(tagged_faces)),
            "tagged_defect_area_total": tagged_area,
            "center_node": int(center_node),
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


def summarize_fixed_anchor_paired(paired, run_df, radial_windows=None):
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
                "mean_delta_A": float(g["delta_A_vs_m0"].mean()),
                "median_delta_A_over_R2": float(g["delta_A_over_R2_vs_m0"].median()),
                "mean_delta_A_over_R2": float(g["delta_A_over_R2_vs_m0"].mean()),
                "median_delta_C": float(g["delta_C_vs_m0"].median()),
                "mean_delta_C": float(g["delta_C_vs_m0"].mean()),
                "median_delta_C_over_R": float(g["delta_C_over_R_vs_m0"].median()),
                "mean_delta_C_over_R": float(g["delta_C_over_R_vs_m0"].mean()),
                "median_C_over_R": float(g["C_over_R"].median()),
                "n_annuli": int(len(g)),
                "median_source_faces_incident_to_center_node": float(g["source_faces_incident_to_center_node"].median()),
                "median_tagged_defect_faces_total": float(g["tagged_defect_faces_total"].median()),
                "median_tagged_defect_area_total": float(g["tagged_defect_area_total"].median()),
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(), pd.DataFrame()

    meta_cols = [
        "seed", "m_defects", "forced_defect_splits", "defect_active_faces_final",
        "basin_splits", "final_active_faces", "final_nodes", "epochs", "final_epoch",
        "defect_center_node", "defect_anchor_raw_clearance",
    ]
    available_meta = [c for c in meta_cols if c in run_df.columns]
    per_seed = per_seed.merge(run_df[available_meta], on=["seed", "m_defects"], how="left")

    ens_rows = []
    for (wname, m), g in per_seed.groupby(["radial_window", "m_defects"]):
        row = {"radial_window": wname, "m_defects": int(m), "seeds": int(g["seed"].nunique())}
        for col in [
            "median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R",
            "median_source_faces_incident_to_center_node", "median_tagged_defect_faces_total",
            "median_tagged_defect_area_total",
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
        for col in ["forced_defect_splits", "defect_active_faces_final", "basin_splits", "final_active_faces"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        ens_rows.append(row)
    ensemble_summary = pd.DataFrame(ens_rows)

    fit_rows = []
    for wname, gd in per_seed[per_seed["m_defects"] > 0].groupby("radial_window"):
        for min_m in [1, 2, 4, 8]:
            h = gd[gd["m_defects"] >= min_m].copy()
            if h.empty:
                continue
            for x in ["m_defects", "forced_defect_splits", "defect_active_faces_final"]:
                if x not in h.columns:
                    continue
                for y in ["median_delta_A", "median_delta_A_over_R2", "median_delta_C", "median_delta_C_over_R"]:
                    fr = _linear_fit_df(h, x, y)
                    fr["radial_window"] = wname
                    fr["min_m"] = int(min_m)
                    fit_rows.append(fr)
    fit_table = pd.DataFrame(fit_rows)
    if not fit_table.empty:
        fit_table = fit_table[["radial_window", "min_m", "x", "y", "n", "slope", "intercept", "r2"]]

    return per_seed, ensemble_summary, fit_table


def run_fixed_epoch_anchor_ensemble(
    *,
    OUT=None,
    final_epoch=40,
    cap=256,
    seeds=(101, 202, 303),
    m_values=(0, 2, 4, 8),
    defect_inject_epoch=15,
    radius_edges=None,
    radial_windows=None,
    label=None,
    make_plots=True,
):
    """Run the corrected fixed-epoch/fixed-anchor controlled defect ensemble."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

    if label is None:
        label = f"fixed_anchor_cap{cap}_epoch{final_epoch}_{len(seeds)}seeds"

    runs = {}
    curve_rows = []
    run_rows = []

    for seed in seeds:
        for m in m_values:
            print(f"\nRunning fixed-epoch/fixed-anchor defect test: seed={seed}, m={m}")
            r = grow_fixed_epoch_fixed_anchor_defect(
                final_epoch=int(final_epoch),
                seed=int(seed),
                max_splits_per_epoch=int(cap),
                max_ticks_per_epoch=int(cap),
                m_defects=int(m),
                defect_inject_epoch=int(defect_inject_epoch),
                snapshot_every=10,
                record_final=True,
            )
            runs[(int(seed), int(m))] = r
            ep = max(r.spatial_snapshots)
            snap = r.spatial_snapshots[ep]
            center_node = int(snap.stats["defect_center_node"])
            curve = weighted_anchor_vertex_curve(snap, center_node=center_node, radius_edges=radius_edges)
            curve["seed"] = int(seed)
            curve["m_defects"] = int(m)
            curve["epoch"] = int(ep)
            curve["cap"] = int(cap)
            curve["final_epoch_target"] = int(final_epoch)
            curve_rows.append(curve)

            row = dict(r.stats)
            row["seed"] = int(seed)
            row["m_defects"] = int(m)
            row["final_epoch"] = int(ep)
            row["defect_active_faces_final"] = int(sum(1 for f in snap.active_faces if snap.face_defect.get(f, False)))
            row["source_faces_incident_to_center_node_final"] = int((curve["source_faces_incident_to_center_node"].iloc[0]))
            row["tagged_defect_area_total_final"] = float(curve["tagged_defect_area_total"].iloc[0])
            run_rows.append(row)
            print({
                k: row.get(k)
                for k in [
                    "seed", "m_defects", "final_epoch", "basin_splits", "forced_defect_splits",
                    "defect_active_faces_final", "source_faces_incident_to_center_node_final",
                    "final_active_faces",
                ]
            })

    curves = pd.concat(curve_rows, ignore_index=True)
    run_df = pd.DataFrame(run_rows)

    # Pair each seed and annulus bin against that seed's m=0 curve.
    base = curves[curves["m_defects"] == 0][[
        "seed", "bin", "R_mid", "A_ball", "A_over_R2", "C_est", "C_over_R"
    ]].rename(columns={
        "A_ball": "A_ball_m0",
        "A_over_R2": "A_over_R2_m0",
        "C_est": "C_est_m0",
        "C_over_R": "C_over_R_m0",
    })
    paired = curves.merge(base, on=["seed", "bin", "R_mid"], how="left")
    paired["delta_A_vs_m0"] = paired["A_ball_m0"] - paired["A_ball"]
    paired["delta_A_over_R2_vs_m0"] = paired["A_over_R2_m0"] - paired["A_over_R2"]
    paired["delta_C_vs_m0"] = paired["C_est_m0"] - paired["C_est"]
    paired["delta_C_over_R_vs_m0"] = paired["C_over_R_m0"] - paired["C_over_R"]

    per_seed, ensemble_summary, fit_table = summarize_fixed_anchor_paired(
        paired, run_df, radial_windows=radial_windows
    )

    stem = str(label).replace(" ", "_")
    paths = {
        "curves": OUT / f"{stem}_anchor_vertex_curves.csv",
        "paired": OUT / f"{stem}_paired_deficits.csv",
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
        # Plot wide-window area deficit by m.
        plot_df = ensemble_summary[ensemble_summary["radial_window"] == "wide_0p25_1p60"].copy()
        if not plot_df.empty:
            plt.figure(figsize=(7, 4))
            plt.errorbar(
                plot_df["m_defects"],
                plot_df["median_delta_A_over_R2_mean_over_seeds"],
                yerr=plot_df["median_delta_A_over_R2_sem_over_seeds"],
                marker="o",
                linewidth=1,
                capsize=3,
            )
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel("median ΔA/R² vs m=0")
            plt.title("Fixed-epoch fixed-anchor: area deficit")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_wide_area_deficit.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths["plot_area"] = p

            plt.figure(figsize=(7, 4))
            plt.errorbar(
                plot_df["m_defects"],
                plot_df["median_delta_C_mean_over_seeds"],
                yerr=plot_df["median_delta_C_sem_over_seeds"],
                marker="o",
                linewidth=1,
                capsize=3,
            )
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel("median ΔC vs m=0")
            plt.title("Fixed-epoch fixed-anchor: circumference deficit")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_wide_circumference_deficit.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths["plot_circumference"] = p

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
