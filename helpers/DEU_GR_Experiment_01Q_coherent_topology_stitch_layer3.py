"""
DEU GR Experiment 01Q: Coherent Topological Stitch Layer-3 Test

Purpose
-------
The refinement-sink defect produced a robust lapse/bandwidth signal but did not
produce a robust conical spatial deficit. This helper implements a deliberately
coherent topological stitch: a tagged defect face is updated by a 1->2 move
instead of the area-preserving 1->3 move, omitting the same child sector every
time. This is a controlled surrogate for a true holonomy/stitch rule.

Key fixes relative to the earlier topology_stitch_pilot:
1. The omitted sector is coherent, not rotating. A rotating omission can average
   away wedge orientation and behave more like distributed damage than a cone.
2. The measurement anchor is robust. If the virtual center node loses all active
   incident faces, the ball source falls back to the fixed landmark triangle
   nodes created at injection, rather than failing with "No active faces".
3. The pipeline is fixed-epoch, fixed-anchor, and refinement-weighted, matching
   the Layer-3 conical-deficit extraction convention.

Dependencies
------------
Load src/deu_exp456_minimal.py first so _weighted_face_adj is present. The helper
will also work if _weighted_face_adj is already in globals from previous cells.
"""

from dataclasses import dataclass
from collections import defaultdict, Counter, deque
from pathlib import Path
import itertools
import math
import ast
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class CoherentStitchSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_depth: dict
    face_neighbors: dict
    face_defect: dict
    stats: dict


@dataclass
class CoherentStitchRun:
    stats: dict
    spatial_snapshots: dict
    epoch_log: pd.DataFrame


def _cs_adj_from_state(faces, edge_to_faces, active):
    def get_neighbors(fid):
        ns = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            ns |= edge_to_faces[frozenset(e)]
        ns.discard(fid)
        return ns & active
    return {fid: get_neighbors(fid) for fid in active}


def _cs_components(adj):
    unseen = set(adj)
    comps = []
    while unseen:
        s = next(iter(unseen))
        unseen.remove(s)
        q = deque([s])
        comp = {s}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
                    comp.add(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _cs_raw_bulk_center(adj):
    comps = _cs_components(adj)
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


def grow_fixed_epoch_coherent_topology_stitch_defect(
    *,
    final_epoch=37,
    seed=101,
    scheduler="capped",
    max_splits_per_epoch=256,
    max_ticks_per_epoch=None,
    m_defects=0,
    defect_inject_epoch=15,
    snapshot_every=10,
    record_initial=True,
    record_final=True,
    stitch_omit_index=1,
):
    """
    Fixed-epoch spatial-depth evolution with a coherent topological stitch.

    m_defects: number of tagged defect faces per epoch to update by stitch.
    stitch_omit_index: which child sector to omit at every stitch. Use 0, 1, or 2.

    A normal split creates three children:
        0: (new, a, b) type S
        1: (new, a, c) type I
        2: (new, b, c) type G

    A coherent stitch always omits the same child index. This intentionally
    introduces a persistent angular linkage deficit instead of merely extra
    refinement density.
    """
    if scheduler != "capped":
        raise ValueError("Only scheduler='capped' is supported in this pilot.")
    if max_ticks_per_epoch is None:
        max_ticks_per_epoch = max_splits_per_epoch
    stitch_omit_index = int(stitch_omit_index)
    if stitch_omit_index not in (0, 1, 2):
        raise ValueError("stitch_omit_index must be 0, 1, or 2")

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
        neigh0 = _cs_adj_from_state(faces, edge_to_faces, active0)
        return active0, dict(face_types), dict(face_depth), dict(faces), neigh0, dict(face_defect)

    def record_snapshot(ep):
        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()
        sdict = dict(stats)
        if anchor_face is not None:
            sdict["defect_anchor_face"] = int(anchor_face)
        if anchor_center_node is not None:
            sdict["defect_center_node"] = int(anchor_center_node)
        if anchor_nodes is not None:
            sdict["defect_anchor_nodes"] = tuple(int(x) for x in anchor_nodes)
            sdict["defect_anchor_nodes_repr"] = repr(tuple(int(x) for x in anchor_nodes))
        spatial_snapshots[int(ep)] = CoherentStitchSnapshot(
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

    def split_face(fid, *, forced=False, force_defect_children=None, stitch=False):
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

        child_specs = [
            ((new_node, a_node, b_node), "S"),
            ((new_node, a_node, c_node), "I"),
            ((new_node, b_node, c_node), "G"),
        ]

        if stitch:
            for j, (nodes, ftype) in enumerate(child_specs):
                if j == stitch_omit_index:
                    continue
                add_face(nodes, ftype, old_depth + 1, child_defect)
            stats["forced_stitch_splits"] += 1
            stats["omitted_child_faces"] += 1
            stats["omitted_weighted_area"] += float(3.0 ** (-(old_depth + 1)))
            stats[f"omitted_child_index_{stitch_omit_index}"] += 1
        else:
            for nodes, ftype in child_specs:
                add_face(nodes, ftype, old_depth + 1, child_defect)

        stats["basin_splits"] += 1
        if forced:
            stats["forced_defect_splits"] += 1
        return new_node

    def inject_marker_if_needed(ep):
        nonlocal anchor_face, anchor_nodes, anchor_center_node
        if anchor_center_node is not None or ep < defect_inject_epoch:
            return 0
        adj = _cs_adj_from_state(faces, edge_to_faces, active)
        center, raw_clearance, boundary_faces = _cs_raw_bulk_center(adj)
        anchor_face = int(center)
        anchor_nodes = tuple(sorted(int(x) for x in faces[center]))
        anchor_center_node = split_face(center, forced=False, force_defect_children=True, stitch=False)
        if anchor_center_node is None:
            raise RuntimeError("Marker split failed unexpectedly")
        stats["marker_anchor_split"] += 1
        stats["defect_anchor_face"] = int(anchor_face)
        stats["defect_center_node"] = int(anchor_center_node)
        stats["defect_anchor_raw_clearance"] = int(raw_clearance)
        stats["defect_boundary_faces_at_injection"] = int(boundary_faces)
        stats["defect_inject_epoch"] = int(ep)
        stats["defect_anchor_nodes"] = tuple(int(x) for x in anchor_nodes)
        stats["defect_anchor_nodes_repr"] = repr(tuple(int(x) for x in anchor_nodes))
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

        actual_forced = 0
        if int(m_defects) > 0 and anchor_center_node is not None and budget_remaining > 0:
            # Tagged faces are the defect lineage. Stitch deepest first so the
            # defect remains localized near the refinement singularity.
            tagged = [fid for fid in active0 if defect0.get(fid, False)]
            rng.shuffle(tagged)
            tagged = sorted(tagged, key=lambda f: (depth0[f], repr(f)), reverse=True)
            forced_selected = tagged[: min(int(m_defects), int(budget_remaining))]
            for fid in forced_selected:
                if budget_remaining <= 0:
                    break
                if split_face(fid, forced=True, force_defect_children=True, stitch=True) is not None:
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
                if split_face(fid, forced=False, force_defect_children=None, stitch=False) is not None:
                    actual_normal += 1
                    budget_remaining -= 1
            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(stats.get("max_splits_in_epoch", 0), int(marker_cost + actual_forced + actual_normal))
            kind = "split"
            if actual_forced:
                kind = "stitch_plus_split"
            if marker_cost:
                kind = "marker_plus_" + kind
        else:
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
                kind = "stitch_plus_tick"
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
            "omitted_weighted_area_total": float(stats.get("omitted_weighted_area", 0.0)),
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
    stats["stitch_omit_index"] = int(stitch_omit_index)

    if record_final and int(final_epoch) not in spatial_snapshots:
        record_snapshot(int(final_epoch))

    return CoherentStitchRun(stats=dict(stats), spatial_snapshots=dict(sorted(spatial_snapshots.items())), epoch_log=pd.DataFrame(epoch_log))


def _cs_parse_anchor_nodes(stats):
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


def coherent_stitch_weighted_anchor_curve(snapshot, center_node=None, radius_edges=None, source_mode="landmark_first"):
    """
    Weighted ball/circumference curve around a fixed virtual anchor.

    The previous stitch pilot failed if the center node lost all active incident
    faces. Here the fixed virtual anchor is represented by the center node plus
    the original anchor triangle nodes. Source priority:
      1. active faces incident to center_node;
      2. active faces incident to any landmark node;
      3. tagged defect faces, as a last-resort diagnostic fallback.
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
    anchor_nodes = _cs_parse_anchor_nodes(snapshot.stats)
    landmark_nodes = [center_node] + [int(x) for x in anchor_nodes]
    landmark_nodes = list(dict.fromkeys(landmark_nodes))

    wadj, areas, depths = _weighted_face_adj(snapshot, component="largest")
    nodes = set(wadj)
    lengths = {f: float((3.0 ** -0.5) ** int(depths[f])) for f in nodes}

    def incident_faces_for_node_set(node_set):
        ns = set(int(x) for x in node_set)
        return [f for f in nodes if snapshot.face_nodes.get(f, frozenset()) & ns]

    center_faces = incident_faces_for_node_set([center_node])
    landmark_faces = incident_faces_for_node_set(landmark_nodes)
    tagged_faces = [f for f in nodes if getattr(snapshot, "face_defect", {}).get(f, False)]

    if center_faces:
        source_faces = center_faces
        strategy = "center_node"
    elif landmark_faces:
        source_faces = landmark_faces
        strategy = "landmark_nodes"
    elif tagged_faces:
        source_faces = tagged_faces
        strategy = "tagged_defect_fallback"
    else:
        raise RuntimeError(f"No usable source faces for center_node={center_node}, landmark_nodes={landmark_nodes}")

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
            "source_faces_count": int(len(source_faces)),
            "center_faces_count": int(len(center_faces)),
            "landmark_faces_count": int(len(landmark_faces)),
            "tagged_defect_faces_total": int(len(tagged_faces)),
            "tagged_defect_area_total": tagged_area,
            "component_weighted_area": component_area,
            "center_node": int(center_node),
            "landmark_nodes_repr": repr(tuple(int(x) for x in landmark_nodes)),
            "depth_min": int(min(depths.values())) if depths else np.nan,
            "depth_med": float(np.median(list(depths.values()))) if depths else np.nan,
            "depth_max": int(max(depths.values())) if depths else np.nan,
        })
    return pd.DataFrame(rows)


def _cs_sem(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) <= 1:
        return np.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _cs_frac_pos(vals):
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) == 0:
        return np.nan
    return float((vals > 0).mean())


def _cs_ols(x, y):
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
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)}


def summarize_coherent_stitch_paired(paired, run_df, radial_windows=None):
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
        return per_seed, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    meta_cols = [
        "seed", "m_defects", "forced_defect_splits", "forced_stitch_splits",
        "omitted_child_faces", "omitted_weighted_area", "defect_active_faces_final",
        "basin_splits", "final_active_faces", "final_nodes", "epochs", "final_epoch",
        "defect_center_node", "defect_anchor_raw_clearance", "stitch_omit_index",
    ]
    available_meta = [c for c in meta_cols if c in run_df.columns]
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
            row[f"{col}_sem_over_seeds"] = _cs_sem(vals)
            row[f"{col}_seed_frac_positive"] = _cs_frac_pos(vals)
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
                    fit_rows.append({"radial_window": wname, "min_m": int(min_m), "x": x, "y": y, **_cs_ols(h[x], h[y])})
    fit_table = pd.DataFrame(fit_rows)
    if not fit_table.empty:
        fit_table = fit_table[["radial_window", "min_m", "x", "y", "n", "slope", "intercept", "r2"]]

    verdict_rows = []
    core = ensemble_summary[
        ensemble_summary["radial_window"].isin(["mid_0p60_1p00", "outer_1p00_1p60", "wide_0p25_1p60"])
        & (ensemble_summary["m_defects"] > 0)
    ].copy()
    if not core.empty:
        core["A_pos"] = core["median_delta_A_over_R2_mean_over_seeds"] > 0
        core["C_pos"] = core["median_delta_C_mean_over_seeds"] > 0
        core["A_frac_good"] = core["median_delta_A_over_R2_seed_frac_positive"] >= (2/3)
        core["C_frac_good"] = core["median_delta_C_seed_frac_positive"] >= (2/3)
        core["row_conical"] = core["A_pos"] & core["C_pos"] & core["A_frac_good"] & core["C_frac_good"]
        high = core[core["m_defects"] >= 4]
        n_high = int(len(high))
        n_conical_high = int(high["row_conical"].sum()) if n_high else 0
        verdict = "STITCH_INCONCLUSIVE"
        reason = "Insufficient support rows."
        if n_high and n_conical_high >= max(1, math.ceil(0.5 * n_high)):
            verdict = "TOPOLOGICAL_STITCH_CONICAL_SIGNAL"
            reason = "Most m>=4 mid/outer/wide rows have positive area and circumference deficits with seed support."
        elif n_high:
            verdict = "STITCH_WEAK_OR_NULL"
            reason = "Fewer than half of m>=4 mid/outer/wide rows support both positive area and circumference deficits."
        verdict_rows.append({
            "verdict": verdict,
            "verdict_reason": reason,
            "n_core_rows": int(len(core)),
            "n_core_high_m_rows": n_high,
            "n_core_high_m_conical_rows": n_conical_high,
            "n_core_rows_conical": int(core["row_conical"].sum()),
        })
    verdict_df = pd.DataFrame(verdict_rows)
    return per_seed, ensemble_summary, fit_table, verdict_df


def run_coherent_topological_stitch_layer3_ensemble(
    *,
    OUT=None,
    final_epoch=37,
    cap=256,
    seeds=(101, 202, 303),
    m_values=(0, 1, 2, 4),
    defect_inject_epoch=15,
    stitch_omit_index=1,
    radius_edges=None,
    radial_windows=None,
    label=None,
    make_plots=True,
):
    """Run the coherent topology-stitch Layer-3 ensemble and pair against m=0 by seed."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)
    if label is None:
        label = f"coherent_stitch_cap{cap}_epoch{final_epoch}_{len(seeds)}seeds_omit{stitch_omit_index}"

    runs = {}
    curve_rows = []
    run_rows = []
    for seed in seeds:
        for m in m_values:
            print(f"\nRunning coherent topology stitch: seed={seed}, m={m}, omit={stitch_omit_index}")
            r = grow_fixed_epoch_coherent_topology_stitch_defect(
                final_epoch=int(final_epoch),
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
            ep = max(r.spatial_snapshots)
            snap = r.spatial_snapshots[ep]
            center_node = int(snap.stats["defect_center_node"])
            curve = coherent_stitch_weighted_anchor_curve(snap, center_node=center_node, radius_edges=radius_edges)
            curve["seed"] = int(seed)
            curve["m_defects"] = int(m)
            curve["epoch"] = int(ep)
            curve["cap"] = int(cap)
            curve["final_epoch_target"] = int(final_epoch)
            curve["stitch_omit_index"] = int(stitch_omit_index)
            curve_rows.append(curve)
            row = dict(r.stats)
            row["seed"] = int(seed)
            row["m_defects"] = int(m)
            row["final_epoch"] = int(ep)
            row["defect_active_faces_final"] = int(sum(1 for f in snap.active_faces if snap.face_defect.get(f, False)))
            row["source_strategy_final"] = str(curve["source_strategy"].iloc[0])
            row["source_faces_count_final"] = int(curve["source_faces_count"].iloc[0])
            row["center_faces_count_final"] = int(curve["center_faces_count"].iloc[0])
            row["landmark_faces_count_final"] = int(curve["landmark_faces_count"].iloc[0])
            row["tagged_defect_area_total_final"] = float(curve["tagged_defect_area_total"].iloc[0])
            row["component_weighted_area_final"] = float(curve["component_weighted_area"].iloc[0])
            row.setdefault("forced_defect_splits", 0)
            row.setdefault("forced_stitch_splits", 0)
            row.setdefault("omitted_weighted_area", 0.0)
            run_rows.append(row)
            print({k: row.get(k) for k in ["seed", "m_defects", "final_epoch", "basin_splits", "forced_stitch_splits", "omitted_weighted_area", "defect_active_faces_final", "source_strategy_final", "source_faces_count_final", "final_active_faces"]})

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

    per_seed, ensemble_summary, fit_table, verdict = summarize_coherent_stitch_paired(paired, run_df, radial_windows=radial_windows)

    stem = str(label).replace(" ", "_")
    paths = {
        "curves": OUT / f"{stem}_anchor_curves.csv",
        "paired": OUT / f"{stem}_paired_deficits.csv",
        "run_summary": OUT / f"{stem}_run_summary.csv",
        "per_seed_summary": OUT / f"{stem}_per_seed_window_summary.csv",
        "ensemble_summary": OUT / f"{stem}_ensemble_window_summary.csv",
        "fit_table": OUT / f"{stem}_fit_table.csv",
        "verdict": OUT / f"{stem}_verdict.csv",
    }
    curves.to_csv(paths["curves"], index=False)
    paired.to_csv(paths["paired"], index=False)
    run_df.to_csv(paths["run_summary"], index=False)
    per_seed.to_csv(paths["per_seed_summary"], index=False)
    ensemble_summary.to_csv(paths["ensemble_summary"], index=False)
    fit_table.to_csv(paths["fit_table"], index=False)
    verdict.to_csv(paths["verdict"], index=False)

    print("\nRun summary:")
    try:
        display(run_df)
    except NameError:
        print(run_df)
    print("\nCoherent stitch ensemble window summary:")
    try:
        display(ensemble_summary)
    except NameError:
        print(ensemble_summary)
    print("\nCoherent stitch fit table:")
    try:
        display(fit_table)
    except NameError:
        print(fit_table)
    print("\nCoherent stitch verdict:")
    try:
        display(verdict)
    except NameError:
        print(verdict)

    if make_plots and not ensemble_summary.empty:
        plot_df = ensemble_summary[ensemble_summary["radial_window"] == "wide_0p25_1p60"].copy()
        if not plot_df.empty:
            plt.figure(figsize=(7, 4))
            plt.errorbar(plot_df["m_defects"], plot_df["median_delta_A_over_R2_mean_over_seeds"], yerr=plot_df["median_delta_A_over_R2_sem_over_seeds"], marker="o", linewidth=1, capsize=3)
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel("median Delta A/R^2 vs m=0")
            plt.title("Coherent topology stitch: area deficit")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_wide_area_deficit.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            paths["plot_area"] = p

            plt.figure(figsize=(7, 4))
            plt.errorbar(plot_df["m_defects"], plot_df["median_delta_C_mean_over_seeds"], yerr=plot_df["median_delta_C_sem_over_seeds"], marker="o", linewidth=1, capsize=3)
            plt.axhline(0, linewidth=1)
            plt.xlabel("m_defects")
            plt.ylabel("median Delta C vs m=0")
            plt.title("Coherent topology stitch: circumference deficit")
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
        "verdict": verdict,
        "paths": paths,
    }
