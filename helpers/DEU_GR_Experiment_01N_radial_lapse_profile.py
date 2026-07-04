"""
DEU GR Experiment 01N: Radial Lapse / Local Bandwidth Profile

Purpose
-------
The conical-deficit tests found no robust spatial holonomy from the controlled
split-sink. The useful-budget tests did find a strong global lapse/bandwidth
signal. This helper asks the next diagnostic question:

    Is the lapse signal local around the mass/defect, or just a global cap/budget tax?

It reruns the fixed-epoch, fixed-anchor controlled defect experiment while logging
updates by refinement-weighted radius from the fixed anchor vertex. For each
seed and m value it records, by radial annulus and epoch exposure:

    forced splits      = defect-sink work
    normal splits      = ordinary generative expansion
    sterile ticks      = non-splitting frontier updates
    useful updates     = normal splits + ticks
    useful split-only  = normal splits
    area exposure      = sum over epochs of weighted face area in the annulus

It then pairs every m>0 run against the same-seed m=0 baseline and computes
local lapse estimates:

    Phi_update_density(R) = rho_useful_updates_m(R) / rho_useful_updates_0(R)
    Omega_update_density(R) = 1 - Phi_update_density(R)

and the analogous split-only estimate. A genuine local gravitational lapse should
peak near the defect and weaken outward. A purely global budget theft should be
nearly flat in R.

This is still a controlled-sink experiment, not a native S-G registry defect.
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
class RadialLapseSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_depth: dict
    face_neighbors: dict
    face_defect: dict
    stats: dict


@dataclass
class RadialLapseRun:
    stats: dict
    spatial_snapshots: dict
    epoch_log: pd.DataFrame
    update_log: pd.DataFrame
    exposure_log: pd.DataFrame


def _rl_adj_from_state(faces, edge_to_faces, active):
    def get_neighbors(fid):
        ns = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            ns |= edge_to_faces[frozenset(e)]
        ns.discard(fid)
        return ns & active
    return {fid: get_neighbors(fid) for fid in active}


def _rl_components(adj):
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


def _rl_raw_bulk_center(adj):
    comps = _rl_components(adj)
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


def _rl_largest_component(adj):
    comps = _rl_components(adj)
    return comps[0] if comps else set()


def _rl_weighted_distances_to_anchor(faces, face_depth, adj, active, center_node, *, max_radius=None):
    """
    Dijkstra distances from a fixed anchor vertex to active faces. Source faces
    incident on center_node start at half their local face length.
    """
    active = set(active)
    if not active:
        return {}, {}, {}, []
    comp = _rl_largest_component({f: set(adj.get(f, set())) & active for f in active})
    if not comp:
        return {}, {}, {}, []

    depths = {f: int(face_depth[f]) for f in comp}
    lengths = {f: float(3.0 ** (-0.5 * depths[f])) for f in comp}
    areas = {f: float(3.0 ** (-depths[f])) for f in comp}

    source_faces = [f for f in comp if int(center_node) in faces.get(f, frozenset())]
    if not source_faces:
        return {}, areas, lengths, []

    dist = {s: 0.5 * lengths[s] for s in source_faces}
    heap = [(dist[s], s) for s in source_faces]
    heapq.heapify(heap)

    while heap:
        du, u = heapq.heappop(heap)
        if du != dist.get(u):
            continue
        if max_radius is not None and du > max_radius:
            continue
        for v in adj.get(u, set()) & comp:
            w = 0.5 * (lengths[u] + lengths[v])
            nd = du + w
            if max_radius is not None and nd > max_radius:
                continue
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist, areas, lengths, source_faces


def _rl_bin_index(R, radius_edges):
    if R is None or not np.isfinite(R):
        return None
    i = int(np.searchsorted(radius_edges, float(R), side="right") - 1)
    if i < 0 or i >= len(radius_edges) - 1:
        return None
    return i + 1


def _rl_exposure_rows(epoch, m_defects, seed, cap, final_epoch, dist, areas, radius_edges):
    rows = []
    if not dist:
        return rows
    for i in range(1, len(radius_edges)):
        R0 = float(radius_edges[i - 1])
        R1 = float(radius_edges[i])
        R_mid = 0.5 * (R0 + R1)
        faces_in = [f for f, d in dist.items() if R0 < d <= R1]
        A = float(sum(areas.get(f, 0.0) for f in faces_in))
        rows.append({
            "epoch": int(epoch),
            "seed": int(seed),
            "m_defects": int(m_defects),
            "cap": int(cap),
            "final_epoch_target": int(final_epoch),
            "bin": int(i),
            "R0": R0,
            "R1": R1,
            "R_mid": R_mid,
            "area_exposure": A,
            "face_exposure": int(len(faces_in)),
        })
    return rows


def grow_fixed_epoch_anchor_radial_lapse(
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
    radius_edges=None,
    log_exposure_every_epoch=True,
):
    """
    Fixed-epoch/fixed-anchor controlled defect run with radial update logging.

    Update kinds in update_log:
        marker_split, forced_split, normal_split, tick

    Useful local clock/update proxies:
        useful_updates = normal_split + tick
        useful_splits  = normal_split
    """
    if scheduler != "capped":
        raise ValueError("Only scheduler='capped' is supported.")
    if max_ticks_per_epoch is None:
        max_ticks_per_epoch = max_splits_per_epoch
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    radius_edges = np.asarray(radius_edges, dtype=float)

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
    update_rows = []
    exposure_rows = []
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
        neigh0 = _rl_adj_from_state(faces, edge_to_faces, active0)
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
        spatial_snapshots[int(ep)] = RadialLapseSnapshot(
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

    def current_radial_context(ep):
        if anchor_center_node is None:
            return {}, {}, {}, []
        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()
        dist, areas, lengths, source_faces = _rl_weighted_distances_to_anchor(
            faces0, depth0, neigh0, active0, anchor_center_node,
            max_radius=float(np.max(radius_edges))
        )
        return dist, areas, lengths, source_faces

    def log_update(ep, update_kind, fid, dist, areas):
        R = dist.get(fid, np.nan)
        b = _rl_bin_index(R, radius_edges)
        update_rows.append({
            "epoch": int(ep),
            "seed": int(seed),
            "m_defects": int(m_defects),
            "cap": int(max_splits_per_epoch),
            "final_epoch_target": int(final_epoch),
            "update_kind": str(update_kind),
            "face_id_before": int(fid) if fid is not None else -1,
            "face_type_before": str(face_types.get(fid, "")) if fid in face_types else "",
            "face_depth_before": int(face_depth.get(fid, -1)) if fid in face_depth else -1,
            "face_area_before": float(areas.get(fid, np.nan)) if fid is not None else np.nan,
            "R_anchor_weighted": float(R) if np.isfinite(R) else np.nan,
            "bin": int(b) if b is not None else -1,
            "R0": float(radius_edges[b - 1]) if b is not None else np.nan,
            "R1": float(radius_edges[b]) if b is not None else np.nan,
            "R_mid": float(0.5 * (radius_edges[b - 1] + radius_edges[b])) if b is not None else np.nan,
        })

    def maybe_log_exposure(ep, dist, areas):
        if anchor_center_node is None or not log_exposure_every_epoch:
            return
        exposure_rows.extend(_rl_exposure_rows(
            ep, m_defects, seed, max_splits_per_epoch, final_epoch, dist, areas, radius_edges
        ))

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
        adj = _rl_adj_from_state(faces, edge_to_faces, active)
        center, raw_clearance, boundary_faces = _rl_raw_bulk_center(adj)
        anchor_face = int(center)
        anchor_nodes = tuple(sorted(int(x) for x in faces[center]))

        # Log marker at R=0-ish by computing context before the marker has a center node.
        anchor_center_node = split_face(center, forced=False, force_defect_children=True)
        if anchor_center_node is None:
            raise RuntimeError("Marker split failed unexpectedly")
        stats["marker_anchor_split"] += 1
        stats["defect_anchor_face"] = int(anchor_face)
        stats["defect_center_node"] = int(anchor_center_node)
        stats["defect_anchor_raw_clearance"] = int(raw_clearance)
        stats["defect_boundary_faces_at_injection"] = int(boundary_faces)
        stats["defect_inject_epoch"] = int(ep)

        # Now a center exists. Place marker split in the innermost bin for accounting.
        update_rows.append({
            "epoch": int(ep),
            "seed": int(seed),
            "m_defects": int(m_defects),
            "cap": int(max_splits_per_epoch),
            "final_epoch_target": int(final_epoch),
            "update_kind": "marker_split",
            "face_id_before": int(center),
            "face_type_before": "anchor_marker",
            "face_depth_before": -1,
            "face_area_before": np.nan,
            "R_anchor_weighted": 0.0,
            "bin": 1,
            "R0": float(radius_edges[0]),
            "R1": float(radius_edges[1]),
            "R_mid": float(0.5 * (radius_edges[0] + radius_edges[1])),
        })
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

        # Context after marker injection, before forced/normal/tick updates this epoch.
        dist_pre, areas_pre, lengths_pre, source_faces_pre = current_radial_context(epoch)
        if anchor_center_node is not None:
            maybe_log_exposure(epoch, dist_pre, areas_pre)

        active0, types0, depth0, faces0, neigh0, defect0 = snapshot_raw()

        actual_forced = 0
        if int(m_defects) > 0 and anchor_center_node is not None and budget_remaining > 0:
            tagged = [fid for fid in active0 if defect0.get(fid, False)]
            rng.shuffle(tagged)
            tagged = sorted(tagged, key=lambda f: depth0[f], reverse=True)
            forced_selected = tagged[: min(int(m_defects), int(budget_remaining))]
            for fid in forced_selected:
                if budget_remaining <= 0:
                    break
                if fid not in active:
                    continue
                log_update(epoch, "forced_split", fid, dist_pre, areas_pre)
                if split_face(fid, forced=True, force_defect_children=True) is not None:
                    actual_forced += 1
                    budget_remaining -= 1

        # Recompute after forced splits. Normal/tick candidates should be classified by
        # the state they actually see after the defect has consumed its share.
        dist_mid, areas_mid, lengths_mid, source_faces_mid = current_radial_context(epoch)
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
                if fid not in active:
                    continue
                log_update(epoch, "normal_split", fid, dist_mid, areas_mid)
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
                log_update(epoch, "tick", fid, dist_mid, areas_mid)
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
    stats["defect_active_faces_final"] = int(sum(1 for f in active if face_defect.get(f, False)))

    if record_final and int(final_epoch) not in spatial_snapshots:
        record_snapshot(int(final_epoch))

    return RadialLapseRun(
        stats=dict(stats),
        spatial_snapshots=dict(sorted(spatial_snapshots.items())),
        epoch_log=pd.DataFrame(epoch_log),
        update_log=pd.DataFrame(update_rows),
        exposure_log=pd.DataFrame(exposure_rows),
    )


def _sem(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) <= 1:
        return np.nan
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def _pos_frac(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) == 0:
        return np.nan
    return float((s > 0).mean())


def _ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or len(np.unique(x)) < 2:
        return {"n": int(len(x)), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def summarize_radial_lapse_runs(runs, *, OUT=None, label="radial_lapse", make_plots=True):
    """
    Summarize a dict keyed by (seed, m) or list of RadialLapseRun objects.
    Returns run_summary, radial_rates, paired_radial, ensemble_radial, fit_table.
    """
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if isinstance(runs, dict):
        run_items = list(runs.items())
    else:
        run_items = [(i, r) for i, r in enumerate(runs)]

    run_rows = []
    update_parts = []
    exposure_parts = []
    for key, r in run_items:
        row = dict(r.stats)
        # normalize seed if missing from stats
        if "seed" not in row:
            if isinstance(key, tuple) and len(key) >= 1:
                row["seed"] = key[0]
            elif not r.update_log.empty and "seed" in r.update_log.columns:
                row["seed"] = int(r.update_log["seed"].iloc[0])
            else:
                row["seed"] = 0
        run_rows.append(row)
        if not r.update_log.empty:
            update_parts.append(r.update_log.copy())
        if not r.exposure_log.empty:
            exposure_parts.append(r.exposure_log.copy())

    run_summary = pd.DataFrame(run_rows)
    updates = pd.concat(update_parts, ignore_index=True) if update_parts else pd.DataFrame()
    exposure = pd.concat(exposure_parts, ignore_index=True) if exposure_parts else pd.DataFrame()

    if updates.empty or exposure.empty:
        raise RuntimeError("Need nonempty update_log and exposure_log to summarize radial lapse.")

    # Drop out-of-range updates.
    updates = updates[updates["bin"] > 0].copy()

    update_counts = updates.groupby(["seed", "m_defects", "bin", "R_mid", "update_kind"]).size().unstack(fill_value=0).reset_index()
    for col in ["marker_split", "forced_split", "normal_split", "tick"]:
        if col not in update_counts.columns:
            update_counts[col] = 0

    exp = exposure.groupby(["seed", "m_defects", "bin", "R_mid"]).agg(
        area_exposure=("area_exposure", "sum"),
        face_exposure=("face_exposure", "sum"),
        epochs_observed=("epoch", "nunique"),
    ).reset_index()

    radial = exp.merge(update_counts, on=["seed", "m_defects", "bin", "R_mid"], how="left")
    for col in ["marker_split", "forced_split", "normal_split", "tick"]:
        radial[col] = radial[col].fillna(0.0)

    radial["useful_updates"] = radial["normal_split"] + radial["tick"]
    radial["useful_splits"] = radial["normal_split"]
    radial["total_logged_updates"] = radial["marker_split"] + radial["forced_split"] + radial["normal_split"] + radial["tick"]

    radial["useful_update_density"] = radial["useful_updates"] / radial["area_exposure"].replace(0, np.nan)
    radial["useful_split_density"] = radial["useful_splits"] / radial["area_exposure"].replace(0, np.nan)
    radial["forced_density"] = radial["forced_split"] / radial["area_exposure"].replace(0, np.nan)
    radial["forced_fraction_logged"] = radial["forced_split"] / radial["total_logged_updates"].replace(0, np.nan)

    base = radial[radial["m_defects"] == 0][[
        "seed", "bin", "R_mid", "useful_update_density", "useful_split_density", "useful_updates", "useful_splits"
    ]].rename(columns={
        "useful_update_density": "useful_update_density_m0",
        "useful_split_density": "useful_split_density_m0",
        "useful_updates": "useful_updates_m0",
        "useful_splits": "useful_splits_m0",
    })

    paired = radial.merge(base, on=["seed", "bin", "R_mid"], how="left")
    paired["Phi_update_density"] = paired["useful_update_density"] / paired["useful_update_density_m0"].replace(0, np.nan)
    paired["Omega_update_density"] = 1.0 - paired["Phi_update_density"]
    paired["Phi_split_density"] = paired["useful_split_density"] / paired["useful_split_density_m0"].replace(0, np.nan)
    paired["Omega_split_density"] = 1.0 - paired["Phi_split_density"]
    paired["delta_useful_updates"] = paired["useful_updates_m0"] - paired["useful_updates"]
    paired["delta_useful_splits"] = paired["useful_splits_m0"] - paired["useful_splits"]

    ensemble = paired.groupby(["m_defects", "bin", "R_mid"]).agg(
        seeds=("seed", "nunique"),
        Omega_update_density_mean=("Omega_update_density", "mean"),
        Omega_update_density_sem=("Omega_update_density", _sem),
        Omega_update_density_positive_frac=("Omega_update_density", _pos_frac),
        Omega_split_density_mean=("Omega_split_density", "mean"),
        Omega_split_density_sem=("Omega_split_density", _sem),
        Omega_split_density_positive_frac=("Omega_split_density", _pos_frac),
        forced_density_mean=("forced_density", "mean"),
        forced_fraction_logged_mean=("forced_fraction_logged", "mean"),
        area_exposure_mean=("area_exposure", "mean"),
    ).reset_index()

    # Add run-level load metrics to paired for fits.
    load_cols = ["seed", "m_defects", "forced_defect_splits", "defect_active_faces_final", "basin_splits", "epochs"]
    load_cols = [c for c in load_cols if c in run_summary.columns]
    paired_fit = paired.merge(run_summary[load_cols].drop_duplicates(), on=["seed", "m_defects"], how="left")

    fit_rows = []
    radial_windows = {
        "inner_0p15_0p45": (0.15, 0.45),
        "mid_0p45_0p90": (0.45, 0.90),
        "outer_0p90_1p50": (0.90, 1.50),
        "wide_0p15_1p50": (0.15, 1.50),
    }
    for wname, (lo, hi) in radial_windows.items():
        subw = paired_fit[(paired_fit["R_mid"] >= lo) & (paired_fit["R_mid"] <= hi)].copy()
        # Collapse each seed/m/window to median to avoid overcounting annuli.
        cols = ["seed", "m_defects"]
        keep_loads = [c for c in ["forced_defect_splits", "defect_active_faces_final"] if c in subw.columns]
        agg = {
            "Omega_update_density": "median",
            "Omega_split_density": "median",
            "forced_density": "median",
            "forced_fraction_logged": "median",
        }
        for c in keep_loads:
            agg[c] = "first"
        comp = subw.groupby(cols).agg(agg).reset_index()
        for min_m in [1, 2, 4]:
            s = comp[comp["m_defects"] >= min_m]
            for x in ["m_defects"] + keep_loads:
                for y in ["Omega_update_density", "Omega_split_density"]:
                    fit_rows.append({"radial_window": wname, "min_m": int(min_m), "x": x, "y": y, **_ols(s[x], s[y])})
    fit_table = pd.DataFrame(fit_rows)

    stem = f"radial_lapse_{label}"
    paths = {
        "run_summary": OUT / f"{stem}_run_summary.csv",
        "updates": OUT / f"{stem}_update_log.csv",
        "exposure": OUT / f"{stem}_exposure_log.csv",
        "radial_rates": OUT / f"{stem}_radial_rates.csv",
        "paired_radial": OUT / f"{stem}_paired_radial_lapse.csv",
        "ensemble_radial": OUT / f"{stem}_ensemble_radial_lapse.csv",
        "fits": OUT / f"{stem}_fits.csv",
    }
    run_summary.to_csv(paths["run_summary"], index=False)
    updates.to_csv(paths["updates"], index=False)
    exposure.to_csv(paths["exposure"], index=False)
    radial.to_csv(paths["radial_rates"], index=False)
    paired.to_csv(paths["paired_radial"], index=False)
    ensemble.to_csv(paths["ensemble_radial"], index=False)
    fit_table.to_csv(paths["fits"], index=False)

    print("Wrote radial lapse outputs:")
    for p in paths.values():
        print(" ", p)

    try:
        display(run_summary)
        display(ensemble.head(80))
        display(fit_table)
    except NameError:
        print(run_summary)
        print(ensemble.head(80))
        print(fit_table)

    if make_plots:
        for y, ylabel in [
            ("Omega_update_density_mean", "Omega_update_density"),
            ("Omega_split_density_mean", "Omega_split_density"),
            ("forced_density_mean", "forced update density"),
        ]:
            plt.figure(figsize=(7, 4))
            for m, g in ensemble[ensemble["m_defects"] > 0].groupby("m_defects"):
                gg = g.sort_values("R_mid")
                plt.plot(gg["R_mid"], gg[y], marker="o", linewidth=1, label=f"m={m}")
            plt.axhline(0.0, linewidth=1, linestyle="--")
            plt.xlabel("weighted radius from fixed anchor")
            plt.ylabel(ylabel)
            plt.title(f"Radial lapse profile: {label}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            p = OUT / f"{stem}_{y}.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            print("Wrote:", p)

    return {
        "run_summary": run_summary,
        "updates": updates,
        "exposure": exposure,
        "radial_rates": radial,
        "paired_radial": paired,
        "ensemble_radial": ensemble,
        "fit_table": fit_table,
        "paths": paths,
    }


def run_radial_lapse_ensemble(
    *,
    OUT=None,
    final_epoch=37,
    cap=256,
    seeds=(101, 202, 303, 404, 505),
    m_values=(0, 2, 4, 8),
    defect_inject_epoch=15,
    radius_edges=None,
    label=None,
    make_plots=True,
):
    """Run and summarize the radial lapse ensemble."""
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if radius_edges is None:
        radius_edges = np.linspace(0.05, 1.8, 36)
    if label is None:
        label = f"cap{cap}_epoch{final_epoch}_{len(seeds)}seeds"

    runs = {}
    for seed in seeds:
        for m in m_values:
            print(f"\nRunning radial-lapse fixed-anchor: seed={seed}, m={m}, cap={cap}, epoch={final_epoch}")
            r = grow_fixed_epoch_anchor_radial_lapse(
                final_epoch=final_epoch,
                seed=int(seed),
                max_splits_per_epoch=int(cap),
                max_ticks_per_epoch=int(cap),
                m_defects=int(m),
                defect_inject_epoch=int(defect_inject_epoch),
                snapshot_every=max(1, int(final_epoch)),
                record_initial=True,
                record_final=True,
                radius_edges=radius_edges,
            )
            r.stats["seed"] = int(seed)
            runs[(int(seed), int(m))] = r

    result = summarize_radial_lapse_runs(runs, OUT=OUT, label=label, make_plots=make_plots)
    result["runs"] = runs
    return result
