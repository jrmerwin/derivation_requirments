"""
DEU GR Experiment 01H: Depth Metric Repair

Use this in the DEU_GR_Experiment_01B_SR_Audit_Bridge.ipynb notebook after the
src files have been loaded. It runs the native depth-only spatial replay,
verifies it matches the causal SR run snapshots, attaches face_depth to causal
snapshots, and runs the native refinement-weighted spatial metric audit.
"""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_src_loaded(base: Path):
    """Load the project src files into the caller's globals if needed."""
    src = base / "src"
    for p in [base, src]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    return src


def attach_depth_to_causal_run(causal_run, depth_run, *, strict=True):
    """
    Copy face_depth maps from a SpatialDepthRun into matching causal-run snapshots.
    Returns a DataFrame of alignment checks.
    """
    rows = []
    shared_epochs = sorted(set(causal_run.spatial_snapshots).intersection(depth_run.spatial_snapshots))
    for ep in shared_epochs:
        s = causal_run.spatial_snapshots[ep]
        d = depth_run.spatial_snapshots[ep]
        same_nodes = (s.face_nodes == d.face_nodes)
        same_types = (s.face_types == d.face_types)
        same_neighbors = ({k: set(v) for k, v in s.face_neighbors.items()} ==
                          {k: set(v) for k, v in d.face_neighbors.items()})
        ok = same_nodes and same_types and same_neighbors
        rows.append({
            "epoch": int(ep),
            "causal_faces": len(s.face_nodes),
            "depth_faces": len(d.face_nodes),
            "same_face_nodes": bool(same_nodes),
            "same_face_types": bool(same_types),
            "same_face_neighbors": bool(same_neighbors),
            "attached": bool(ok or not strict),
        })
        if strict and not ok:
            raise AssertionError(f"Depth replay did not match causal snapshot at epoch {ep}: {rows[-1]}")
        s.face_depth = dict(d.face_depth)
    return pd.DataFrame(rows)


def run_depth_metric_repair(
    *,
    BASE=None,
    OUT=None,
    foam_result=None,
    cap=512,
    seed=101,
    target_basin_splits=10000,
    n_centers=32,
):
    """Run the depth metric repair workflow."""
    if BASE is None:
        BASE = Path.cwd()
    BASE = Path(BASE).resolve()
    SRC = _ensure_src_loaded(BASE)
    if OUT is None:
        OUT = BASE / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # These names are expected because the notebook loads src files with exec(..., globals()).
    g = globals()
    missing = [name for name in [
        "grow_typed_foam_spatial_depth_only",
        "audit_weighted_spatial_snapshots",
    ] if name not in g]
    if missing:
        raise RuntimeError(
            "Missing native functions in this namespace: " + ", ".join(missing) +
            ". Run/exec src/deu_exp456_minimal.py first, or run this file with %run -i."
        )

    print("Running native depth-only spatial replay...")
    depth_run = grow_typed_foam_spatial_depth_only(
        target_basin_splits=int(target_basin_splits),
        seed=int(seed),
        scheduler="capped",
        max_splits_per_epoch=int(cap),
        max_ticks_per_epoch=int(cap),
        snapshot_every=10,
        record_final=True,
    )

    print("Depth replay complete:")
    print("  epochs:", depth_run.stats.get("epochs"))
    print("  final active faces:", depth_run.stats.get("final_active_faces"))
    print("  snapshots:", sorted(depth_run.spatial_snapshots))

    alignment_df = pd.DataFrame()
    if foam_result is not None:
        print("\nAttaching depth maps to causal SR run snapshots...")
        causal_run = foam_result["run"] if isinstance(foam_result, dict) else foam_result
        alignment_df = attach_depth_to_causal_run(causal_run, depth_run, strict=True)
        alignment_df.to_csv(OUT / f"depth_replay_alignment_cap{cap}_seed{seed}_{target_basin_splits}.csv", index=False)
        print("Alignment checks:")
        print(alignment_df.to_string(index=False))

    print("\nRunning native weighted spatial audit on depth replay...")
    weighted_rows, depth_rows, weighted_curves = audit_weighted_spatial_snapshots(
        depth_run,
        n_centers=int(n_centers),
        seed=int(seed),
    )

    stem = f"depth_metric_cap{cap}_seed{seed}_{target_basin_splits}"
    weighted_rows.to_csv(OUT / f"{stem}_weighted_spatial_summary.csv", index=False)
    depth_rows.to_csv(OUT / f"{stem}_depth_summary.csv", index=False)
    for ep, curve in weighted_curves.items():
        curve.to_csv(OUT / f"{stem}_curve_epoch{ep}.csv", index=False)

    print("\nWeighted spatial summary:")
    print(weighted_rows.to_string(index=False))
    print("\nDepth summary:")
    print(depth_rows.to_string(index=False))

    # Plot dimension over epoch.
    fig = plt.figure(figsize=(7, 4))
    plt.plot(weighted_rows["epoch"], weighted_rows["weighted_dim"], marker="o", linewidth=1)
    plt.axhline(2.0, linestyle="--", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("weighted spatial dimension")
    plt.title(f"Refinement-weighted spatial dimension, cap {cap}, seed {seed}")
    plt.grid(True, alpha=0.3)
    plot_path = OUT / f"{stem}_weighted_dimension_over_epoch.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("\nWrote plot:", plot_path)

    return {
        "depth_run": depth_run,
        "alignment": alignment_df,
        "weighted_rows": weighted_rows,
        "depth_rows": depth_rows,
        "weighted_curves": weighted_curves,
    }
