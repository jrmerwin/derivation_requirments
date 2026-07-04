"""
DEU GR Experiment 01M: Useful-Bandwidth / Lapse Analysis

Purpose
-------
The controlled defect-sink experiments did not produce a robust conical spatial
circumference deficit. However, they still provide a direct test of the lapse /
bandwidth side of the DEU-GR hypothesis.

A forced defect split is counted as a split by the engine, but for a distant/rest
probe it is not useful vacuum generation. Therefore the relevant rest-clock
quantity is not total split rate; it is ordinary/useful split rate after subtracting
the defect-sink work:

    useful_splits = basin_splits - forced_defect_splits
    useful_rate   = useful_splits / epochs
    Phi_useful    = useful_rate(m) / useful_rate(m=0, same seed)
    Omega_useful  = 1 - Phi_useful

This helper accepts the 01J ensemble result dicts, 01I pilot result dicts, fixed-
anchor result dicts, or a run_summary DataFrame, and writes paired/ensemble/fits
CSVs plus simple plots.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _display_df(df):
    try:
        display(df)
    except NameError:
        print(df)


def _sem(s):
    s = pd.Series(s).dropna().astype(float)
    if len(s) <= 1:
        return np.nan
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def _positive_frac(s):
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return np.nan
    return float((s > 0).mean())


def _ols_fit(x, y):
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
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def extract_run_summary_any(result):
    """Accept 01I/01J/01K/01L result dicts or a run-summary DataFrame."""
    if isinstance(result, pd.DataFrame):
        runs = result.copy()
    elif isinstance(result, dict):
        for key in ["run_summary", "runs", "paired_runs"]:
            if key in result and isinstance(result[key], pd.DataFrame):
                runs = result[key].copy()
                break
        else:
            if "runs" in result and isinstance(result["runs"], dict):
                rows = []
                for k, v in result["runs"].items():
                    row = dict(getattr(v, "stats", {}))
                    try:
                        row.setdefault("m_defects", int(k))
                    except Exception:
                        pass
                    rows.append(row)
                runs = pd.DataFrame(rows)
            else:
                raise KeyError(
                    "Could not find a run summary. Expected a DataFrame or a result dict with "
                    "'run_summary', 'runs' as DataFrame, 'paired_runs', or 'runs' as dict."
                )
    else:
        raise TypeError("result must be a DataFrame or a result dictionary")

    # Normalize numeric columns where possible.
    for col in runs.columns:
        runs[col] = pd.to_numeric(runs[col], errors="ignore")

    if "seed" not in runs.columns:
        runs["seed"] = 0
    if "m_defects" not in runs.columns:
        raise KeyError("run summary needs an m_defects column")
    if "epochs" not in runs.columns and "final_epoch" in runs.columns:
        runs["epochs"] = runs["final_epoch"]
    if "final_epoch" not in runs.columns and "epochs" in runs.columns:
        runs["final_epoch"] = runs["epochs"]
    if "basin_splits" not in runs.columns and "target_basin_splits" in runs.columns:
        runs["basin_splits"] = runs["target_basin_splits"]

    required = {"seed", "m_defects", "epochs", "basin_splits"}
    missing = required - set(runs.columns)
    if missing:
        raise KeyError(f"run summary missing required columns: {sorted(missing)}")

    if "forced_defect_splits" not in runs.columns:
        runs["forced_defect_splits"] = 0.0

    runs["forced_defect_splits"] = runs["forced_defect_splits"].fillna(0.0)
    runs["basin_splits"] = pd.to_numeric(runs["basin_splits"], errors="coerce")
    runs["epochs"] = pd.to_numeric(runs["epochs"], errors="coerce")

    return runs


def analyze_useful_lapse_budget(
    result,
    *,
    OUT=None,
    label="useful_lapse_budget",
    min_m_for_fit=2,
    make_plots=True,
):
    """
    Compute DEU lapse/bandwidth drain using useful, non-defect split rate.

    Main quantities:
        ordinary_splits = basin_splits - forced_defect_splits
        useful_rate = ordinary_splits / epochs
        Phi_useful = useful_rate / useful_rate_m0_by_seed
        Omega_useful = 1 - Phi_useful

    Positive Omega_useful means the defect-loaded run leaves less non-defect
    generation available per coordinate epoch than the paired m=0 run.
    """
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    runs = extract_run_summary_any(result).copy()

    runs["ordinary_splits"] = runs["basin_splits"] - runs["forced_defect_splits"].fillna(0.0)
    runs["ordinary_splits"] = runs["ordinary_splits"].clip(lower=0.0)
    runs["total_split_rate"] = runs["basin_splits"] / runs["epochs"].replace(0, np.nan)
    runs["useful_split_rate"] = runs["ordinary_splits"] / runs["epochs"].replace(0, np.nan)
    runs["defect_split_rate"] = runs["forced_defect_splits"] / runs["epochs"].replace(0, np.nan)
    runs["forced_fraction_of_splits"] = runs["forced_defect_splits"] / runs["basin_splits"].replace(0, np.nan)

    base_cols = [
        "seed", "epochs", "basin_splits", "ordinary_splits", "total_split_rate", "useful_split_rate"
    ]
    base = runs[runs["m_defects"] == 0][base_cols].rename(columns={
        "epochs": "epochs_m0",
        "basin_splits": "basin_splits_m0",
        "ordinary_splits": "ordinary_splits_m0",
        "total_split_rate": "total_split_rate_m0",
        "useful_split_rate": "useful_split_rate_m0",
    })

    paired = runs.merge(base, on="seed", how="left")
    paired["delta_epochs_vs_m0"] = paired["epochs"] - paired["epochs_m0"]
    paired["relative_epoch_delay"] = paired["delta_epochs_vs_m0"] / paired["epochs_m0"].replace(0, np.nan)

    paired["Phi_total_rate"] = paired["total_split_rate"] / paired["total_split_rate_m0"].replace(0, np.nan)
    paired["Omega_total_rate"] = 1.0 - paired["Phi_total_rate"]

    paired["Phi_useful_rate"] = paired["useful_split_rate"] / paired["useful_split_rate_m0"].replace(0, np.nan)
    paired["Omega_useful_rate"] = 1.0 - paired["Phi_useful_rate"]

    # Same-target convenience. If target split count is identical across m, E0/E approximates total-rate Phi.
    paired["Phi_epoch_same_target"] = paired["epochs_m0"] / paired["epochs"].replace(0, np.nan)
    paired["Omega_epoch_same_target"] = 1.0 - paired["Phi_epoch_same_target"]

    ycols = [
        "Omega_useful_rate",
        "Omega_total_rate",
        "Omega_epoch_same_target",
        "relative_epoch_delay",
        "forced_fraction_of_splits",
        "Phi_useful_rate",
    ]

    agg_spec = {
        "seeds": ("seed", "nunique"),
        "epochs_mean": ("epochs", "mean"),
        "basin_splits_mean": ("basin_splits", "mean"),
        "ordinary_splits_mean": ("ordinary_splits", "mean"),
        "forced_defect_splits_mean": ("forced_defect_splits", "mean"),
        "useful_split_rate_mean": ("useful_split_rate", "mean"),
        "defect_split_rate_mean": ("defect_split_rate", "mean"),
    }

    for y in ycols:
        agg_spec[f"{y}_mean"] = (y, "mean")
        agg_spec[f"{y}_median"] = (y, "median")
        agg_spec[f"{y}_sem"] = (y, _sem)
        agg_spec[f"{y}_positive_frac"] = (y, _positive_frac)

    for c in ["defect_active_faces_final", "final_active_faces", "final_nodes"]:
        if c in paired.columns:
            agg_spec[f"{c}_mean"] = (c, "mean")

    ensemble = paired.groupby("m_defects").agg(**agg_spec).reset_index()

    fit_rows = []
    for min_m in sorted(set([1, int(min_m_for_fit), 4])):
        sub = paired[paired["m_defects"] >= min_m].copy()
        if sub.empty:
            continue
        for xcol in ["m_defects", "forced_defect_splits", "defect_active_faces_final", "forced_fraction_of_splits"]:
            if xcol not in sub.columns:
                continue
            for ycol in ["Omega_useful_rate", "Omega_total_rate", "relative_epoch_delay", "forced_fraction_of_splits"]:
                if ycol not in sub.columns:
                    continue
                fit_rows.append({"min_m": min_m, "x": xcol, "y": ycol, **_ols_fit(sub[xcol], sub[ycol])})
    fits = pd.DataFrame(fit_rows)

    stem = f"defect_useful_lapse_{label}"
    paired_path = OUT / f"{stem}_paired_runs.csv"
    ens_path = OUT / f"{stem}_ensemble_summary.csv"
    fits_path = OUT / f"{stem}_linear_fits.csv"
    paired.to_csv(paired_path, index=False)
    ensemble.to_csv(ens_path, index=False)
    fits.to_csv(fits_path, index=False)

    print("Paired useful-lapse rows:")
    _display_df(paired)
    print("\nUseful-lapse ensemble summary:")
    _display_df(ensemble)
    print("\nUseful-lapse linear fits:")
    _display_df(fits)

    plot_paths = []
    if make_plots:
        plt.figure(figsize=(7, 4))
        for seed, d in paired.groupby("seed"):
            dd = d.sort_values("m_defects")
            plt.plot(dd["m_defects"], dd["Omega_useful_rate"], marker="o", linewidth=1, label=f"seed={seed}")
        plt.axhline(0, linewidth=1)
        plt.xlabel("controlled sink strength m")
        plt.ylabel("Omega_useful = 1 - useful-rate fraction")
        plt.title(f"Useful bandwidth drain by seed: {label}")
        plt.grid(True, alpha=0.3)
        if paired["seed"].nunique() <= 8:
            plt.legend()
        p = OUT / f"{stem}_omega_useful_by_seed.png"
        plt.savefig(p, dpi=160, bbox_inches="tight")
        plt.show()
        plot_paths.append(p)

        plt.figure(figsize=(7, 4))
        plt.errorbar(
            ensemble["m_defects"],
            ensemble["Omega_useful_rate_mean"],
            yerr=ensemble["Omega_useful_rate_sem"],
            marker="o",
            linewidth=1,
            capsize=3,
        )
        plt.axhline(0, linewidth=1)
        plt.xlabel("controlled sink strength m")
        plt.ylabel("seed-mean Omega_useful")
        plt.title(f"Useful bandwidth drain ensemble: {label}")
        plt.grid(True, alpha=0.3)
        p = OUT / f"{stem}_omega_useful_ensemble.png"
        plt.savefig(p, dpi=160, bbox_inches="tight")
        plt.show()
        plot_paths.append(p)

        if "forced_defect_splits" in paired.columns:
            d = paired[paired["m_defects"] > 0].dropna(subset=["forced_defect_splits", "Omega_useful_rate"])
            plt.figure(figsize=(7, 4))
            plt.scatter(d["forced_defect_splits"], d["Omega_useful_rate"])
            if len(d) >= 3:
                fit = _ols_fit(d["forced_defect_splits"], d["Omega_useful_rate"])
                xs = np.linspace(float(d["forced_defect_splits"].min()), float(d["forced_defect_splits"].max()), 100)
                ys = fit["slope"] * xs + fit["intercept"]
                plt.plot(xs, ys, linewidth=1)
            plt.axhline(0, linewidth=1)
            plt.xlabel("actual forced defect splits")
            plt.ylabel("Omega_useful")
            plt.title(f"Omega_useful vs realized defect work: {label}")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_omega_useful_vs_forced_splits.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            plot_paths.append(p)

    print("\nWrote:")
    for p in [paired_path, ens_path, fits_path, *plot_paths]:
        print(" ", p)

    return {
        "paired_runs": paired,
        "ensemble_summary": ensemble,
        "fit_table": fits,
        "paths": {
            "paired_runs": paired_path,
            "ensemble_summary": ens_path,
            "fit_table": fits_path,
            "plots": plot_paths,
        },
    }
