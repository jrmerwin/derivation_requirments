"""
DEU GR Experiment 01O: Radial Lapse Residual / Local-vs-Global Diagnostic

Purpose
-------
Experiment 01N measures a robust useful-bandwidth lapse profile around a fixed
anchor, but the profile can mix two effects:

    1. a global scheduler/cap tax: every radius is slowed by about the same factor;
    2. a genuinely local gravitational-lapse field: extra slowdown near the defect
       that decays outward.

This helper decomposes the 01N radial-lapse output into:

    Omega_local(R)
    Omega_global_for_same_seed_and_m
    Omega_residual(R) = Omega_local(R) - Omega_global

A local GR-like lapse should show positive inner residuals and an inner-minus-outer
gradient that grows with defect load. A pure global cap tax should have residuals
near zero with no stable radial gradient.

Typical notebook use
--------------------
    exec(open(BASE / "DEU_GR_Experiment_01O_radial_lapse_residual.py", encoding="utf-8").read(), globals())

    residual_cap256 = analyze_radial_lapse_residual(
        radial_lapse_cap256,
        OUT=OUT,
        label="cap256_epoch37_5seeds",
    )

    display(residual_cap256["window_summary"])
    display(residual_cap256["gradient_summary"])
    display(residual_cap256["gradient_fits"])
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _display(df):
    try:
        display(df)
    except NameError:
        print(df)


def analyze_radial_lapse_residual(
    radial_lapse_result,
    *,
    OUT=None,
    label="radial_lapse_residual",
    radial_windows=None,
    make_plots=True,
):
    """
    Analyze Experiment 01N output by subtracting the same-seed/m global radial lapse.

    Parameters
    ----------
    radial_lapse_result : dict
        Output returned by run_radial_lapse_ensemble / summarize_radial_lapse_runs.
    OUT : path-like
        Output directory.
    label : str
        Stem for saved files.
    radial_windows : dict or None
        Windows over R_mid for summary/gradient diagnostics.
    make_plots : bool
        Save radial Omega and residual plots.

    Returns
    -------
    dict with global_rates, paired_with_residual, ensemble_residual,
    window_by_seed, window_summary, gradient_by_seed, gradient_summary, gradient_fits.
    """
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if radial_windows is None:
        radial_windows = {
            "inner_0p15_0p45": (0.15, 0.45),
            "mid_0p45_0p90": (0.45, 0.90),
            "outer_0p90_1p50": (0.90, 1.50),
            "wide_0p15_1p50": (0.15, 1.50),
        }

    required = ["radial_rates", "paired_radial", "run_summary"]
    missing = [k for k in required if k not in radial_lapse_result]
    if missing:
        raise KeyError(f"radial_lapse_result is missing keys: {missing}")

    radial = radial_lapse_result["radial_rates"].copy()
    paired = radial_lapse_result["paired_radial"].copy()
    run_summary = radial_lapse_result["run_summary"].copy()

    # Normalize useful columns if needed.
    if "useful_updates" not in radial.columns:
        radial["useful_updates"] = radial.get("normal_split", 0) + radial.get("tick", 0)
    if "useful_splits" not in radial.columns:
        radial["useful_splits"] = radial.get("normal_split", 0)
    if "forced_split" not in radial.columns:
        radial["forced_split"] = 0.0

    # Coverage-weighted global density inside the measured radial aperture.
    glob = radial.groupby(["seed", "m_defects"]).agg(
        useful_updates=("useful_updates", "sum"),
        useful_splits=("useful_splits", "sum"),
        forced_split=("forced_split", "sum"),
        area_exposure=("area_exposure", "sum"),
        face_exposure=("face_exposure", "sum"),
    ).reset_index()

    glob["rho_update_global"] = glob["useful_updates"] / glob["area_exposure"].replace(0, np.nan)
    glob["rho_split_global"] = glob["useful_splits"] / glob["area_exposure"].replace(0, np.nan)
    glob["rho_forced_global"] = glob["forced_split"] / glob["area_exposure"].replace(0, np.nan)

    base = glob[glob["m_defects"] == 0][[
        "seed", "rho_update_global", "rho_split_global", "useful_updates", "useful_splits", "area_exposure"
    ]].rename(columns={
        "rho_update_global": "rho_update_global_m0",
        "rho_split_global": "rho_split_global_m0",
        "useful_updates": "useful_updates_global_m0",
        "useful_splits": "useful_splits_global_m0",
        "area_exposure": "area_exposure_global_m0",
    })

    glob = glob.merge(base, on="seed", how="left")
    glob["Phi_update_global"] = glob["rho_update_global"] / glob["rho_update_global_m0"].replace(0, np.nan)
    glob["Omega_update_global"] = 1.0 - glob["Phi_update_global"]
    glob["Phi_split_global"] = glob["rho_split_global"] / glob["rho_split_global_m0"].replace(0, np.nan)
    glob["Omega_split_global"] = 1.0 - glob["Phi_split_global"]

    # Attach global values to each local annulus.
    cols = ["seed", "m_defects", "Omega_update_global", "Omega_split_global", "rho_forced_global"]
    paired2 = paired.merge(glob[cols], on=["seed", "m_defects"], how="left")

    paired2["Omega_update_residual"] = paired2["Omega_update_density"] - paired2["Omega_update_global"]
    paired2["Omega_split_residual"] = paired2["Omega_split_density"] - paired2["Omega_split_global"]

    # Attach run-level load metrics.
    load_cols = [
        "seed", "m_defects", "forced_defect_splits", "defect_active_faces_final",
        "basin_splits", "epochs", "final_active_faces", "final_nodes",
    ]
    load_cols = [c for c in load_cols if c in run_summary.columns]
    paired2 = paired2.merge(run_summary[load_cols].drop_duplicates(), on=["seed", "m_defects"], how="left")

    ensemble = paired2.groupby(["m_defects", "bin", "R_mid"]).agg(
        seeds=("seed", "nunique"),
        Omega_update_density_mean=("Omega_update_density", "mean"),
        Omega_update_density_sem=("Omega_update_density", _sem),
        Omega_update_density_positive_frac=("Omega_update_density", _pos_frac),
        Omega_update_global_mean=("Omega_update_global", "mean"),
        Omega_update_residual_mean=("Omega_update_residual", "mean"),
        Omega_update_residual_sem=("Omega_update_residual", _sem),
        Omega_update_residual_positive_frac=("Omega_update_residual", _pos_frac),
        Omega_split_density_mean=("Omega_split_density", "mean"),
        Omega_split_density_sem=("Omega_split_density", _sem),
        Omega_split_global_mean=("Omega_split_global", "mean"),
        Omega_split_residual_mean=("Omega_split_residual", "mean"),
        Omega_split_residual_sem=("Omega_split_residual", _sem),
        Omega_split_residual_positive_frac=("Omega_split_residual", _pos_frac),
        forced_density_mean=("forced_density", "mean"),
        area_exposure_mean=("area_exposure", "mean"),
    ).reset_index()

    # Window summaries by seed/m, then ensemble.
    wrows = []
    for wname, (lo, hi) in radial_windows.items():
        sub = paired2[(paired2["R_mid"] >= lo) & (paired2["R_mid"] <= hi)].copy()
        if sub.empty:
            continue
        agg = {
            "Omega_update_density": "median",
            "Omega_update_global": "first",
            "Omega_update_residual": "median",
            "Omega_split_density": "median",
            "Omega_split_global": "first",
            "Omega_split_residual": "median",
            "forced_density": "median",
            "area_exposure": "sum",
        }
        for c in ["forced_defect_splits", "defect_active_faces_final", "basin_splits", "epochs"]:
            if c in sub.columns:
                agg[c] = "first"
        comp = sub.groupby(["seed", "m_defects"]).agg(agg).reset_index()
        comp["radial_window"] = wname
        comp["R_lo"] = float(lo)
        comp["R_hi"] = float(hi)
        wrows.append(comp)

    window_by_seed = pd.concat(wrows, ignore_index=True) if wrows else pd.DataFrame()

    window_summary = window_by_seed.groupby(["radial_window", "m_defects"]).agg(
        seeds=("seed", "nunique"),
        Omega_update_density_mean=("Omega_update_density", "mean"),
        Omega_update_density_sem=("Omega_update_density", _sem),
        Omega_update_density_positive_frac=("Omega_update_density", _pos_frac),
        Omega_update_global_mean=("Omega_update_global", "mean"),
        Omega_update_residual_mean=("Omega_update_residual", "mean"),
        Omega_update_residual_sem=("Omega_update_residual", _sem),
        Omega_update_residual_positive_frac=("Omega_update_residual", _pos_frac),
        Omega_split_density_mean=("Omega_split_density", "mean"),
        Omega_split_density_sem=("Omega_split_density", _sem),
        Omega_split_residual_mean=("Omega_split_residual", "mean"),
        Omega_split_residual_sem=("Omega_split_residual", _sem),
        Omega_split_residual_positive_frac=("Omega_split_residual", _pos_frac),
        forced_density_mean=("forced_density", "mean"),
        forced_defect_splits_mean=("forced_defect_splits", "mean") if "forced_defect_splits" in window_by_seed else ("seed", "size"),
        defect_active_faces_final_mean=("defect_active_faces_final", "mean") if "defect_active_faces_final" in window_by_seed else ("seed", "size"),
    ).reset_index()

    # Inner-minus-outer gradients.
    inner_name = "inner_0p15_0p45"
    outer_name = "outer_0p90_1p50"
    if inner_name in set(window_by_seed.get("radial_window", [])) and outer_name in set(window_by_seed.get("radial_window", [])):
        inner = window_by_seed[window_by_seed["radial_window"] == inner_name].copy()
        outer = window_by_seed[window_by_seed["radial_window"] == outer_name].copy()
        common_cols = ["seed", "m_defects"]
        keep = [
            "Omega_update_density", "Omega_update_residual",
            "Omega_split_density", "Omega_split_residual",
            "forced_density",
        ]
        load_keep = [c for c in ["forced_defect_splits", "defect_active_faces_final", "basin_splits", "epochs"] if c in inner.columns]
        left = inner[common_cols + keep + load_keep].rename(columns={c: f"inner_{c}" for c in keep})
        right = outer[common_cols + keep].rename(columns={c: f"outer_{c}" for c in keep})
        grad = left.merge(right, on=common_cols, how="inner")
        for c in keep:
            grad[f"inner_minus_outer_{c}"] = grad[f"inner_{c}"] - grad[f"outer_{c}"]
        gradient_by_seed = grad
    else:
        gradient_by_seed = pd.DataFrame()

    if not gradient_by_seed.empty:
        gradient_summary = gradient_by_seed.groupby("m_defects").agg(
            seeds=("seed", "nunique"),
            inner_minus_outer_update_density_mean=("inner_minus_outer_Omega_update_density", "mean"),
            inner_minus_outer_update_density_sem=("inner_minus_outer_Omega_update_density", _sem),
            inner_minus_outer_update_density_positive_frac=("inner_minus_outer_Omega_update_density", _pos_frac),
            inner_minus_outer_update_residual_mean=("inner_minus_outer_Omega_update_residual", "mean"),
            inner_minus_outer_update_residual_sem=("inner_minus_outer_Omega_update_residual", _sem),
            inner_minus_outer_update_residual_positive_frac=("inner_minus_outer_Omega_update_residual", _pos_frac),
            inner_minus_outer_split_density_mean=("inner_minus_outer_Omega_split_density", "mean"),
            inner_minus_outer_split_density_sem=("inner_minus_outer_Omega_split_density", _sem),
            inner_minus_outer_split_residual_mean=("inner_minus_outer_Omega_split_residual", "mean"),
            inner_minus_outer_split_residual_sem=("inner_minus_outer_Omega_split_residual", _sem),
            forced_defect_splits_mean=("forced_defect_splits", "mean") if "forced_defect_splits" in gradient_by_seed else ("seed", "size"),
            defect_active_faces_final_mean=("defect_active_faces_final", "mean") if "defect_active_faces_final" in gradient_by_seed else ("seed", "size"),
        ).reset_index()

        fit_rows = []
        ycols = [
            "inner_minus_outer_Omega_update_density",
            "inner_minus_outer_Omega_update_residual",
            "inner_minus_outer_Omega_split_density",
            "inner_minus_outer_Omega_split_residual",
        ]
        xcols = ["m_defects", "forced_defect_splits", "defect_active_faces_final"]
        for min_m in [1, 2, 4]:
            sub = gradient_by_seed[gradient_by_seed["m_defects"] >= min_m].copy()
            for x in xcols:
                if x not in sub.columns:
                    continue
                for y in ycols:
                    if y not in sub.columns:
                        continue
                    fit_rows.append({"min_m": int(min_m), "x": x, "y": y, **_ols(sub[x], sub[y])})
        gradient_fits = pd.DataFrame(fit_rows)
    else:
        gradient_summary = pd.DataFrame()
        gradient_fits = pd.DataFrame()

    stem = f"radial_lapse_residual_{label}"
    paths = {
        "global_rates": OUT / f"{stem}_global_rates.csv",
        "paired_with_residual": OUT / f"{stem}_paired_with_residual.csv",
        "ensemble_residual": OUT / f"{stem}_ensemble_residual.csv",
        "window_by_seed": OUT / f"{stem}_window_by_seed.csv",
        "window_summary": OUT / f"{stem}_window_summary.csv",
        "gradient_by_seed": OUT / f"{stem}_gradient_by_seed.csv",
        "gradient_summary": OUT / f"{stem}_gradient_summary.csv",
        "gradient_fits": OUT / f"{stem}_gradient_fits.csv",
    }
    glob.to_csv(paths["global_rates"], index=False)
    paired2.to_csv(paths["paired_with_residual"], index=False)
    ensemble.to_csv(paths["ensemble_residual"], index=False)
    window_by_seed.to_csv(paths["window_by_seed"], index=False)
    window_summary.to_csv(paths["window_summary"], index=False)
    gradient_by_seed.to_csv(paths["gradient_by_seed"], index=False)
    gradient_summary.to_csv(paths["gradient_summary"], index=False)
    gradient_fits.to_csv(paths["gradient_fits"], index=False)

    print("Wrote radial residual outputs:")
    for p in paths.values():
        print(" ", p)

    print("\nGlobal radial-aperture lapse:")
    _display(glob[["seed", "m_defects", "Omega_update_global", "Omega_split_global", "rho_forced_global"]].head(40))
    print("\nWindow summary:")
    _display(window_summary)
    print("\nInner-minus-outer gradient summary:")
    _display(gradient_summary)
    print("\nGradient fits:")
    _display(gradient_fits)

    plot_paths = []
    if make_plots:
        for y, ylabel in [
            ("Omega_update_density_mean", "local Omega_update_density"),
            ("Omega_update_residual_mean", "residual Omega_update_density"),
            ("Omega_split_density_mean", "local Omega_split_density"),
            ("Omega_split_residual_mean", "residual Omega_split_density"),
        ]:
            plt.figure(figsize=(7, 4))
            for m, g in ensemble[ensemble["m_defects"] > 0].groupby("m_defects"):
                gg = g.sort_values("R_mid")
                plt.plot(gg["R_mid"], gg[y], marker="o", linewidth=1, label=f"m={m}")
            plt.axhline(0.0, linewidth=1, linestyle="--")
            plt.xlabel("weighted radius from fixed anchor")
            plt.ylabel(ylabel)
            plt.title(f"Radial lapse residual diagnostic: {label}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            p = OUT / f"{stem}_{y}.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            plot_paths.append(p)
            print("Wrote:", p)

        if not gradient_summary.empty:
            plt.figure(figsize=(7, 4))
            d = gradient_summary[gradient_summary["m_defects"] > 0].sort_values("m_defects")
            plt.errorbar(
                d["m_defects"],
                d["inner_minus_outer_update_residual_mean"],
                yerr=d["inner_minus_outer_update_residual_sem"],
                marker="o", linewidth=1, capsize=3,
            )
            plt.axhline(0.0, linewidth=1, linestyle="--")
            plt.xlabel("controlled defect strength m")
            plt.ylabel("inner - outer residual Omega_update")
            plt.title(f"Local-lapse gradient after global subtraction: {label}")
            plt.grid(True, alpha=0.3)
            p = OUT / f"{stem}_inner_minus_outer_residual.png"
            plt.savefig(p, dpi=160, bbox_inches="tight")
            plt.show()
            plot_paths.append(p)
            print("Wrote:", p)

    return {
        "global_rates": glob,
        "paired_with_residual": paired2,
        "ensemble_residual": ensemble,
        "window_by_seed": window_by_seed,
        "window_summary": window_summary,
        "gradient_by_seed": gradient_by_seed,
        "gradient_summary": gradient_summary,
        "gradient_fits": gradient_fits,
        "paths": paths,
        "plots": plot_paths,
    }
