"""
Experiment 01P: Layer 3 spatial holonomy extraction.

Use on fixed_anchor_* result dictionaries from Experiment 01L.
It extracts the exact area/circumference deficit metrics needed for the
Layer 3 conical-deficit verdict.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


def _safe_label(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def _semantics_for_window(w):
    w = str(w).lower()
    if "inner" in w:
        return "inner"
    if "mid" in w:
        return "mid"
    if "outer" in w:
        return "outer"
    if "wide" in w:
        return "wide"
    return "other"


def _choose_windows(ensemble_summary, requested_windows=None):
    available = list(pd.Series(ensemble_summary["radial_window"].dropna().unique()).astype(str))
    if requested_windows is None:
        requested_windows = [
            "mid_0p60_1p00", "outer_1p00_1p60", "wide_0p15_1p50",
            "mid_0p45_0p90", "outer_0p90_1p50", "wide_0p25_1p60",
        ]

    selected = []
    for rw in requested_windows:
        if rw in available and rw not in selected:
            selected.append(rw)

    # Fallback: include all mid/outer/wide windows actually present.
    if not selected:
        selected = [w for w in available if any(tok in w.lower() for tok in ["mid", "outer", "wide"])]

    return selected, available


def _linear_fit(df, x, y):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 2:
        return dict(x=x, y=y, n=len(d), slope=np.nan, intercept=np.nan, r2=np.nan)
    xv = d[x].astype(float).to_numpy()
    yv = d[y].astype(float).to_numpy()
    if len(np.unique(xv)) < 2:
        return dict(x=x, y=y, n=len(d), slope=np.nan, intercept=np.nan, r2=np.nan)
    slope, intercept = np.polyfit(xv, yv, 1)
    yhat = slope * xv + intercept
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return dict(x=x, y=y, n=len(d), slope=float(slope), intercept=float(intercept), r2=float(r2))


def extract_layer3_holonomy(result, *, OUT=None, label="layer3_holonomy", m_values=(2, 4, 8, 16), requested_windows=None):
    """
    Extract Layer 3 spatial holonomy diagnostics from a fixed-anchor ensemble result.

    Required result keys:
      - ensemble_summary
      - fit_table
    Optional result keys:
      - per_seed_summary
      - paired
    """
    if OUT is None:
        OUT = Path.cwd() / "deu_gr_exp01B_outputs"
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if "ensemble_summary" not in result:
        raise KeyError("result must contain result['ensemble_summary']")
    if "fit_table" not in result:
        raise KeyError("result must contain result['fit_table']")

    ens = result["ensemble_summary"].copy()
    fit = result["fit_table"].copy()
    per_seed = result.get("per_seed_summary", pd.DataFrame()).copy()

    selected_windows, available_windows = _choose_windows(ens, requested_windows=requested_windows)

    print("Available radial windows:", available_windows)
    print("Selected Layer 3 windows:", selected_windows)

    cols = [
        "radial_window", "m_defects", "seeds",
        "median_delta_A_over_R2_mean_over_seeds",
        "median_delta_A_over_R2_sem_over_seeds",
        "median_delta_A_over_R2_seed_frac_positive",
        "median_delta_C_mean_over_seeds",
        "median_delta_C_sem_over_seeds",
        "median_delta_C_seed_frac_positive",
        "forced_defect_splits_mean",
        "defect_active_faces_final_mean",
        "basin_splits_mean",
        "final_active_faces_mean",
    ]
    cols = [c for c in cols if c in ens.columns]

    main = ens[
        ens["radial_window"].astype(str).isin(selected_windows)
        & ens["m_defects"].isin(list(m_values))
    ][cols].copy()

    main["window_class"] = main["radial_window"].apply(_semantics_for_window)

    # Row-level sign tests.
    def col_or_nan(row, col):
        return row[col] if col in row.index else np.nan

    main["A_positive"] = main.get("median_delta_A_over_R2_mean_over_seeds", np.nan) > 0
    main["C_positive"] = main.get("median_delta_C_mean_over_seeds", np.nan) > 0
    main["A_frac_ge_2of3"] = main.get("median_delta_A_over_R2_seed_frac_positive", np.nan) >= (2/3)
    main["C_frac_ge_2of3"] = main.get("median_delta_C_seed_frac_positive", np.nan) >= (2/3)
    main["row_supports_conical"] = main["A_positive"] & main["C_positive"] & main["A_frac_ge_2of3"] & main["C_frac_ge_2of3"]
    main["row_supports_null"] = (~main["A_positive"]) | (~main["C_positive"])

    # Fit extraction from supplied fit_table.
    fit_extract = fit[
        fit["radial_window"].astype(str).isin(selected_windows)
        & fit["x"].eq("m_defects")
        & fit["y"].isin(["median_delta_C", "median_delta_A_over_R2"])
    ].copy()
    if "min_m" in fit_extract.columns:
        fit_extract = fit_extract[fit_extract["min_m"].isin([1, 2, 4, 8])]
    fit_extract["window_class"] = fit_extract["radial_window"].apply(_semantics_for_window)
    fit_extract["slope_positive"] = fit_extract["slope"] > 0
    fit_extract["slope_negative_or_zero"] = fit_extract["slope"] <= 0

    # Conservative per-dataset verdict.
    core = main[main["window_class"].isin(["mid", "outer", "wide"]) & (main["m_defects"] > 0)].copy()
    core_no_m2 = core[core["m_defects"] >= 4].copy()
    fit_core = fit_extract[fit_extract["window_class"].isin(["mid", "outer", "wide"])].copy()
    # Prefer min_m >= 2 fits if present.
    if "min_m" in fit_core.columns and (fit_core["min_m"] >= 2).any():
        fit_core_preferred = fit_core[fit_core["min_m"] >= 2]
    else:
        fit_core_preferred = fit_core

    n_core = len(core)
    n_support = int(core["row_supports_conical"].sum()) if n_core else 0
    n_null = int(core["row_supports_null"].sum()) if n_core else 0
    n_core_no_m2 = len(core_no_m2)
    n_support_no_m2 = int(core_no_m2["row_supports_conical"].sum()) if n_core_no_m2 else 0

    slope_rows = fit_core_preferred[fit_core_preferred["y"].isin(["median_delta_C", "median_delta_A_over_R2"])]
    n_slope = len(slope_rows)
    n_slope_pos = int((slope_rows["slope"] > 0).sum()) if n_slope else 0
    n_slope_neg = int((slope_rows["slope"] <= 0).sum()) if n_slope else 0

    if n_core > 0 and n_support >= math.ceil(0.70 * n_core) and n_slope > 0 and n_slope_pos >= math.ceil(0.70 * n_slope):
        verdict = "CONICAL_DEFICIT_SUPPORTED"
        verdict_reason = "Most mid/outer/wide rows have positive area and circumference deficits and most slopes are positive."
    elif n_core_no_m2 > 0 and n_support_no_m2 == 0 and n_slope > 0 and n_slope_neg >= math.ceil(0.60 * n_slope):
        verdict = "REFINEMENT_SINK_NULL__TOPOLOGICAL_STITCH_REQUIRED"
        verdict_reason = "No m>=4 mid/outer/wide rows support both positive area and circumference deficits; slopes are mostly non-positive."
    else:
        verdict = "MIXED_OR_INCONCLUSIVE"
        verdict_reason = "Signs/slopes do not meet either the conical-confirmation or strong-null threshold."

    verdict_df = pd.DataFrame([{
        "label": label,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "n_core_rows": n_core,
        "n_core_rows_supporting_conical": n_support,
        "n_core_rows_supporting_null": n_null,
        "n_core_m_ge_4_rows": n_core_no_m2,
        "n_core_m_ge_4_supporting_conical": n_support_no_m2,
        "n_fit_slope_rows": n_slope,
        "n_fit_slope_positive": n_slope_pos,
        "n_fit_slope_nonpositive": n_slope_neg,
    }])

    stem = _safe_label(label)
    main_path = OUT / f"{stem}_layer3_holonomy_main_metrics.csv"
    fit_path = OUT / f"{stem}_layer3_holonomy_fit_metrics.csv"
    verdict_path = OUT / f"{stem}_layer3_holonomy_verdict.csv"

    main.to_csv(main_path, index=False)
    fit_extract.to_csv(fit_path, index=False)
    verdict_df.to_csv(verdict_path, index=False)

    print("\nLayer 3 main metrics:")
    try:
        display(main)
    except NameError:
        print(main)

    print("\nLayer 3 fit metrics:")
    try:
        display(fit_extract)
    except NameError:
        print(fit_extract)

    print("\nLayer 3 verdict:")
    try:
        display(verdict_df)
    except NameError:
        print(verdict_df)

    print("\nWrote:")
    print(" ", main_path)
    print(" ", fit_path)
    print(" ", verdict_path)

    return {
        "main_metrics": main,
        "fit_metrics": fit_extract,
        "verdict": verdict_df,
        "paths": {
            "main_metrics": main_path,
            "fit_metrics": fit_path,
            "verdict": verdict_path,
        },
    }


def extract_available_fixed_anchor_results(globals_dict, *, OUT=None):
    """Convenience wrapper to run extraction on all fixed_anchor_* result dicts in the notebook."""
    names = [
        "fixed_anchor_3k_like",
        "fixed_anchor_10k_like",
        "ensemble_3k_5seeds",
        "ensemble_10k_cap512",
    ]
    out = {}
    for name in names:
        obj = globals_dict.get(name)
        if isinstance(obj, dict) and "ensemble_summary" in obj and "fit_table" in obj:
            out[name] = extract_layer3_holonomy(obj, OUT=OUT, label=name)
    if not out:
        raise RuntimeError("No compatible fixed-anchor/ensemble result dictionaries found.")
    return out
