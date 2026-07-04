"""
Overflow-safe patch for deu_exp456_minimal.py broad multi-gap diamond sampling.

Use one of these AFTER loading deu_exp456_minimal.py:
    %run -i deu_exp456_overflow_patch.py
or:
    exec(open("deu_exp456_overflow_patch.py").read(), globals())

Do NOT use plain import for this patch unless you manually inject it into the
minimal-loader namespace.  The broad runner resolves helper functions from the
namespace in which it was originally defined.
"""

import numpy as np
import pandas as pd


def _diamond_records_to_object_df(records):
    """Build a diamond DataFrame while forcing interval_bits to object dtype."""
    columns = [
        "gap_bin", "gap_lo", "gap_hi", "gap_mid",
        "a_index", "b_index", "a_event", "b_event",
        "ep_a", "ep_b", "epoch_gap",
        "interval_bits", "V_strict_prescan",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    scalar_cols = [c for c in columns if c != "interval_bits"]
    scalar_rows = []
    interval_bits_values = []

    for rec in records:
        scalar_rows.append({c: rec.get(c, np.nan) for c in scalar_cols})
        interval_bits_values.append(int(rec.get("interval_bits", 0)))

    # No huge ints appear in this constructor.
    df = pd.DataFrame(scalar_rows, columns=scalar_cols)

    # Insert huge Python integers as object dtype so Pandas never tries int64/float64.
    insert_at = columns.index("interval_bits")
    df.insert(insert_at, "interval_bits", pd.Series(interval_bits_values, dtype=object))
    return df.loc[:, columns]


def sample_diamonds_by_epoch_gaps(
    causet,
    gap_bins=((2, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 48), (48, 64)),
    n_endpoint_samples_per_bin=25000,
    keep_top_per_bin=200,
    min_size=2,
    seed=0,
):
    """
    Overflow-safe replacement for the broad epoch-gap diamond sampler.

    Difference from the original:
      - per-bin top-k and endpoint deduplication are done in Python lists
      - interval_bits is inserted into the final DataFrame as object dtype

    This avoids:
        OverflowError: int too large to convert to float
    when Pandas sees enormous bitset integers.
    """
    rng = np.random.default_rng(seed)
    eps, range_bits = _epoch_range_bit_indexer(causet)
    ep_max = float(np.nanmax(eps)) if len(eps) else 0.0

    all_kept_records = []

    for gi, (gap_lo, gap_hi) in enumerate(gap_bins):
        gap_lo = float(gap_lo)
        gap_hi = float(gap_hi)
        if gap_hi < gap_lo:
            raise ValueError("gap bins must have gap_hi >= gap_lo")

        eligible_a = np.where(eps <= ep_max - gap_lo)[0]
        if len(eligible_a) == 0:
            continue

        records = []
        for _ in range(int(n_endpoint_samples_per_bin)):
            a = int(eligible_a[rng.integers(len(eligible_a))])
            b_epoch_bits = range_bits(eps[a] + gap_lo, eps[a] + gap_hi)
            candidate_b_bits = int(causet.descendant_bits[a] & b_epoch_bits)
            if candidate_b_bits == 0:
                continue

            b = _random_set_bit(candidate_b_bits, rng)
            if b is None:
                continue

            strict_bits = int(causet.descendant_bits[a] & causet.ancestor_bits[b])
            V = strict_bits.bit_count()
            if V < min_size:
                continue

            records.append({
                "gap_bin": int(gi),
                "gap_lo": gap_lo,
                "gap_hi": gap_hi,
                "gap_mid": 0.5 * (gap_lo + gap_hi),
                "a_index": int(a),
                "b_index": int(b),
                "a_event": causet.events[a],
                "b_event": causet.events[b],
                "ep_a": float(eps[a]),
                "ep_b": float(eps[b]),
                "epoch_gap": float(eps[b] - eps[a]),
                "interval_bits": strict_bits,
                "V_strict_prescan": int(V),
            })

        if not records:
            continue

        # Top-k + dedupe without constructing a temporary DataFrame containing giant ints.
        records.sort(key=lambda r: r["V_strict_prescan"], reverse=True)
        seen_pairs = set()
        kept = []
        for rec in records:
            pair = (rec["a_index"], rec["b_index"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            kept.append(rec)
            if len(kept) >= int(keep_top_per_bin):
                break

        all_kept_records.extend(kept)

    if not all_kept_records:
        return _diamond_records_to_object_df([])

    all_kept_records.sort(key=lambda r: (r["gap_bin"], -r["V_strict_prescan"], r["a_index"], r["b_index"]))
    return _diamond_records_to_object_df(all_kept_records)


# Patch the namespace used by run_broad_multigap_causal_scaling if it already exists.
try:
    _target_globals = run_broad_multigap_causal_scaling.__globals__
    # Make the helper dependencies visible to this function when the patch is executed in-place.
    for _name in ("_epoch_range_bit_indexer", "_random_set_bit"):
        if _name not in globals() and _name in _target_globals:
            globals()[_name] = _target_globals[_name]
    _target_globals["_diamond_records_to_object_df"] = _diamond_records_to_object_df
    _target_globals["sample_diamonds_by_epoch_gaps"] = sample_diamonds_by_epoch_gaps
except NameError:
    pass

print("Loaded overflow-safe sample_diamonds_by_epoch_gaps patch. interval_bits will use object dtype.")
