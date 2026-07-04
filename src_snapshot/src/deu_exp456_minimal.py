# deu_exp456_minimal.py
# Minimal bootstrap for DEU foam Experiments 4, 5, and 6.
# Extracted from final_MM_attempt.ipynb plus the Experiment 4/5/6 appendix and refinement patch.
# Import or %run this file instead of executing the full notebook.



# --- extracted notebook cell 4 definitions ---

import itertools

from dataclasses import dataclass

from collections import defaultdict, Counter

import numpy as np

import pandas as pd

from scipy.special import gammaln

from scipy.optimize import brentq



# --- extracted notebook cell 5 definitions ---

def r_of_d(d):
    """
    Myrheim-Meyer ordering fraction for a d-dimensional Minkowski Alexandrov interval.

    r(d=1) = 1
    r(d=2) = 1/2
    r(d=3) ≈ 0.2286
    r(d=4) = 0.1
    """
    return 1.5 * np.exp(
        gammaln(d + 1)
        + gammaln(d / 2 + 1)
        - gammaln(3 * d / 2 + 1)
    )

def mm_dim(ordering_fraction, dmax=12.0):
    """
    Invert the Myrheim-Meyer ordering fraction.
    Returns NaN if the observed ordering fraction is below the dmax range.
    """
    r = float(ordering_fraction)

    if r >= 1.0:
        return 1.0

    if r <= r_of_d(dmax):
        return np.nan

    return brentq(lambda d: r_of_d(d) - r, 1.0, dmax)



# --- extracted notebook cell 6 definitions ---

@dataclass
class BitsetCauset:
    events: list
    index: dict
    children: dict
    child_bits: list
    parent_bits: list
    ancestor_bits: list
    descendant_bits: list
    event_role: dict
    event_kind: dict
    event_basin: dict
    event_epoch: dict

def iter_bits(bits):
    """
    Yield the positions of set bits in a Python integer bitset.
    """
    while bits:
        lsb = bits & -bits
        yield lsb.bit_length() - 1
        bits ^= lsb

def bit_at(i):
    return 1 << i

def build_bitset_causet(
    elements,
    children,
    event_role=None,
    event_kind=None,
    event_basin=None,
    event_epoch=None,
):
    """
    Build ancestor/descendant bitsets.

    Assumption: event IDs are topologically ordered.
    The foam generator below satisfies this because every parent is created
    before every child.
    """
    events = sorted(elements)
    index = {e: i for i, e in enumerate(events)}
    n = len(events)

    child_bits = [0] * n
    parent_bits = [0] * n

    for p, cs in children.items():
        if p not in index:
            continue

        pi = index[p]

        for c in cs:
            if c in index:
                ci = index[c]
                child_bits[pi] |= bit_at(ci)
                parent_bits[ci] |= bit_at(pi)

    ancestor_bits = [0] * n

    for i in range(n):
        bits = parent_bits[i]

        for p in iter_bits(parent_bits[i]):
            bits |= ancestor_bits[p]

        ancestor_bits[i] = bits

    descendant_bits = [0] * n

    for i in range(n - 1, -1, -1):
        bits = child_bits[i]

        for c in iter_bits(child_bits[i]):
            bits |= descendant_bits[c]

        descendant_bits[i] = bits

    return BitsetCauset(
        events=events,
        index=index,
        children=children,
        child_bits=child_bits,
        parent_bits=parent_bits,
        ancestor_bits=ancestor_bits,
        descendant_bits=descendant_bits,
        event_role=event_role or {},
        event_kind=event_kind or {},
        event_basin=event_basin or {},
        event_epoch=event_epoch or {},
    )

def count_related_pairs(interval_bits, descendant_bits):
    """
    Count comparable ordered pairs inside an induced interval.

    For every x in interval, count y in interval with x < y.
    This counts each comparable pair exactly once.
    """
    rel = 0
    bits = interval_bits

    while bits:
        lsb = bits & -bits
        i = lsb.bit_length() - 1
        rel += (descendant_bits[i] & interval_bits).bit_count()
        bits ^= lsb

    return rel

def random_bit_index(bits, rng):
    """
    Uniformly sample one set-bit position from a bitset.
    """
    k = int(rng.integers(bits.bit_count()))

    for i in iter_bits(bits):
        if k == 0:
            return i
        k -= 1

    raise RuntimeError("random_bit_index failed unexpectedly.")



# --- extracted notebook cell 7 definitions ---

@dataclass
class FoamRun:
    elements: set
    children: dict
    event_role: dict
    event_kind: dict
    event_basin: dict
    event_epoch: dict
    stats: dict

def grow_typed_foam_causet_parallel(
    target_basin_splits=3000,
    seed=0,
    max_epochs=None,
    max_splits_per_epoch=256,
    max_ticks_per_epoch=256,
):
    """
    Parallel typed foam with explicit basin nodes.

    Causal-set event roles:
        T : sterile temporal driver
        S : resolved mode S
        I : resolved mode I
        G : resolved mode G

    A complete resolved basin is:
        T -> {S, I, G}

    Expansion rule:
        A face splits only if it is an S face touching at least one G face
        and touching no I face.

    Sterile tick rule:
        When no S-face is frustrated, sterile ticks advance local face types.
        The important non-null move is I -> S, which removes screening and
        creates split-capable frontier.

        G -> I is also allowed as part of the stated G -> I -> S hierarchy.
        S -> G is only used as a last-resort perturbation if the foam is stuck.

    Crucial testing correction:
        Updates happen in parallel epochs. Events created in the same epoch are
        not automatically causally related just because Python created them
        sequentially. Parents are computed from the previous epoch snapshot.
    """
    rng = np.random.default_rng(seed)

    faces = {}
    face_types = {}
    face_creator = {}
    face_basin = {}
    edge_to_faces = defaultdict(set)
    active = set()

    causal_parents = defaultdict(set)
    event_role = {}
    event_kind = {}
    event_basin = {}
    event_epoch = {}

    next_face = 0
    next_node = 6
    next_event = 0
    next_basin = 0
    epoch = 0

    stats = Counter()

    def make_event(role, kind, basin, parents=(), ep=None):
        nonlocal next_event

        ev = next_event
        next_event += 1

        ps = {int(p) for p in parents if p is not None and int(p) != ev}
        causal_parents[ev] |= ps

        event_role[ev] = role
        event_kind[ev] = kind
        event_basin[ev] = basin
        event_epoch[ev] = epoch if ep is None else ep

        stats[f"events_{role}"] += 1
        stats[f"kind_{kind}"] += 1
        stats["parent_sum"] += len(ps)
        stats["parent_max"] = max(stats.get("parent_max", 0), len(ps))

        return ev

    def add_face(nodes, ftype, creator, basin):
        nonlocal next_face

        fid = next_face
        next_face += 1

        nodes = frozenset(int(x) for x in nodes)
        faces[fid] = nodes
        face_types[fid] = ftype
        face_creator[fid] = int(creator)
        face_basin[fid] = basin

        for e in itertools.combinations(sorted(nodes), 2):
            edge_to_faces[frozenset(e)].add(fid)

        active.add(fid)

        return fid

    def remove_face(fid):
        for e in itertools.combinations(sorted(faces[fid]), 2):
            edge_to_faces[frozenset(e)].discard(fid)

        active.discard(fid)

        del faces[fid]
        del face_types[fid]
        del face_creator[fid]
        del face_basin[fid]

    def get_neighbors_current(fid):
        neighbors = set()

        for e in itertools.combinations(sorted(faces[fid]), 2):
            neighbors |= edge_to_faces[frozenset(e)]

        neighbors.discard(fid)

        return neighbors

    def snapshot():
        active0 = set(active)
        types0 = dict(face_types)
        creator0 = dict(face_creator)
        basin0 = dict(face_basin)
        faces0 = dict(faces)

        neigh0 = {
            fid: get_neighbors_current(fid) & active0
            for fid in active0
        }

        return active0, types0, creator0, basin0, faces0, neigh0

    def is_frustrated0(fid, types0, neigh0):
        if types0[fid] != "S":
            return False

        neighbor_types = {types0[n] for n in neigh0[fid]}

        return ("G" in neighbor_types) and ("I" not in neighbor_types)

    def causal_context0(fid, creator0, neigh0):
        """
        Causal weave: face lineage plus creators of spatial neighbors,
        all taken from the previous snapshot.
        """
        parents = {creator0[fid]}

        for g in neigh0[fid]:
            parents.add(creator0[g])

        return parents

    # Initial complete basin.
    root = make_event("ROOT", "root", -1, [], ep=-1)

    b0 = next_basin
    next_basin += 1

    t0 = make_event("T", "basin_driver", b0, [root], ep=0)
    s0 = make_event("S", "resolved_mode", b0, [t0], ep=0)
    i0 = make_event("I", "resolved_mode", b0, [t0], ep=0)
    g0 = make_event("G", "resolved_mode", b0, [t0], ep=0)

    # Open seed patch.
    # This intentionally contains one real unscreened S-G conflict.
    # The I face is present, but not edge-adjacent to the initial S-G pair.
    add_face((0, 1, 2), "S", s0, b0)
    add_face((0, 1, 3), "G", g0, b0)
    add_face((2, 4, 5), "I", i0, b0)
    add_face((3, 4, 5), "S", s0, b0)

    if max_epochs is None:
        max_epochs = max(10, 5 * target_basin_splits)

    epoch = 1

    while stats["basin_splits"] < target_basin_splits and epoch <= max_epochs:
        active0, types0, creator0, basin0, faces0, neigh0 = snapshot()

        frustrated = [
            fid for fid in active0
            if is_frustrated0(fid, types0, neigh0)
        ]

        stats["frontier_max"] = max(stats.get("frontier_max", 0), len(frustrated))

        remaining = target_basin_splits - stats["basin_splits"]

        if frustrated:
            rng.shuffle(frustrated)

            selected = frustrated[
                : min(len(frustrated), max_splits_per_epoch, remaining)
            ]

            split_records = []

            for fid in selected:
                b = next_basin
                next_basin += 1

                t = make_event(
                    "T",
                    "basin_driver",
                    b,
                    causal_context0(fid, creator0, neigh0),
                    ep=epoch,
                )

                s = make_event("S", "resolved_mode", b, [t], ep=epoch)
                i = make_event("I", "resolved_mode", b, [t], ep=epoch)
                g = make_event("G", "resolved_mode", b, [t], ep=epoch)

                split_records.append((fid, b, s, i, g, sorted(faces0[fid])))

            for fid, b, s, i, g, old_nodes in split_records:
                if fid not in active:
                    stats["selected_already_removed"] += 1
                    continue

                a_node, b_node, c_node = old_nodes
                new_node = next_node
                next_node += 1

                remove_face(fid)

                # One split creates the three resolved basin faces.
                add_face((new_node, a_node, b_node), "S", s, b)
                add_face((new_node, a_node, c_node), "I", i, b)
                add_face((new_node, b_node, c_node), "G", g, b)

                stats["basin_splits"] += 1

            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(
                stats.get("max_splits_in_epoch", 0),
                len(split_records),
            )

        else:
            # Parallel sterile tick epoch.
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
            selected = candidates[: min(len(candidates), max_ticks_per_epoch)]

            tick_records = []

            for fid in selected:
                t = make_event(
                    "T",
                    "sterile_tick",
                    -1,
                    causal_context0(fid, creator0, neigh0),
                    ep=epoch,
                )

                old_type = types0[fid]

                if old_type == "I":
                    new_type = "S"
                elif old_type == "G":
                    new_type = "I"
                else:
                    new_type = "G"

                tick_records.append((fid, t, old_type, new_type))

            for fid, t, old_type, new_type in tick_records:
                if fid not in active:
                    continue

                face_types[fid] = new_type
                face_creator[fid] = t
                face_basin[fid] = -1

                stats["sterile_ticks"] += 1
                stats[f"sterile_{old_type}_to_{new_type}"] += 1

            stats["tick_epochs"] += 1
            stats["max_ticks_in_epoch"] = max(
                stats.get("max_ticks_in_epoch", 0),
                len(tick_records),
            )

            if not tick_records:
                stats["sterile_starved"] += 1
                break

        epoch += 1

    stats["epochs"] = epoch - 1
    stats["final_active_faces"] = len(active)
    stats["final_nodes"] = next_node
    stats["final_events_including_root"] = next_event
    stats["final_basins_including_initial"] = next_basin
    stats["avg_parents_per_event"] = stats["parent_sum"] / max(1, next_event)

    elements = {e for e in causal_parents if e != root}

    for e in event_role:
        if e != root:
            elements.add(e)

    children = {e: set() for e in elements}

    for e, ps in causal_parents.items():
        if e == root:
            continue

        for p in ps:
            if p in elements:
                children.setdefault(p, set()).add(e)

    return FoamRun(
        elements=elements,
        children=children,
        event_role=event_role,
        event_kind=event_kind,
        event_basin=event_basin,
        event_epoch=event_epoch,
        stats=dict(stats),
    )



# --- extracted notebook cell 8 definitions ---

def interval_mode_stats(causet, bits):
    """
    Count roles and complete basins inside a bitset region.

    A complete basin means the same basin id contains:
        T, S, I, G
    """
    role_counts = Counter()
    kind_counts = Counter()
    basin_roles = defaultdict(set)

    for i in iter_bits(bits):
        e = causet.events[i]

        role = causet.event_role.get(e, "?")
        kind = causet.event_kind.get(e, "?")
        basin = causet.event_basin.get(e, None)

        role_counts[role] += 1
        kind_counts[kind] += 1

        if basin is not None and basin >= 0 and role in {"T", "S", "I", "G"}:
            basin_roles[basin].add(role)

    complete_basins = sum(
        1 for rs in basin_roles.values()
        if {"T", "S", "I", "G"}.issubset(rs)
    )

    resolved_triplets = sum(
        1 for rs in basin_roles.values()
        if {"S", "I", "G"}.issubset(rs)
    )

    return role_counts, kind_counts, complete_basins, resolved_triplets

def sample_mm_intervals(
    causet,
    n_int=200,
    min_size=60,
    max_size=None,
    seed=0,
    measure="strict",
    focus="inclusive",
    require_roles=None,
    min_complete_basins=0,
    min_resolved_triplets=0,
    max_tries_per_interval=3000,
):
    """
    Sample causal intervals and estimate MM dimension.

    measure:
        "strict"    -> measure only elements strictly between endpoints.
        "inclusive" -> include endpoints in the measured interval.

    focus:
        "inclusive" -> apply mode/basin filters to endpoints + interior.
        "strict"    -> apply mode/basin filters only to strict interior.

    Recommended for the main test:
        measure="strict"
        focus="inclusive"
        require_roles={"T", "S", "I", "G"}
        min_complete_basins=1

    This avoids endpoint bias in the MM estimator while still requiring the
    sampled causal diamond to be centered on the full basin structure.
    """
    rng = np.random.default_rng(seed)
    N = len(causet.events)

    require_roles = set(require_roles or [])

    dims = []
    widths = []
    order_fracs = []
    sizes = []
    complete_counts = []
    triplet_counts = []
    endpoint_pairs = []

    rejects = Counter()
    tries = 0
    max_tries = max(1, n_int * max_tries_per_interval)

    candidates_a = [
        i for i in range(N)
        if causet.descendant_bits[i].bit_count() >= max(1, min_size)
    ]

    if not candidates_a:
        return {
            "dims": np.array([]),
            "widths": np.array([]),
            "order_fracs": np.array([]),
            "sizes": np.array([]),
            "complete_basins": np.array([]),
            "resolved_triplets": np.array([]),
            "endpoint_pairs": [],
            "tries": 0,
            "rejects": Counter({"no_endpoint_with_large_future": 1}),
        }

    while len(dims) < n_int and tries < max_tries:
        tries += 1

        a = int(rng.choice(candidates_a))
        dbits = causet.descendant_bits[a]

        if dbits == 0:
            rejects["no_descendants"] += 1
            continue

        b = random_bit_index(dbits, rng)

        interior = causet.descendant_bits[a] & causet.ancestor_bits[b]
        inclusive = interior | bit_at(a) | bit_at(b)

        focus_bits = inclusive if focus == "inclusive" else interior
        measure_bits = interior if measure == "strict" else inclusive

        nn = measure_bits.bit_count()

        if nn < min_size:
            rejects["too_small"] += 1
            continue

        if max_size is not None and nn > max_size:
            rejects["too_large"] += 1
            continue

        complete = 0
        triplets = 0

        if require_roles or min_complete_basins or min_resolved_triplets:
            role_counts, kind_counts, complete, triplets = interval_mode_stats(
                causet,
                focus_bits,
            )

            missing = require_roles - set(role_counts)

            if missing:
                rejects["missing_required_roles"] += 1
                continue

            if complete < min_complete_basins:
                rejects["not_enough_complete_basins"] += 1
                continue

            if triplets < min_resolved_triplets:
                rejects["not_enough_resolved_triplets"] += 1
                continue

        total = nn * (nn - 1) // 2

        if total == 0:
            rejects["degenerate"] += 1
            continue

        rel = count_related_pairs(measure_bits, causet.descendant_bits)
        f = rel / total
        d = mm_dim(f)

        if not np.isfinite(d):
            rejects["dimension_out_of_range"] += 1
            continue

        dims.append(d)
        widths.append(1.0 - f)
        order_fracs.append(f)
        sizes.append(nn)
        complete_counts.append(complete)
        triplet_counts.append(triplets)
        endpoint_pairs.append((causet.events[a], causet.events[b]))

    return {
        "dims": np.array(dims),
        "widths": np.array(widths),
        "order_fracs": np.array(order_fracs),
        "sizes": np.array(sizes),
        "complete_basins": np.array(complete_counts),
        "resolved_triplets": np.array(triplet_counts),
        "endpoint_pairs": endpoint_pairs,
        "tries": tries,
        "rejects": rejects,
    }

def bootstrap_ci(x, fn=np.median, n_boot=1000, seed=0, alpha=0.05):
    x = np.asarray(x)

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        vals.append(fn(sample))

    lo = np.quantile(vals, alpha / 2)
    hi = np.quantile(vals, 1 - alpha / 2)

    return float(lo), float(hi)

def summarize_mm(label, sample, target_dim=4.0, seed=0):
    dims = sample["dims"]

    if len(dims) == 0:
        return {
            "label": label,
            "n_int": 0,
            "MM_med": np.nan,
            "MM_CI_lo": np.nan,
            "MM_CI_hi": np.nan,
            "MM_mean": np.nan,
            "CV": np.nan,
            "width_med": np.nan,
            "r_med": np.nan,
            "size_med": np.nan,
            "complete_basins_med": np.nan,
            "tries": sample["tries"],
            "rejects": dict(sample["rejects"]),
            "target_dim": target_dim,
            "target_in_CI": False,
        }

    lo, hi = bootstrap_ci(dims, np.median, seed=seed)

    return {
        "label": label,
        "n_int": len(dims),
        "MM_med": float(np.median(dims)),
        "MM_CI_lo": lo,
        "MM_CI_hi": hi,
        "MM_mean": float(np.mean(dims)),
        "CV": float(np.std(dims) / np.mean(dims)),
        "width_med": float(np.median(sample["widths"])),
        "r_med": float(np.median(sample["order_fracs"])),
        "size_med": float(np.median(sample["sizes"])),
        "size_min": int(np.min(sample["sizes"])),
        "size_max": int(np.max(sample["sizes"])),
        "complete_basins_med": (
            float(np.median(sample["complete_basins"]))
            if len(sample["complete_basins"]) else np.nan
        ),
        "tries": sample["tries"],
        "rejects": dict(sample["rejects"]),
        "target_dim": target_dim,
        "target_in_CI": bool(lo <= target_dim <= hi),
    }



# --- extracted notebook cell 9 definitions ---

def make_chain_causet(n=1000):
    elements = set(range(n))
    children = {i: {i + 1} for i in range(n - 1)}
    children[n - 1] = set()

    roles = {i: "T" for i in elements}
    kinds = {i: "chain_control" for i in elements}
    basins = {i: -1 for i in elements}
    epochs = {i: i for i in elements}

    return build_bitset_causet(
        elements,
        children,
        roles,
        kinds,
        basins,
        epochs,
    )

def sprinkle_minkowski_diamond(n=1200, d=4, seed=0, T=2.0):
    """
    Uniformly sprinkle n points into a d-dimensional Minkowski Alexandrov diamond.

    The causal relation is:
        x < y iff dt >= spatial_distance
    """
    rng = np.random.default_rng(seed)

    R = T / 2.0
    spatial_dim = d - 1

    times = []
    positions = []

    for _ in range(n):
        side = -1 if rng.random() < 0.5 else 1

        u = rng.random()
        s = 1.0 - (1.0 - u) ** (1.0 / d)
        t = side * R * s

        cross_section_radius = R - abs(t)

        if spatial_dim > 0:
            direction = rng.normal(size=spatial_dim)
            norm = np.linalg.norm(direction)

            if norm == 0:
                direction[0] = 1.0
                norm = 1.0

            direction = direction / norm
            radius = cross_section_radius * (rng.random() ** (1.0 / spatial_dim))
            x = direction * radius
        else:
            x = np.empty(0)

        times.append(t)
        positions.append(x)

    times = np.asarray(times)
    positions = np.asarray(positions)

    order = np.argsort(times)

    elements = set(range(n))
    children = {i: set() for i in range(n)}

    for ai in range(n):
        i = order[ai]

        future_candidates = order[ai + 1:]
        dt = times[future_candidates] - times[i]

        if spatial_dim > 0:
            dx = positions[future_candidates] - positions[i]
            ds = np.linalg.norm(dx, axis=1)
        else:
            ds = np.zeros_like(dt)

        related = future_candidates[dt >= ds]
        children[i].update(int(j) for j in related)

    roles = {i: "sprinkle" for i in elements}
    kinds = {i: f"Minkowski_{d}D_control" for i in elements}
    basins = {i: -1 for i in elements}
    epochs = {i: 0 for i in elements}

    return build_bitset_causet(
        elements,
        children,
        roles,
        kinds,
        basins,
        epochs,
    )

def run_controls(seed=123, n_int=200, min_size=40):
    chain = make_chain_causet(n=1000)

    chain_sample = sample_mm_intervals(
        chain,
        n_int=n_int,
        min_size=min_size,
        seed=seed,
        measure="strict",
    )

    m4 = sprinkle_minkowski_diamond(
        n=1200,
        d=4,
        seed=seed,
    )

    m4_sample = sample_mm_intervals(
        m4,
        n_int=n_int,
        min_size=min_size,
        seed=seed + 1,
        measure="strict",
        max_tries_per_interval=8000,
    )

    rows = [
        summarize_mm("negative_control_chain", chain_sample, target_dim=1.0, seed=seed),
        summarize_mm("positive_control_4D_sprinkle", m4_sample, target_dim=4.0, seed=seed + 1),
    ]

    return pd.DataFrame(rows)



# --- extracted notebook cell 23 definitions ---

def topo_sort_events(elements, children, event_epoch=None):
    """
    Topologically sort arbitrary event IDs.

    This is safer than assuming event IDs are already topological.
    """
    elements = set(elements)
    indeg = {e: 0 for e in elements}
    parents = {e: set() for e in elements}

    for p, cs in children.items():
        if p not in elements:
            continue
        for c in cs:
            if c in elements and c != p:
                indeg[c] += 1
                parents[c].add(p)

    def sort_key(e):
        ep = event_epoch.get(e, 0) if event_epoch else 0
        return (ep, e)

    q = deque(sorted([e for e in elements if indeg[e] == 0], key=sort_key))
    out = []

    while q:
        e = q.popleft()
        out.append(e)

        for c in sorted(children.get(e, ()), key=sort_key):
            if c not in elements:
                continue
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    if len(out) != len(elements):
        raise ValueError(
            f"Cycle detected or invalid quotient: sorted {len(out)} of {len(elements)} events."
        )

    return out

def build_bitset_causet_robust(
    elements,
    children,
    event_role=None,
    event_kind=None,
    event_basin=None,
    event_epoch=None,
):
    """
    Same purpose as build_bitset_causet, but robust to event IDs not being
    topologically sorted.
    """
    elements = set(elements)
    event_role = event_role or {}
    event_kind = event_kind or {}
    event_basin = event_basin or {}
    event_epoch = event_epoch or {}

    events = topo_sort_events(elements, children, event_epoch=event_epoch)
    index = {e: i for i, e in enumerate(events)}
    n = len(events)

    child_bits = [0] * n
    parent_bits = [0] * n

    for p, cs in children.items():
        if p not in index:
            continue

        pi = index[p]

        for c in cs:
            if c in index and c != p:
                ci = index[c]
                child_bits[pi] |= bit_at(ci)
                parent_bits[ci] |= bit_at(pi)

    ancestor_bits = [0] * n

    for i in range(n):
        bits = parent_bits[i]

        for p in iter_bits(parent_bits[i]):
            bits |= ancestor_bits[p]

        ancestor_bits[i] = bits

    descendant_bits = [0] * n

    for i in range(n - 1, -1, -1):
        bits = child_bits[i]

        for c in iter_bits(child_bits[i]):
            bits |= descendant_bits[c]

        descendant_bits[i] = bits

    return BitsetCauset(
        events=events,
        index=index,
        children={e: set(children.get(e, set())) & elements for e in elements},
        child_bits=child_bits,
        parent_bits=parent_bits,
        ancestor_bits=ancestor_bits,
        descendant_bits=descendant_bits,
        event_role=event_role,
        event_kind=event_kind,
        event_basin=event_basin,
        event_epoch=event_epoch,
    )

def finite_sample_mm_intervals(
    causet,
    label="sample",
    endpoint_samples=50000,
    max_intervals=250,
    min_size=20,
    max_size=None,
    seed=0,
    measure="strict",
    target_dim=4.0,
):
    """
    Finite non-blocking MM sampler.

    It samples a fixed number of comparable endpoint pairs and keeps the
    intervals that satisfy the size cut. It never tries forever to fill
    impossible bins.
    """
    rng = np.random.default_rng(seed)
    N = len(causet.events)

    candidates_a = [
        i for i in range(N)
        if causet.descendant_bits[i] != 0
    ]

    dims = []
    widths = []
    order_fracs = []
    sizes = []
    endpoints = []
    rejects = Counter()

    if not candidates_a:
        sample = {
            "dims": np.array([]),
            "widths": np.array([]),
            "order_fracs": np.array([]),
            "sizes": np.array([]),
            "complete_basins": np.array([]),
            "resolved_triplets": np.array([]),
            "endpoint_pairs": [],
            "tries": 0,
            "rejects": Counter({"no_future_events": 1}),
        }
        row = summarize_mm(label, sample, target_dim=target_dim, seed=seed)
        return row, sample

    for t in range(1, endpoint_samples + 1):
        if len(dims) >= max_intervals:
            break

        a = int(rng.choice(candidates_a))
        dbits = causet.descendant_bits[a]

        if dbits == 0:
            rejects["no_descendants"] += 1
            continue

        b = random_bit_index(dbits, rng)

        interior = causet.descendant_bits[a] & causet.ancestor_bits[b]
        inclusive = interior | bit_at(a) | bit_at(b)

        bits = interior if measure == "strict" else inclusive
        nn = bits.bit_count()

        if nn < min_size:
            rejects["too_small"] += 1
            continue

        if max_size is not None and nn > max_size:
            rejects["too_large"] += 1
            continue

        total = nn * (nn - 1) // 2

        if total == 0:
            rejects["degenerate"] += 1
            continue

        rel = count_related_pairs(bits, causet.descendant_bits)
        r = rel / total
        d = mm_dim(r)

        if not np.isfinite(d):
            rejects["dimension_out_of_range"] += 1
            continue

        dims.append(d)
        widths.append(1.0 - r)
        order_fracs.append(r)
        sizes.append(nn)
        endpoints.append((causet.events[a], causet.events[b]))

    sample = {
        "dims": np.asarray(dims),
        "widths": np.asarray(widths),
        "order_fracs": np.asarray(order_fracs),
        "sizes": np.asarray(sizes),
        "complete_basins": np.zeros(len(dims)),
        "resolved_triplets": np.zeros(len(dims)),
        "endpoint_pairs": endpoints,
        "tries": t if endpoint_samples else 0,
        "rejects": rejects,
    }

    row = summarize_mm(label, sample, target_dim=target_dim, seed=seed)
    row["endpoint_samples_cap"] = endpoint_samples
    row["max_intervals_cap"] = max_intervals
    row["min_size_requested"] = min_size
    row["max_size_requested"] = max_size

    return row, sample

def adaptive_min_size(causet, floor=8, ceiling=60, divisor=120):
    """
    Conservative adaptive interval-size cut for macro-causets.
    """
    n = len(causet.events)
    return int(max(floor, min(ceiling, n // divisor)))



# --- extracted notebook cell 24 definitions ---

def strongly_connected_components(nodes, children):
    """
    Tarjan SCC. Used to repair quotient cycles by merging cyclic macro-blocks.
    A quotient of a DAG by arbitrary blocks can create cycles; merging SCCs
    is the clean causal-set-safe repair.
    """
    nodes = list(nodes)
    node_set = set(nodes)

    index_counter = [0]
    index = {}
    lowlink = {}
    stack = []
    on_stack = set()
    comps = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in children.get(v, ()):
            if w not in node_set:
                continue

            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            comp = []

            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)

                if w == v:
                    break

            comps.append(comp)

    for v in nodes:
        if v not in index:
            strongconnect(v)

    return comps

def condense_children_if_needed(elements, children):
    """
    If quotienting created cycles, merge strongly connected macro-blocks.
    Returns:
        new_elements, new_children, old_to_new, stats
    """
    elements = set(elements)
    children = {e: set(children.get(e, set())) & elements for e in elements}

    comps = strongly_connected_components(elements, children)
    cyclic_comps = [c for c in comps if len(c) > 1]

    if not cyclic_comps:
        return elements, children, {e: e for e in elements}, {
            "scc_merged": 0,
            "scc_components": len(comps),
            "scc_largest": 1,
        }

    comps_sorted = sorted(comps, key=lambda c: min(c))
    old_to_new = {}

    for new_id, comp in enumerate(comps_sorted):
        for old in comp:
            old_to_new[old] = new_id

    new_elements = set(range(len(comps_sorted)))
    new_children = {e: set() for e in new_elements}

    for p, cs in children.items():
        np_ = old_to_new[p]

        for c in cs:
            nc = old_to_new[c]

            if np_ != nc:
                new_children[np_].add(nc)

    return new_elements, new_children, old_to_new, {
        "scc_merged": sum(len(c) - 1 for c in comps),
        "scc_components": len(comps),
        "scc_largest": max(len(c) for c in comps),
    }

def event_block_summary(causet, event_indices):
    roles = Counter()
    kinds = Counter()
    basins = set()
    epochs = []

    for i in event_indices:
        e = causet.events[i]
        roles[causet.event_role.get(e, "?")] += 1
        kinds[causet.event_kind.get(e, "?")] += 1

        b = causet.event_basin.get(e, None)
        if b is not None:
            basins.add(b)

        ep = causet.event_epoch.get(e, None)
        if ep is not None:
            epochs.append(ep)

    if epochs:
        ep_mean = float(np.mean(epochs))
        ep_min = float(np.min(epochs))
        ep_max = float(np.max(epochs))
    else:
        ep_mean = ep_min = ep_max = 0.0

    role_label = "+".join(f"{k}:{v}" for k, v in sorted(roles.items()))
    kind_label = "+".join(f"{k}:{v}" for k, v in sorted(kinds.items()))

    return {
        "role_label": role_label,
        "kind_label": kind_label,
        "n_micro": len(event_indices),
        "n_basins": len(basins),
        "epoch_mean": ep_mean,
        "epoch_min": ep_min,
        "epoch_max": ep_max,
    }

def build_quotient_causet_from_blocks(
    causet,
    block_key_by_index,
    relation_mode="direct",
    relation_threshold=0.0,
    drop_unassigned=True,
):
    """
    Build a macro-causet by mapping micro-events to block keys.

    relation_mode:
        "direct"   : macro edge A->B exists if a direct micro edge crosses A,B.
        "any"      : macro edge A->B exists if any micro event in A precedes
                     any micro event in B.
        "fraction" : macro edge A->B exists if the fraction of related micro-pairs
                     A->B is >= relation_threshold.

    For late-DEU testing, start with relation_mode="direct". Then audit "any"
    and "fraction" to see how sensitive the result is.
    """
    N = len(causet.events)

    if len(block_key_by_index) != N:
        raise ValueError("block_key_by_index must have length len(causet.events).")

    raw_blocks = defaultdict(list)

    for i, key in enumerate(block_key_by_index):
        if key is None:
            if not drop_unassigned:
                raw_blocks[("singleton", i)].append(i)
            continue
        raw_blocks[key].append(i)

    # Stable macro-id ordering by mean epoch then first original index.
    def raw_block_sort_key(item):
        key, inds = item
        epochs = [
            causet.event_epoch.get(causet.events[i], 0.0)
            for i in inds
        ]
        return (float(np.mean(epochs)), min(inds), str(key))

    block_items = sorted(raw_blocks.items(), key=raw_block_sort_key)
    raw_key_to_macro = {key: k for k, (key, inds) in enumerate(block_items)}
    macro_to_indices = {
        raw_key_to_macro[key]: inds
        for key, inds in block_items
    }

    event_to_macro = np.full(N, -1, dtype=int)

    for key, inds in block_items:
        m = raw_key_to_macro[key]
        for i in inds:
            event_to_macro[i] = m

    M = len(block_items)
    macro_children = {m: set() for m in range(M)}

    if relation_mode == "direct":
        for i in range(N):
            mi = event_to_macro[i]
            if mi < 0:
                continue

            for j in iter_bits(causet.child_bits[i]):
                mj = event_to_macro[j]
                if mj >= 0 and mj != mi:
                    macro_children[mi].add(mj)

    elif relation_mode in {"any", "fraction"}:
        block_bits = [0] * M

        for m, inds in macro_to_indices.items():
            bits = 0
            for i in inds:
                bits |= bit_at(i)
            block_bits[m] = bits

        block_sizes = [len(macro_to_indices[m]) for m in range(M)]

        for ma in range(M):
            inds_a = macro_to_indices[ma]
            denom_base = block_sizes[ma]

            desc_counts_by_block = np.zeros(M, dtype=int)

            for i in inds_a:
                desc = causet.descendant_bits[i]

                for mb in range(M):
                    if mb == ma:
                        continue
                    desc_counts_by_block[mb] += (desc & block_bits[mb]).bit_count()

            for mb in range(M):
                if mb == ma:
                    continue

                related_count = int(desc_counts_by_block[mb])

                if relation_mode == "any":
                    if related_count > 0:
                        macro_children[ma].add(mb)

                else:
                    denom = denom_base * block_sizes[mb]
                    frac = related_count / denom if denom else 0.0

                    if frac >= relation_threshold and related_count > 0:
                        macro_children[ma].add(mb)

    else:
        raise ValueError("relation_mode must be 'direct', 'any', or 'fraction'.")

    # Repair quotient cycles if the block partition created them.
    macro_elements = set(range(M))
    condensed_elements, condensed_children, old_to_new, scc_stats = condense_children_if_needed(
        macro_elements,
        macro_children,
    )

    # Combine metadata after SCC condensation.
    condensed_to_old_macros = defaultdict(list)

    for old, new in old_to_new.items():
        condensed_to_old_macros[new].append(old)

    event_role = {}
    event_kind = {}
    event_basin = {}
    event_epoch = {}

    for new_m in condensed_elements:
        old_macros = condensed_to_old_macros[new_m]
        inds = []

        for old_m in old_macros:
            inds.extend(macro_to_indices[old_m])

        summary = event_block_summary(causet, inds)

        event_role[new_m] = "MACRO"
        event_kind[new_m] = (
            f"block[n={summary['n_micro']},basins={summary['n_basins']}]"
        )
        event_basin[new_m] = -1
        event_epoch[new_m] = summary["epoch_mean"]

    macro_causet = build_bitset_causet_robust(
        condensed_elements,
        condensed_children,
        event_role=event_role,
        event_kind=event_kind,
        event_basin=event_basin,
        event_epoch=event_epoch,
    )

    stats = {
        "micro_events": N,
        "raw_blocks": M,
        "macro_events": len(macro_causet.events),
        "relation_mode": relation_mode,
        "relation_threshold": relation_threshold,
        **scc_stats,
    }

    return macro_causet, stats



# --- extracted notebook cell 25 definitions ---

def deu_basin_block_keys(
    causet,
    sterile_mode="singleton",
    include_root=False,
):
    """
    Natural DEU first coarse-graining:
        each complete T/S/I/G basin -> one macro-event.

    sterile_mode:
        "singleton" : keep sterile_tick events as their own macro-events.
        "epoch"     : group sterile ticks by epoch.
        "drop"      : drop sterile ticks from the quotient.

    Recommended first run:
        sterile_mode="singleton"
    """
    keys = []

    for i, e in enumerate(causet.events):
        role = causet.event_role.get(e, "?")
        kind = causet.event_kind.get(e, "?")
        basin = causet.event_basin.get(e, -1)
        epoch = causet.event_epoch.get(e, 0)

        if role == "ROOT" and not include_root:
            keys.append(None)
            continue

        if basin is not None and basin >= 0 and role in {"T", "S", "I", "G"}:
            keys.append(("basin", int(basin)))
            continue

        if kind == "sterile_tick" or role == "T":
            if sterile_mode == "singleton":
                keys.append(("sterile", int(e)))
            elif sterile_mode == "epoch":
                keys.append(("sterile_epoch", int(epoch)))
            elif sterile_mode == "drop":
                keys.append(None)
            else:
                raise ValueError("Unknown sterile_mode.")
            continue

        # Fallback: keep unusual events rather than silently discarding them.
        keys.append(("other", int(e)))

    return keys

def make_deu_basin_macro_causet(
    causet,
    sterile_mode="singleton",
    relation_mode="direct",
    relation_threshold=0.0,
):
    keys = deu_basin_block_keys(
        causet,
        sterile_mode=sterile_mode,
    )

    macro, stats = build_quotient_causet_from_blocks(
        causet,
        keys,
        relation_mode=relation_mode,
        relation_threshold=relation_threshold,
        drop_unassigned=True,
    )

    stats["coarse_grain_step"] = "deu_basin_compression"
    stats["sterile_mode"] = sterile_mode

    return macro, stats



# --- extracted notebook cell 26 definitions ---

def sample_bits_tuple(bits, k=6):
    """
    Deterministic compact signature of a bitset.
    Uses approximately evenly spaced set bits rather than just the first k.
    """
    inds = list(iter_bits(bits))
    n = len(inds)

    if n <= k:
        return tuple(inds)

    positions = np.linspace(0, n - 1, k).astype(int)
    return tuple(inds[p] for p in positions)

def causal_signature_block_keys(
    causet,
    epoch_width=2,
    max_block_size=8,
    signature_k=6,
    include_role=False,
):
    """
    Build block keys by grouping events in the same epoch slab with similar
    causal parent/child signatures.

    This is intentionally neutral:
        - it does not inject a target dimension,
        - it does not use coordinates,
        - it only uses the existing causal graph plus epoch labels.
    """
    N = len(causet.events)

    event_epochs = np.array([
        causet.event_epoch.get(e, i)
        for i, e in enumerate(causet.events)
    ], dtype=float)

    ep_min = float(np.min(event_epochs))
    slab_ids = np.floor((event_epochs - ep_min) / epoch_width).astype(int)

    by_slab = defaultdict(list)

    for i, slab in enumerate(slab_ids):
        by_slab[int(slab)].append(i)

    block_keys = [None] * N
    block_counter = 0

    for slab in sorted(by_slab):
        inds = by_slab[slab]

        records = []

        for i in inds:
            pbits = causet.parent_bits[i]
            cbits = causet.child_bits[i]

            role_key = causet.event_role.get(causet.events[i], "?") if include_role else "*"

            sig = (
                int(slab),
                role_key,
                pbits.bit_count(),
                cbits.bit_count(),
                sample_bits_tuple(pbits, k=signature_k),
                sample_bits_tuple(cbits, k=signature_k),
            )

            records.append((sig, i))

        records.sort(key=lambda x: (str(x[0]), x[1]))

        # Chunk consecutive similar-signature records.
        for start in range(0, len(records), max_block_size):
            chunk = records[start:start + max_block_size]
            key = ("rg", block_counter)

            for sig, i in chunk:
                block_keys[i] = key

            block_counter += 1

    return block_keys

def rg_coarse_grain_once(
    causet,
    epoch_width=2,
    max_block_size=8,
    signature_k=6,
    relation_mode="direct",
    relation_threshold=0.0,
    include_role=False,
):
    keys = causal_signature_block_keys(
        causet,
        epoch_width=epoch_width,
        max_block_size=max_block_size,
        signature_k=signature_k,
        include_role=include_role,
    )

    macro, stats = build_quotient_causet_from_blocks(
        causet,
        keys,
        relation_mode=relation_mode,
        relation_threshold=relation_threshold,
        drop_unassigned=True,
    )

    stats["coarse_grain_step"] = "causal_signature_rg"
    stats["epoch_width"] = epoch_width
    stats["max_block_size"] = max_block_size
    stats["signature_k"] = signature_k
    stats["include_role"] = include_role

    return macro, stats



# --- extracted notebook cell 36 definitions ---

def causal_signature_block_keys_safe(
    causet,
    epoch_width=1,
    max_block_size=4,
    signature_k=6,
    signature_mode="degree",
    include_role=False,
):
    """
    Safer version of causal_signature_block_keys.

    Difference from previous version:
        It only chunks events WITHIN the same exact signature group.
        It never groups merely adjacent sorted signatures.

    signature_mode:
        "degree"  -> groups by epoch slab, parent count, child count.
        "sampled" -> also includes sampled parent/child identity signatures.
                     This is stricter and may compress less.
    """
    N = len(causet.events)

    event_epochs = np.array([
        causet.event_epoch.get(e, i)
        for i, e in enumerate(causet.events)
    ], dtype=float)

    ep_min = float(np.min(event_epochs))
    slab_ids = np.floor((event_epochs - ep_min) / epoch_width).astype(int)

    by_slab = defaultdict(list)

    for i, slab in enumerate(slab_ids):
        by_slab[int(slab)].append(i)

    block_keys = [None] * N
    block_counter = 0

    for slab in sorted(by_slab):
        groups = defaultdict(list)

        for i in by_slab[slab]:
            pbits = causet.parent_bits[i]
            cbits = causet.child_bits[i]

            role_key = causet.event_role.get(causet.events[i], "?") if include_role else "*"

            if signature_mode == "degree":
                sig = (
                    int(slab),
                    role_key,
                    pbits.bit_count(),
                    cbits.bit_count(),
                )
            elif signature_mode == "sampled":
                sig = (
                    int(slab),
                    role_key,
                    pbits.bit_count(),
                    cbits.bit_count(),
                    sample_bits_tuple(pbits, k=signature_k),
                    sample_bits_tuple(cbits, k=signature_k),
                )
            else:
                raise ValueError("signature_mode must be 'degree' or 'sampled'.")

            groups[sig].append(i)

        for sig in sorted(groups.keys(), key=str):
            inds = sorted(groups[sig])

            for start in range(0, len(inds), max_block_size):
                chunk = inds[start:start + max_block_size]
                key = ("safe_rg", block_counter)

                for i in chunk:
                    block_keys[i] = key

                block_counter += 1

    return block_keys

def rg_coarse_grain_once_safe(
    causet,
    epoch_width=1,
    max_block_size=4,
    signature_k=6,
    signature_mode="degree",
    relation_mode="direct",
    relation_threshold=0.0,
    include_role=False,
):
    keys = causal_signature_block_keys_safe(
        causet,
        epoch_width=epoch_width,
        max_block_size=max_block_size,
        signature_k=signature_k,
        signature_mode=signature_mode,
        include_role=include_role,
    )

    macro, stats = build_quotient_causet_from_blocks(
        causet,
        keys,
        relation_mode=relation_mode,
        relation_threshold=relation_threshold,
        drop_unassigned=True,
    )

    stats["coarse_grain_step"] = "safe_exact_signature_rg"
    stats["epoch_width"] = epoch_width
    stats["max_block_size"] = max_block_size
    stats["signature_k"] = signature_k
    stats["signature_mode"] = signature_mode
    stats["include_role"] = include_role

    return macro, stats



# --- extracted notebook cell 38 definitions ---

def extract_source_fields(source):
    """
    Accepts either:
        - BitsetCauset object, e.g. causet
        - FoamRun object, e.g. run

    Returns:
        elements, children, event_role, event_kind, event_basin, event_epoch
    """
    if hasattr(source, "events") and hasattr(source, "child_bits"):
        elements = set(source.events)
        children = {
            e: set(source.children.get(e, set())) & elements
            for e in elements
        }
        event_role = source.event_role
        event_kind = source.event_kind
        event_basin = source.event_basin
        event_epoch = source.event_epoch
        return elements, children, event_role, event_kind, event_basin, event_epoch

    if hasattr(source, "elements") and hasattr(source, "children"):
        elements = set(source.elements)
        children = {
            e: set(source.children.get(e, set())) & elements
            for e in elements
        }
        event_role = source.event_role
        event_kind = source.event_kind
        event_basin = source.event_basin
        event_epoch = source.event_epoch
        return elements, children, event_role, event_kind, event_basin, event_epoch

    raise TypeError("source must be a BitsetCauset or FoamRun-like object.")

def source_epoch_range(source):
    elements, children, event_role, event_kind, event_basin, event_epoch = extract_source_fields(source)

    eps = [
        event_epoch.get(e, None)
        for e in elements
        if event_epoch.get(e, None) is not None
    ]

    return float(np.min(eps)), float(np.max(eps))

def build_epoch_window_causet(
    source,
    ep_lo,
    ep_hi,
    max_events=12000,
    seed=0,
):
    """
    Build an induced sub-causet from events with ep_lo <= epoch <= ep_hi.

    This avoids building a huge full transitive closure for very large runs.
    If the selected window is too large, it samples an induced subset.
    """
    rng = np.random.default_rng(seed)

    elements, children, event_role, event_kind, event_basin, event_epoch = extract_source_fields(source)

    selected = [
        e for e in elements
        if ep_lo <= event_epoch.get(e, -np.inf) <= ep_hi
    ]

    if len(selected) == 0:
        return None, {
            "selected_events_before_cap": 0,
            "selected_events_after_cap": 0,
            "ep_lo": ep_lo,
            "ep_hi": ep_hi,
            "capped": False,
        }

    capped = False

    if max_events is not None and len(selected) > max_events:
        capped = True
        selected = list(rng.choice(selected, size=max_events, replace=False))

    selected = set(selected)

    sub_children = {
        e: set(children.get(e, set())) & selected
        for e in selected
    }

    sub_role = {e: event_role.get(e, "?") for e in selected}
    sub_kind = {e: event_kind.get(e, "?") for e in selected}
    sub_basin = {e: event_basin.get(e, -1) for e in selected}
    sub_epoch = {e: event_epoch.get(e, 0) for e in selected}

    sub_causet = build_bitset_causet_robust(
        selected,
        sub_children,
        event_role=sub_role,
        event_kind=sub_kind,
        event_basin=sub_basin,
        event_epoch=sub_epoch,
    )

    return sub_causet, {
        "selected_events_before_cap": len([
            e for e in elements
            if ep_lo <= event_epoch.get(e, -np.inf) <= ep_hi
        ]),
        "selected_events_after_cap": len(selected),
        "ep_lo": ep_lo,
        "ep_hi": ep_hi,
        "capped": capped,
    }

def make_epoch_windows(
    source,
    n_windows=8,
    width=None,
    mode="tail",
):
    """
    mode:
        "tail"   -> fixed-width windows sliding from early to late.
        "prefix" -> cumulative windows from start to later cutoffs.

    For late-time testing, use mode="tail".
    """
    ep_min, ep_max = source_epoch_range(source)

    ep_min_i = int(np.floor(ep_min))
    ep_max_i = int(np.ceil(ep_max))

    total_width = max(1, ep_max_i - ep_min_i + 1)

    if width is None:
        width = max(4, int(np.ceil(total_width / 4)))

    windows = []

    if mode == "tail":
        if n_windows == 1:
            ends = [ep_max_i]
        else:
            ends = np.linspace(ep_min_i + width - 1, ep_max_i, n_windows)
            ends = sorted(set(int(round(x)) for x in ends))

        for end in ends:
            lo = max(ep_min_i, end - width + 1)
            hi = min(ep_max_i, end)
            windows.append((lo, hi))

    elif mode == "prefix":
        cutoffs = np.linspace(ep_min_i + width - 1, ep_max_i, n_windows)
        cutoffs = sorted(set(int(round(x)) for x in cutoffs))

        for hi in cutoffs:
            windows.append((ep_min_i, hi))

    else:
        raise ValueError("mode must be 'tail' or 'prefix'.")

    return windows



# --- extracted notebook cell 48 definitions ---

def bits_for_epoch_band(causet, ep_lo=None, ep_hi=None):
    """
    Return a bitset of events whose epoch lies in [ep_lo, ep_hi].
    If either bound is None, it is left open.
    """
    bits = 0

    for i, e in enumerate(causet.events):
        ep = causet.event_epoch.get(e, 0)

        if ep_lo is not None and ep < ep_lo:
            continue

        if ep_hi is not None and ep > ep_hi:
            continue

        bits |= bit_at(i)

    return bits

def causet_epoch_minmax(causet):
    eps = [
        causet.event_epoch.get(e, 0)
        for e in causet.events
    ]

    return float(np.min(eps)), float(np.max(eps))

def sample_mm_intervals_epoch_separated(
    causet,
    label="epoch_separated",
    n_int=200,
    endpoint_samples=50000,
    early_frac=0.25,
    late_frac=0.25,
    min_epoch_gap=1,
    min_size=8,
    max_size=None,
    seed=0,
    target_dim=4.0,
):
    """
    Sample intervals whose endpoints are separated across the window.

    Instead of:
        random a, random descendant b

    this does:
        a from early part of epoch window
        b from late part of epoch window
        require a < b

    This is a better large-diamond / late-time diagnostic.
    """
    rng = np.random.default_rng(seed)

    if len(causet.events) < 2:
        sample = {
            "dims": np.array([]),
            "widths": np.array([]),
            "order_fracs": np.array([]),
            "sizes": np.array([]),
            "complete_basins": np.array([]),
            "resolved_triplets": np.array([]),
            "endpoint_pairs": [],
            "tries": 0,
            "rejects": Counter({"too_few_events": 1}),
        }
        return summarize_mm(label, sample, target_dim=target_dim, seed=seed), sample

    ep_min, ep_max = causet_epoch_minmax(causet)
    ep_span = max(1.0, ep_max - ep_min + 1.0)

    early_width = max(1.0, np.ceil(ep_span * early_frac))
    late_width = max(1.0, np.ceil(ep_span * late_frac))

    early_hi = ep_min + early_width - 1
    late_lo = ep_max - late_width + 1

    early_bits = bits_for_epoch_band(causet, ep_lo=ep_min, ep_hi=early_hi)
    late_bits = bits_for_epoch_band(causet, ep_lo=late_lo, ep_hi=ep_max)

    candidates_a = [
        i for i in iter_bits(early_bits)
        if (causet.descendant_bits[i] & late_bits) != 0
    ]

    dims = []
    widths = []
    order_fracs = []
    sizes = []
    endpoint_pairs = []
    rejects = Counter()

    if not candidates_a:
        sample = {
            "dims": np.array([]),
            "widths": np.array([]),
            "order_fracs": np.array([]),
            "sizes": np.array([]),
            "complete_basins": np.array([]),
            "resolved_triplets": np.array([]),
            "endpoint_pairs": [],
            "tries": 0,
            "rejects": Counter({
                "no_early_to_late_comparable_pairs": 1,
                "early_bits": early_bits.bit_count(),
                "late_bits": late_bits.bit_count(),
            }),
        }
        row = summarize_mm(label, sample, target_dim=target_dim, seed=seed)
        row["early_count"] = early_bits.bit_count()
        row["late_count"] = late_bits.bit_count()
        row["early_hi"] = early_hi
        row["late_lo"] = late_lo
        return row, sample

    tries = 0

    while len(dims) < n_int and tries < endpoint_samples:
        tries += 1

        a = int(rng.choice(candidates_a))
        possible_b = causet.descendant_bits[a] & late_bits

        if possible_b == 0:
            rejects["no_late_descendant"] += 1
            continue

        b = random_bit_index(possible_b, rng)

        ea = causet.events[a]
        eb = causet.events[b]
        ep_a = causet.event_epoch.get(ea, 0)
        ep_b = causet.event_epoch.get(eb, 0)

        if ep_b - ep_a < min_epoch_gap:
            rejects["epoch_gap_too_small"] += 1
            continue

        interval_bits = causet.descendant_bits[a] & causet.ancestor_bits[b]
        nn = interval_bits.bit_count()

        if nn < min_size:
            rejects["too_small"] += 1
            continue

        if max_size is not None and nn > max_size:
            rejects["too_large"] += 1
            continue

        total = nn * (nn - 1) // 2

        if total <= 0:
            rejects["degenerate"] += 1
            continue

        rel = count_related_pairs(interval_bits, causet.descendant_bits)
        r = rel / total
        d = mm_dim(r)

        if not np.isfinite(d):
            rejects["dimension_out_of_range"] += 1
            continue

        dims.append(d)
        widths.append(1.0 - r)
        order_fracs.append(r)
        sizes.append(nn)
        endpoint_pairs.append((ea, eb))

    sample = {
        "dims": np.asarray(dims),
        "widths": np.asarray(widths),
        "order_fracs": np.asarray(order_fracs),
        "sizes": np.asarray(sizes),
        "complete_basins": np.zeros(len(dims)),
        "resolved_triplets": np.zeros(len(dims)),
        "endpoint_pairs": endpoint_pairs,
        "tries": tries,
        "rejects": rejects,
    }

    row = summarize_mm(label, sample, target_dim=target_dim, seed=seed)
    row["early_count"] = early_bits.bit_count()
    row["late_count"] = late_bits.bit_count()
    row["early_hi"] = early_hi
    row["late_lo"] = late_lo
    row["min_epoch_gap"] = min_epoch_gap
    row["min_size_used"] = min_size

    return row, sample



# --- extracted notebook cell 66 definitions ---

def largest_epoch_separated_diamonds(
    causet,
    n_endpoint_samples=100000,
    keep_top=200,
    early_frac=0.25,
    late_frac=0.25,
    min_epoch_gap=1,
    min_size=2,
    seed=0,
):
    """
    Search for the largest strict intervals between early-window and late-window endpoints.

    Patched version:
        Does NOT let pandas infer interval_bits as numeric.
        Stores interval_bits as dtype=object to avoid int-too-large overflow.
    """
    rng = np.random.default_rng(seed)

    if len(causet.events) < 2:
        return pd.DataFrame()

    ep_min, ep_max = causet_epoch_minmax(causet)
    ep_span = max(1.0, ep_max - ep_min + 1.0)

    early_width = max(1.0, np.ceil(ep_span * early_frac))
    late_width = max(1.0, np.ceil(ep_span * late_frac))

    early_hi = ep_min + early_width - 1
    late_lo = ep_max - late_width + 1

    early_bits = bits_for_epoch_band(causet, ep_lo=ep_min, ep_hi=early_hi)
    late_bits = bits_for_epoch_band(causet, ep_lo=late_lo, ep_hi=ep_max)

    candidates_a = [
        i for i in iter_bits(early_bits)
        if (causet.descendant_bits[i] & late_bits) != 0
    ]

    if not candidates_a:
        return pd.DataFrame({
            "reason": ["no_early_to_late_comparable_pairs"],
            "early_count": [early_bits.bit_count()],
            "late_count": [late_bits.bit_count()],
        })

    meta_records = []
    interval_bits_store = []

    for _ in range(n_endpoint_samples):
        a = int(rng.choice(candidates_a))
        possible_b = causet.descendant_bits[a] & late_bits

        if possible_b == 0:
            continue

        b = random_bit_index(possible_b, rng)

        ea = causet.events[a]
        eb = causet.events[b]
        ep_a = causet.event_epoch.get(ea, 0)
        ep_b = causet.event_epoch.get(eb, 0)

        if ep_b - ep_a < min_epoch_gap:
            continue

        bits = causet.descendant_bits[a] & causet.ancestor_bits[b]
        nn = bits.bit_count()

        if nn < min_size:
            continue

        bit_id = len(interval_bits_store)
        interval_bits_store.append(bits)

        meta_records.append({
            "_bit_id": bit_id,
            "a_index": a,
            "b_index": b,
            "a_event": ea,
            "b_event": eb,
            "ep_a": ep_a,
            "ep_b": ep_b,
            "epoch_gap": ep_b - ep_a,
            "interval_size": nn,
        })

    if not meta_records:
        return pd.DataFrame({
            "reason": ["no_intervals_found"],
            "early_count": [early_bits.bit_count()],
            "late_count": [late_bits.bit_count()],
        })

    # No huge bitsets are in this dataframe yet, so pandas cannot overflow.
    df = pd.DataFrame(meta_records)

    df = (
        df
        .sort_values("interval_size", ascending=False)
        .drop_duplicates(subset=["a_event", "b_event"])
        .head(keep_top)
        .reset_index(drop=True)
    )

    # Attach huge bitsets only after sorting/deduping, as dtype object.
    df["interval_bits"] = pd.Series(
        [interval_bits_store[int(k)] for k in df["_bit_id"]],
        dtype=object,
    )

    df = df.drop(columns=["_bit_id"])

    return df

def mm_from_diamond_table(
    causet,
    diamond_df,
    label="largest_diamond_audit",
    target_dim=4.0,
    seed=0,
):
    """
    Compute MM stats over a dataframe returned by largest_epoch_separated_diamonds().

    Patched version:
        Reads interval_bits as Python objects, avoiding pandas numeric coercion.
    """
    if diamond_df is None or len(diamond_df) == 0 or "interval_bits" not in diamond_df.columns:
        sample = {
            "dims": np.array([]),
            "widths": np.array([]),
            "order_fracs": np.array([]),
            "sizes": np.array([]),
            "complete_basins": np.array([]),
            "resolved_triplets": np.array([]),
            "endpoint_pairs": [],
            "tries": 0,
            "rejects": Counter({"no_diamonds": 1}),
        }
        return summarize_mm(label, sample, target_dim=target_dim, seed=seed), sample

    dims = []
    widths = []
    order_fracs = []
    sizes = []
    endpoints = []
    rejects = Counter()

    for row in diamond_df.itertuples(index=False):
        bits = int(getattr(row, "interval_bits"))
        nn = bits.bit_count()

        total = nn * (nn - 1) // 2

        if total <= 0:
            rejects["degenerate"] += 1
            continue

        rel = count_related_pairs(bits, causet.descendant_bits)
        r = rel / total
        d = mm_dim(r)

        if not np.isfinite(d):
            rejects["dimension_out_of_range"] += 1
            continue

        dims.append(d)
        widths.append(1.0 - r)
        order_fracs.append(r)
        sizes.append(nn)
        endpoints.append((getattr(row, "a_event"), getattr(row, "b_event")))

    sample = {
        "dims": np.asarray(dims),
        "widths": np.asarray(widths),
        "order_fracs": np.asarray(order_fracs),
        "sizes": np.asarray(sizes),
        "complete_basins": np.zeros(len(dims)),
        "resolved_triplets": np.zeros(len(dims)),
        "endpoint_pairs": endpoints,
        "tries": len(diamond_df),
        "rejects": rejects,
    }

    out = summarize_mm(label, sample, target_dim=target_dim, seed=seed)

    if len(sizes):
        out["largest_size"] = int(np.max(sizes))
        out["median_top_size"] = float(np.median(sizes))
    else:
        out["largest_size"] = np.nan
        out["median_top_size"] = np.nan

    return out, sample



# --- extracted notebook cell 72 definitions ---

def make_fractional_epoch_windows(
    source,
    n_windows=10,
    width_frac=0.12,
    min_width=4,
    mode="tail",
):
    """
    Make windows with width defined as a fraction of the run's total epoch span.

    This lets cap=256 and cap=1024 be compared more fairly, because they
    produce different total epoch counts.
    """
    ep_min, ep_max = source_epoch_range(source)

    ep_min = int(np.floor(ep_min))
    ep_max = int(np.ceil(ep_max))

    total = max(1, ep_max - ep_min + 1)
    width = max(min_width, int(round(width_frac * total)))

    return make_epoch_windows(
        source,
        n_windows=n_windows,
        width=width,
        mode=mode,
    )

def add_phase_columns(df):
    """
    Add normalized phase coordinates:
        phase_lo, phase_hi, phase_center
    """
    df = df.copy()

    if "growth_epochs" not in df.columns:
        df["growth_epochs"] = np.nan

    denom = df["growth_epochs"].replace(0, np.nan)

    df["phase_lo"] = df["ep_lo"] / denom
    df["phase_hi"] = df["ep_hi"] / denom
    df["phase_center"] = 0.5 * (df["phase_lo"] + df["phase_hi"])

    return df




# --- appendix: /mnt/data/deu_exp4_56_appendix.py ---

# === Experiment 4: spatial snapshot instrumentation and spatial-slice dimensions ===

from dataclasses import dataclass
from collections import defaultdict, Counter, deque
import itertools
import numpy as np
import pandas as pd

try:
    FoamRun
except NameError:
    @dataclass
    class FoamRun:
        elements: set
        children: dict
        event_role: dict
        event_kind: dict
        event_basin: dict
        event_epoch: dict
        stats: dict


@dataclass
class FoamSpatialSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_creator: dict
    face_basin: dict
    face_neighbors: dict
    stats: dict


def _face_edge_keys(nodes):
    return [frozenset(e) for e in itertools.combinations(sorted(nodes), 2)]


def adjacency_from_face_nodes(face_nodes):
    """
    Build the active face-edge adjacency graph from fid -> frozenset(3 vertices).
    Two active faces are adjacent iff they share a full edge.
    """
    edge_to_faces = defaultdict(list)

    for fid, nodes in face_nodes.items():
        for e in _face_edge_keys(nodes):
            edge_to_faces[e].append(fid)

    adj = {fid: set() for fid in face_nodes}

    for fids in edge_to_faces.values():
        if len(fids) < 2:
            continue
        for a, b in itertools.combinations(fids, 2):
            adj[a].add(b)
            adj[b].add(a)

    return adj, edge_to_faces


def connected_components_from_adj(adj):
    seen = set()
    comps = []

    for start in adj:
        if start in seen:
            continue

        comp = set([start])
        seen.add(start)
        dq = deque([start])

        while dq:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    dq.append(v)

        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def induced_adj(adj, nodes):
    nodes = set(nodes)
    return {u: set(adj.get(u, ())) & nodes for u in nodes}


def snapshot_face_adj(snapshot, component="largest"):
    """
    Return a face adjacency graph for the requested component.

    component:
        "all"      -> keep all active faces, possibly disconnected
        "largest"  -> restrict to largest connected component
        set/list    -> restrict to supplied face IDs
    """
    if getattr(snapshot, "face_neighbors", None):
        adj = {fid: set(ns) for fid, ns in snapshot.face_neighbors.items()}
    else:
        adj, _ = adjacency_from_face_nodes(snapshot.face_nodes)

    if component == "all":
        return adj

    if component == "largest":
        comps = connected_components_from_adj(adj)
        if not comps:
            return {}
        return induced_adj(adj, comps[0])

    return induced_adj(adj, component)


def spatial_slice_sanity(snapshot):
    """
    Basic triangulation/graph sanity for one spatial foam slice.
    """
    face_nodes = snapshot.face_nodes
    adj, edge_to_faces = adjacency_from_face_nodes(face_nodes)
    comps = connected_components_from_adj(adj)

    faces_n = len(face_nodes)
    vertices = set()
    for nodes in face_nodes.values():
        vertices.update(nodes)

    edges_n = len(edge_to_faces)
    vertex_degree = Counter()
    for e in edge_to_faces:
        a, b = tuple(e)
        vertex_degree[a] += 1
        vertex_degree[b] += 1

    face_degrees = np.array([len(adj[f]) for f in adj], dtype=float) if adj else np.array([])
    face_graph_edges = int(sum(len(ns) for ns in adj.values()) // 2)
    component_sizes = [len(c) for c in comps]
    largest_component_size = component_sizes[0] if component_sizes else 0
    largest_adj = induced_adj(adj, comps[0]) if comps else {}
    largest_face_degrees = np.array([len(largest_adj[f]) for f in largest_adj], dtype=float) if largest_adj else np.array([])

    boundary_edges = sum(1 for fs in edge_to_faces.values() if len(fs) == 1)
    interior_edges = sum(1 for fs in edge_to_faces.values() if len(fs) == 2)
    overglued_edges = sum(1 for fs in edge_to_faces.values() if len(fs) > 2)
    face_cycle_rank = face_graph_edges - faces_n + len(comps) if faces_n else 0

    avg_face_degree_largest = (
        float(np.mean(largest_face_degrees)) if len(largest_face_degrees) else np.nan
    )
    largest_fraction = largest_component_size / max(1, faces_n)

    if faces_n == 0:
        shape_hint = "empty"
    elif largest_fraction < 0.5:
        shape_hint = "fragmented"
    elif avg_face_degree_largest < 1.25:
        shape_hint = "mostly isolated"
    elif avg_face_degree_largest < 2.05 and face_cycle_rank <= max(1, len(comps)):
        shape_hint = "path/tree-like"
    elif avg_face_degree_largest < 2.40:
        shape_hint = "narrow/ribbon-like"
    else:
        shape_hint = "2D-mesh-like"

    return {
        "epoch": snapshot.epoch,
        "faces": faces_n,
        "vertices": len(vertices),
        "edges": edges_n,
        "euler_V_minus_E_plus_F": len(vertices) - edges_n + faces_n,
        "boundary_edges": boundary_edges,
        "interior_edges": interior_edges,
        "overglued_edges": overglued_edges,
        "avg_face_degree": float(np.mean(face_degrees)) if len(face_degrees) else np.nan,
        "max_face_degree": int(np.max(face_degrees)) if len(face_degrees) else 0,
        "avg_face_degree_largest_component": avg_face_degree_largest,
        "avg_vertex_degree": float(np.mean(list(vertex_degree.values()))) if vertex_degree else np.nan,
        "components": len(comps),
        "largest_component_size": largest_component_size,
        "largest_component_fraction": largest_fraction,
        "face_graph_edges": face_graph_edges,
        "face_graph_cycle_rank": int(face_cycle_rank),
        "shape_hint": shape_hint,
        "basin_splits_total": snapshot.stats.get("basin_splits", np.nan),
        "sterile_ticks_total": snapshot.stats.get("sterile_ticks", np.nan),
    }


def _ball_counts_for_center(adj, center, max_radius):
    seen = {center}
    frontier = {center}
    counts = [1]

    for _r in range(1, max_radius + 1):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        counts.append(len(seen))
        if not frontier:
            counts.extend([len(seen)] * (max_radius - _r))
            break

    return counts[: max_radius + 1]


def loglog_fit(x, y, min_points=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < min_points:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n_fit": int(len(x)),
        }

    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_fit": int(len(x)),
    }



def semilog_growth_fit(radius, y, min_points=3):
    """
    Fit log(y) = rate * radius + intercept. Useful as a diagnostic for
    exponential/hyperbolic ball growth, where a finite Hausdorff power-law
    dimension is not the right model.
    """
    radius = np.asarray(radius, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(radius) & np.isfinite(y) & (radius >= 0) & (y > 0)
    radius = radius[mask]
    y = y[mask]

    if len(radius) < min_points:
        return {
            "rate": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n_fit": int(len(radius)),
        }

    ly = np.log(y)
    rate, intercept = np.polyfit(radius, ly, 1)
    pred = rate * radius + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "rate": float(rate),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_fit": int(len(radius)),
    }

def estimate_spatial_hausdorff(
    snapshot,
    max_radius=None,
    n_centers=96,
    seed=0,
    component="largest",
    fit_min_radius=2,
    fit_max_fraction=0.35,
):
    """
    Estimate spatial Hausdorff dimension from face-graph ball growth:
        <N(R)> ~ R^d_H.
    """
    rng = np.random.default_rng(seed)
    adj = snapshot_face_adj(snapshot, component=component)
    nodes = list(adj)
    n = len(nodes)

    if n == 0:
        summary = {
            "epoch": snapshot.epoch,
            "component": component,
            "component_size": 0,
            "hausdorff_dim": np.nan,
            "fit_r2": np.nan,
            "n_centers": 0,
            "max_radius": 0,
        }
        return summary, pd.DataFrame()

    if max_radius is None:
        max_radius = int(min(32, max(4, round(np.sqrt(n)))))

    n_centers = int(min(n_centers, n))
    centers = rng.choice(nodes, size=n_centers, replace=False)

    counts = np.array([
        _ball_counts_for_center(adj, int(c), max_radius)
        for c in centers
    ], dtype=float)

    radii = np.arange(max_radius + 1)
    mean_counts = counts.mean(axis=0)
    median_counts = np.median(counts, axis=0)
    q25 = np.quantile(counts, 0.25, axis=0)
    q75 = np.quantile(counts, 0.75, axis=0)

    growth_df = pd.DataFrame({
        "epoch": snapshot.epoch,
        "radius": radii,
        "mean_ball_faces": mean_counts,
        "median_ball_faces": median_counts,
        "q25_ball_faces": q25,
        "q75_ball_faces": q75,
    })

    eligible = (
        (radii >= fit_min_radius)
        & (mean_counts > 1.0)
        & (mean_counts <= max(2.0, fit_max_fraction * n))
    )

    if eligible.sum() < 3:
        eligible = (
            (radii >= 1)
            & (mean_counts > 1.0)
            & (mean_counts < max(3.0, 0.80 * n))
        )

    fit = loglog_fit(radii[eligible], mean_counts[eligible], min_points=3)
    exp_fit = semilog_growth_fit(radii[eligible], mean_counts[eligible], min_points=3)

    if np.isfinite(exp_fit["r2"]) and np.isfinite(fit["r2"]):
        if exp_fit["r2"] > fit["r2"] + 0.03 and exp_fit["r2"] > 0.95:
            growth_model_hint = "exponential-like"
        elif fit["r2"] >= exp_fit["r2"] - 0.03 and fit["r2"] > 0.95:
            growth_model_hint = "power-law-like"
        else:
            growth_model_hint = "mixed/finite-size"
    else:
        growth_model_hint = "undetermined"

    growth_df["fit_used"] = eligible
    growth_df["local_dim_from_mean"] = np.nan
    for k in range(1, len(growth_df)):
        r0, r1 = radii[k - 1], radii[k]
        n0, n1 = mean_counts[k - 1], mean_counts[k]
        if r0 > 0 and r1 > 0 and n0 > 0 and n1 > 0 and r0 != r1 and n0 != n1:
            growth_df.loc[k, "local_dim_from_mean"] = (
                np.log(n1 / n0) / np.log(r1 / r0)
            )

    used_radii = radii[eligible]
    summary = {
        "epoch": snapshot.epoch,
        "component": component,
        "component_size": n,
        "hausdorff_dim": fit["slope"],
        "fit_r2": fit["r2"],
        "n_fit": fit["n_fit"],
        "fit_radius_min": int(used_radii.min()) if len(used_radii) else np.nan,
        "fit_radius_max": int(used_radii.max()) if len(used_radii) else np.nan,
        "exponential_rate_per_radius": exp_fit["rate"],
        "exponential_fit_r2": exp_fit["r2"],
        "growth_model_hint": growth_model_hint,
        "n_centers": n_centers,
        "max_radius": max_radius,
        "mean_ball_at_max_radius": float(mean_counts[-1]),
    }

    return summary, growth_df


def estimate_spatial_spectral(
    snapshot,
    max_steps=64,
    n_walkers=4000,
    seed=0,
    lazy=0.5,
    component="largest",
    fit_min_step=4,
    fit_max_step=None,
):
    """
    Estimate spectral dimension from lazy random-walk return probability:
        P_return(t) ~ t^(-d_s/2).
    """
    rng = np.random.default_rng(seed)
    adj0 = snapshot_face_adj(snapshot, component=component)
    nodes = list(adj0)
    n = len(nodes)

    if n == 0:
        summary = {
            "epoch": snapshot.epoch,
            "component": component,
            "component_size": 0,
            "spectral_dim": np.nan,
            "fit_r2": np.nan,
            "n_walkers": 0,
            "max_steps": max_steps,
        }
        return summary, pd.DataFrame()

    node_to_i = {u: i for i, u in enumerate(nodes)}
    neigh = [np.array([node_to_i[v] for v in adj0[u]], dtype=int) for u in nodes]

    n_walkers = int(max(1, n_walkers))
    starts = rng.integers(0, n, size=n_walkers, endpoint=False)
    pos = starts.copy()
    return_counts = np.zeros(max_steps + 1, dtype=int)
    return_counts[0] = n_walkers

    for t in range(1, max_steps + 1):
        for w in range(n_walkers):
            if rng.random() < lazy:
                continue
            nb = neigh[pos[w]]
            if len(nb):
                pos[w] = int(nb[rng.integers(0, len(nb))])
        return_counts[t] = int(np.sum(pos == starts))

    steps = np.arange(max_steps + 1)
    p_return = return_counts / float(n_walkers)
    spectral_df = pd.DataFrame({
        "epoch": snapshot.epoch,
        "step": steps,
        "return_count": return_counts,
        "return_prob": p_return,
        "component_size": n,
        "n_walkers": n_walkers,
        "lazy": lazy,
    })

    if fit_max_step is None:
        fit_max_step = max_steps

    plateau = 1.0 / max(1, n)
    min_nonzero = 0.5 / max(1, n_walkers)
    eligible = (
        (steps >= fit_min_step)
        & (steps <= fit_max_step)
        & (p_return > max(min_nonzero, 2.0 * plateau))
    )

    if eligible.sum() < 3:
        eligible = (steps >= 2) & (p_return > 0)

    fit = loglog_fit(steps[eligible], p_return[eligible], min_points=3)
    spectral_df["fit_used"] = eligible

    used_steps = steps[eligible]
    summary = {
        "epoch": snapshot.epoch,
        "component": component,
        "component_size": n,
        "spectral_dim": -2.0 * fit["slope"] if np.isfinite(fit["slope"]) else np.nan,
        "fit_slope": fit["slope"],
        "fit_r2": fit["r2"],
        "n_fit": fit["n_fit"],
        "fit_step_min": int(used_steps.min()) if len(used_steps) else np.nan,
        "fit_step_max": int(used_steps.max()) if len(used_steps) else np.nan,
        "n_walkers": n_walkers,
        "max_steps": max_steps,
        "lazy": lazy,
    }

    return summary, spectral_df


def _should_record_snapshot(epoch, snapshot_epochs, snapshot_every):
    if snapshot_epochs is not None and int(epoch) in snapshot_epochs:
        return True
    if snapshot_every is not None and snapshot_every > 0 and int(epoch) % int(snapshot_every) == 0:
        return True
    return False


def grow_typed_foam_causet_spatial(
    target_basin_splits=10000,
    seed=0,
    scheduler="capped",
    max_epochs=None,
    max_splits_per_epoch=256,
    max_ticks_per_epoch=256,
    unbounded_ticks=True,
    snapshot_epochs=None,
    snapshot_every=None,
    record_initial=True,
    record_final=True,
):
    """
    Instrumented version of the existing typed triangular foam generator.

    It preserves the capped/unbounded move rules used above, but additionally
    records selected active-face snapshots so spatial slices can be audited.

    scheduler:
        "capped"    -> same cap logic as grow_typed_foam_causet_parallel
        "unbounded" -> split every frustrated face from the epoch snapshot
    """
    if scheduler not in {"capped", "unbounded"}:
        raise ValueError("scheduler must be 'capped' or 'unbounded'")

    snapshot_epochs = None if snapshot_epochs is None else {int(e) for e in snapshot_epochs}
    rng = np.random.default_rng(seed)

    faces = {}
    face_types = {}
    face_creator = {}
    face_basin = {}
    edge_to_faces = defaultdict(set)
    active = set()

    causal_parents = defaultdict(set)
    event_role = {}
    event_kind = {}
    event_basin = {}
    event_epoch = {}

    next_face = 0
    next_node = 6
    next_event = 0
    next_basin = 0
    epoch = 0

    stats = Counter()
    epoch_log = []
    spatial_snapshots = {}

    def make_event(role, kind, basin, parents=(), ep=None):
        nonlocal next_event
        ev = next_event
        next_event += 1
        ps = {int(p) for p in parents if p is not None and int(p) != ev}
        causal_parents[ev] |= ps
        event_role[ev] = role
        event_kind[ev] = kind
        event_basin[ev] = basin
        event_epoch[ev] = epoch if ep is None else ep
        stats[f"events_{role}"] += 1
        stats[f"kind_{kind}"] += 1
        stats["parent_sum"] += len(ps)
        stats["parent_max"] = max(stats.get("parent_max", 0), len(ps))
        return ev

    def add_face(nodes, ftype, creator, basin):
        nonlocal next_face
        fid = next_face
        next_face += 1
        nodes = frozenset(int(x) for x in nodes)
        faces[fid] = nodes
        face_types[fid] = ftype
        face_creator[fid] = int(creator)
        face_basin[fid] = basin
        for e in itertools.combinations(sorted(nodes), 2):
            edge_to_faces[frozenset(e)].add(fid)
        active.add(fid)
        return fid

    def remove_face(fid):
        for e in itertools.combinations(sorted(faces[fid]), 2):
            edge_to_faces[frozenset(e)].discard(fid)
            if not edge_to_faces[frozenset(e)]:
                del edge_to_faces[frozenset(e)]
        active.discard(fid)
        del faces[fid]
        del face_types[fid]
        del face_creator[fid]
        del face_basin[fid]

    def get_neighbors_current(fid):
        neighbors = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            neighbors |= edge_to_faces[frozenset(e)]
        neighbors.discard(fid)
        return neighbors

    def snapshot_raw():
        active0 = set(active)
        types0 = dict(face_types)
        creator0 = dict(face_creator)
        basin0 = dict(face_basin)
        faces0 = dict(faces)
        neigh0 = {fid: get_neighbors_current(fid) & active0 for fid in active0}
        return active0, types0, creator0, basin0, faces0, neigh0

    def record_snapshot(ep):
        active0, types0, creator0, basin0, faces0, neigh0 = snapshot_raw()
        spatial_snapshots[int(ep)] = FoamSpatialSnapshot(
            epoch=int(ep),
            active_faces=active0,
            face_nodes=faces0,
            face_types=types0,
            face_creator=creator0,
            face_basin=basin0,
            face_neighbors=neigh0,
            stats=dict(stats),
        )

    def is_frustrated0(fid, types0, neigh0):
        if types0[fid] != "S":
            return False
        neighbor_types = {types0[n] for n in neigh0[fid]}
        return ("G" in neighbor_types) and ("I" not in neighbor_types)

    def causal_context0(fid, creator0, neigh0):
        parents = {creator0[fid]}
        for g in neigh0[fid]:
            parents.add(creator0[g])
        return parents

    # Same open seed patch as the existing parallel/unbounded generators.
    root = make_event("ROOT", "root", -1, [], ep=-1)
    b0 = next_basin
    next_basin += 1
    t0 = make_event("T", "basin_driver", b0, [root], ep=0)
    s0 = make_event("S", "resolved_mode", b0, [t0], ep=0)
    i0 = make_event("I", "resolved_mode", b0, [t0], ep=0)
    g0 = make_event("G", "resolved_mode", b0, [t0], ep=0)

    add_face((0, 1, 2), "S", s0, b0)
    add_face((0, 1, 3), "G", g0, b0)
    add_face((2, 4, 5), "I", i0, b0)
    add_face((3, 4, 5), "S", s0, b0)

    if record_initial or _should_record_snapshot(0, snapshot_epochs, snapshot_every):
        record_snapshot(0)

    if max_epochs is None:
        factor = 10 if scheduler == "unbounded" else 5
        max_epochs = max(10, factor * target_basin_splits)

    epoch = 1

    while stats["basin_splits"] < target_basin_splits and epoch <= max_epochs:
        active0, types0, creator0, basin0, faces0, neigh0 = snapshot_raw()
        frustrated = [fid for fid in active0 if is_frustrated0(fid, types0, neigh0)]
        frontier_size = len(frustrated)
        stats["frontier_max"] = max(stats.get("frontier_max", 0), frontier_size)

        if frustrated:
            rng.shuffle(frustrated)
            remaining = target_basin_splits - stats["basin_splits"]
            if scheduler == "unbounded":
                selected = list(frustrated)
            else:
                selected = frustrated[: min(frontier_size, max_splits_per_epoch, remaining)]

            split_records = []
            for fid in selected:
                b = next_basin
                next_basin += 1
                t = make_event(
                    "T",
                    "basin_driver",
                    b,
                    causal_context0(fid, creator0, neigh0),
                    ep=epoch,
                )
                s = make_event("S", "resolved_mode", b, [t], ep=epoch)
                i = make_event("I", "resolved_mode", b, [t], ep=epoch)
                g = make_event("G", "resolved_mode", b, [t], ep=epoch)
                split_records.append((fid, b, s, i, g, sorted(faces0[fid])))

            actual_splits = 0
            for fid, b, s, i, g, old_nodes in split_records:
                if fid not in active:
                    stats["selected_already_removed"] += 1
                    continue
                a_node, b_node, c_node = old_nodes
                new_node = next_node
                next_node += 1
                remove_face(fid)
                add_face((new_node, a_node, b_node), "S", s, b)
                add_face((new_node, a_node, c_node), "I", i, b)
                add_face((new_node, b_node, c_node), "G", g, b)
                stats["basin_splits"] += 1
                actual_splits += 1

            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(stats.get("max_splits_in_epoch", 0), actual_splits)
            if scheduler == "unbounded" and actual_splits != frontier_size:
                stats["frontier_full_split_violations"] += 1

            epoch_log.append({
                "epoch": epoch,
                "kind": "split",
                "frontier_size": frontier_size,
                "actual_splits": actual_splits,
                "ticks": 0,
                "active_faces": len(active),
                "basin_splits_total": stats["basin_splits"],
            })

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
            if scheduler == "unbounded" and unbounded_ticks:
                selected = list(candidates)
            else:
                selected = candidates[: min(len(candidates), max_ticks_per_epoch)]

            tick_records = []
            for fid in selected:
                t = make_event(
                    "T",
                    "sterile_tick",
                    -1,
                    causal_context0(fid, creator0, neigh0),
                    ep=epoch,
                )
                old_type = types0[fid]
                if old_type == "I":
                    new_type = "S"
                elif old_type == "G":
                    new_type = "I"
                else:
                    new_type = "G"
                tick_records.append((fid, t, old_type, new_type))

            actual_ticks = 0
            for fid, t, old_type, new_type in tick_records:
                if fid not in active:
                    continue
                face_types[fid] = new_type
                face_creator[fid] = t
                face_basin[fid] = -1
                stats["sterile_ticks"] += 1
                stats[f"sterile_{old_type}_to_{new_type}"] += 1
                actual_ticks += 1

            stats["tick_epochs"] += 1
            stats["max_ticks_in_epoch"] = max(stats.get("max_ticks_in_epoch", 0), actual_ticks)

            epoch_log.append({
                "epoch": epoch,
                "kind": "tick",
                "frontier_size": 0,
                "actual_splits": 0,
                "ticks": actual_ticks,
                "active_faces": len(active),
                "basin_splits_total": stats["basin_splits"],
            })

            if actual_ticks == 0:
                stats["sterile_starved"] += 1
                if _should_record_snapshot(epoch, snapshot_epochs, snapshot_every):
                    record_snapshot(epoch)
                break

        if _should_record_snapshot(epoch, snapshot_epochs, snapshot_every):
            record_snapshot(epoch)

        epoch += 1

    stats["epochs"] = epoch - 1
    stats["final_active_faces"] = len(active)
    stats["final_nodes"] = next_node
    stats["final_events_including_root"] = next_event
    stats["final_basins_including_initial"] = next_basin
    stats["avg_parents_per_event"] = stats["parent_sum"] / max(1, next_event)
    stats["scheduler"] = scheduler
    stats["unbounded_frontier"] = scheduler == "unbounded"
    stats["unbounded_ticks"] = bool(unbounded_ticks) if scheduler == "unbounded" else False

    if record_final and int(stats["epochs"]) not in spatial_snapshots:
        record_snapshot(int(stats["epochs"]))

    elements = {e for e in causal_parents if e != root}
    for e in event_role:
        if e != root:
            elements.add(e)

    children = {e: set() for e in elements}
    for e, ps in causal_parents.items():
        if e == root:
            continue
        for p in ps:
            if p in elements:
                children.setdefault(p, set()).add(e)

    out = FoamRun(
        elements=elements,
        children=children,
        event_role=event_role,
        event_kind=event_kind,
        event_basin=event_basin,
        event_epoch=event_epoch,
        stats=dict(stats),
    )
    out.epoch_log = pd.DataFrame(epoch_log)
    out.spatial_snapshots = dict(sorted(spatial_snapshots.items()))
    return out


def audit_spatial_snapshots(
    run_obj,
    epochs=None,
    hausdorff_centers=96,
    hausdorff_max_radius=None,
    spectral_walkers=0,
    spectral_max_steps=64,
    seed=0,
):
    """
    Run sanity, Hausdorff, and optional spectral audits over stored snapshots.
    """
    if not hasattr(run_obj, "spatial_snapshots"):
        raise ValueError("run_obj has no spatial_snapshots; use grow_typed_foam_causet_spatial().")

    snapshots = run_obj.spatial_snapshots
    if epochs is None:
        epochs = sorted(snapshots)
    else:
        epochs = [int(e) for e in epochs if int(e) in snapshots]

    sanity_rows = []
    haus_rows = []
    spectral_rows = []
    growth_curves = {}
    spectral_curves = {}

    for k, ep in enumerate(epochs):
        snap = snapshots[ep]
        sanity_rows.append(spatial_slice_sanity(snap))

        h_summary, h_curve = estimate_spatial_hausdorff(
            snap,
            max_radius=hausdorff_max_radius,
            n_centers=hausdorff_centers,
            seed=seed + 1000 * k + ep,
            component="largest",
        )
        haus_rows.append(h_summary)
        growth_curves[ep] = h_curve

        if spectral_walkers and spectral_walkers > 0:
            s_summary, s_curve = estimate_spatial_spectral(
                snap,
                max_steps=spectral_max_steps,
                n_walkers=spectral_walkers,
                seed=seed + 2000 * k + ep,
                component="largest",
            )
            spectral_rows.append(s_summary)
            spectral_curves[ep] = s_curve

    sanity_df = pd.DataFrame(sanity_rows)
    hausdorff_df = pd.DataFrame(haus_rows)
    spectral_df = pd.DataFrame(spectral_rows)

    summary_df = sanity_df.merge(
        hausdorff_df,
        on="epoch",
        how="left",
        suffixes=("", "_haus"),
    )

    if len(spectral_df):
        summary_df = summary_df.merge(
            spectral_df[[
                "epoch", "spectral_dim", "fit_r2", "n_fit",
                "fit_step_min", "fit_step_max", "n_walkers",
            ]].rename(columns={
                "fit_r2": "spectral_fit_r2",
                "n_fit": "spectral_n_fit",
            }),
            on="epoch",
            how="left",
        )

    return summary_df, growth_curves, spectral_curves
# === Experiments 5 and 6: causal volume growth and mid-layer / antichain scaling ===

import numpy as np
import pandas as pd
from collections import Counter


def longest_chain_layers_bits(causet, bits):
    """
    Rank/layer decomposition of an induced causal interval.

    Uses the transitive ancestor bitsets, so the height is the longest chain in
    the poset, not merely the longest path through direct parent edges.
    Returns:
        ranks: dict[index -> rank starting at 1]
        layer_counts: Counter(rank -> number of elements)
        height: max rank
    """
    bits = int(bits)
    ranks = {}
    layer_counts = Counter()

    for i in iter_bits(bits):
        ancestor_bits = causet.ancestor_bits[i] & bits
        prev = 0
        for j in iter_bits(ancestor_bits):
            hj = ranks.get(j, 0)
            if hj > prev:
                prev = hj
        ri = prev + 1
        ranks[i] = ri
        layer_counts[ri] += 1

    height = max(layer_counts.keys(), default=0)
    return ranks, layer_counts, height


def diamond_scaling_table(causet, diamond_df, max_diamonds=None):
    """
    Compute V, longest-chain height, and antichain-layer data for diamonds.

    V is the strict interval size already stored by largest_epoch_separated_diamonds.
    h_inclusive includes the two endpoints, which is the better proxy for the
    endpoint-to-endpoint timelike depth. h_strict is also reported.
    """
    if diamond_df is None or len(diamond_df) == 0 or "interval_bits" not in diamond_df.columns:
        return pd.DataFrame()

    rows = []
    iterator = diamond_df.head(max_diamonds).itertuples(index=False) if max_diamonds else diamond_df.itertuples(index=False)

    for local_id, row in enumerate(iterator):
        strict_bits = int(getattr(row, "interval_bits"))
        a_index = int(getattr(row, "a_index"))
        b_index = int(getattr(row, "b_index"))
        endpoint_bits = bit_at(a_index) | bit_at(b_index)
        inclusive_bits = strict_bits | endpoint_bits

        V_strict = strict_bits.bit_count()
        V_inclusive = inclusive_bits.bit_count()

        strict_ranks, strict_layers, h_strict = longest_chain_layers_bits(causet, strict_bits)
        incl_ranks, incl_layers, h_inclusive = longest_chain_layers_bits(causet, inclusive_bits)

        # For spatial cross-sections, count layers inside the strict interval only,
        # using ranks inherited from the inclusive endpoint-to-endpoint interval.
        strict_layer_counts_inclusive_rank = Counter()
        for i in iter_bits(strict_bits):
            strict_layer_counts_inclusive_rank[incl_ranks.get(i, 0)] += 1

        if h_inclusive > 0:
            mid_rank = int(round((h_inclusive + 1) / 2.0))
            mid_candidates = [mid_rank]
            if h_inclusive % 2 == 0:
                mid_candidates = [h_inclusive // 2, h_inclusive // 2 + 1]
            mid_layer_size = max(
                strict_layer_counts_inclusive_rank.get(r, 0)
                for r in mid_candidates
            )
            mid_band_size = sum(
                strict_layer_counts_inclusive_rank.get(r, 0)
                for r in sorted(set(mid_candidates))
            )
        else:
            mid_rank = 0
            mid_layer_size = 0
            mid_band_size = 0

        max_layer_size = max(strict_layer_counts_inclusive_rank.values(), default=0)
        mean_layer_size = (
            float(np.mean(list(strict_layer_counts_inclusive_rank.values())))
            if strict_layer_counts_inclusive_rank else np.nan
        )

        rel = count_related_pairs(strict_bits, causet.descendant_bits) if V_strict >= 2 else 0
        total = V_strict * (V_strict - 1) // 2
        r = rel / total if total > 0 else np.nan
        d_mm = mm_dim(r) if total > 0 and np.isfinite(r) else np.nan

        rows.append({
            "diamond_id": local_id,
            "a_index": a_index,
            "b_index": b_index,
            "a_event": getattr(row, "a_event", causet.events[a_index]),
            "b_event": getattr(row, "b_event", causet.events[b_index]),
            "ep_a": getattr(row, "ep_a", np.nan),
            "ep_b": getattr(row, "ep_b", np.nan),
            "epoch_gap": getattr(row, "epoch_gap", np.nan),
            "V_strict": V_strict,
            "V_inclusive": V_inclusive,
            "h_strict": h_strict,
            "h_inclusive": h_inclusive,
            "mid_rank": mid_rank,
            "mid_layer_size": mid_layer_size,
            "mid_band_size": mid_band_size,
            "max_layer_size": max_layer_size,
            "mean_nonempty_layer_size": mean_layer_size,
            "n_nonempty_strict_layers": len(strict_layer_counts_inclusive_rank),
            "r_strict": r,
            "MM_dim_strict": d_mm,
        })

    return pd.DataFrame(rows)


def summarize_causal_scaling(scaling_df, label="causal_scaling"):
    """
    Fit:
        V_strict ~ h_inclusive^d
        A_mid    ~ h_inclusive^(d-1)
        A_max    ~ h_inclusive^(d-1)
    """
    if scaling_df is None or len(scaling_df) == 0:
        return {
            "label": label,
            "n_diamonds": 0,
            "volume_slope": np.nan,
            "mid_layer_slope": np.nan,
            "max_layer_slope": np.nan,
        }

    valid_h = scaling_df["h_inclusive"].to_numpy(dtype=float)
    V = scaling_df["V_strict"].to_numpy(dtype=float)
    A_mid = scaling_df["mid_layer_size"].to_numpy(dtype=float)
    A_max = scaling_df["max_layer_size"].to_numpy(dtype=float)

    vol_fit = loglog_fit(valid_h, V, min_points=3)
    mid_fit = loglog_fit(valid_h, A_mid, min_points=3)
    max_fit = loglog_fit(valid_h, A_max, min_points=3)

    return {
        "label": label,
        "n_diamonds": int(len(scaling_df)),
        "V_med": float(np.median(V)) if len(V) else np.nan,
        "V_max": int(np.max(V)) if len(V) else 0,
        "h_med": float(np.median(valid_h)) if len(valid_h) else np.nan,
        "h_max": int(np.max(valid_h)) if len(valid_h) else 0,
        "MM_med": float(np.nanmedian(scaling_df["MM_dim_strict"])) if "MM_dim_strict" in scaling_df else np.nan,
        "r_med": float(np.nanmedian(scaling_df["r_strict"])) if "r_strict" in scaling_df else np.nan,
        "volume_slope": vol_fit["slope"],
        "volume_fit_r2": vol_fit["r2"],
        "volume_n_fit": vol_fit["n_fit"],
        "mid_layer_slope": mid_fit["slope"],
        "mid_layer_fit_r2": mid_fit["r2"],
        "mid_layer_n_fit": mid_fit["n_fit"],
        "max_layer_slope": max_fit["slope"],
        "max_layer_fit_r2": max_fit["r2"],
        "max_layer_n_fit": max_fit["n_fit"],
    }


def largest_diamond_distance_scan_with_scaling(
    source,
    windows,
    max_window_events=12000,
    sterile_mode="singleton",
    rg_epoch_width=1,
    rg_max_block_size=4,
    rg_signature_mode="degree",
    n_endpoint_samples=100000,
    keep_top=200,
    min_epoch_gap=3,
    min_size=2,
    seed=120000,
    target_dim=3.0,
    max_scaling_diamonds=None,
    keep_causets=False,
):
    """
    Same safe-RG2 largest-diamond audit as above, but immediately computes
    causal volume and antichain scaling while the window's rg2_causet is still
    available.
    """
    rows = []
    diamond_tables = {}
    scaling_tables = {}
    causets_by_window = {} if keep_causets else None

    for wi, (ep_lo, ep_hi) in enumerate(windows):
        print(f"scaling window {wi + 1}/{len(windows)}: epochs {ep_lo} to {ep_hi}")

        raw_c, raw_stats = build_epoch_window_causet(
            source,
            ep_lo,
            ep_hi,
            max_events=max_window_events,
            seed=seed + wi,
        )

        if raw_c is None or len(raw_c.events) < 10:
            rows.append({
                "window_id": wi,
                "ep_lo": ep_lo,
                "ep_hi": ep_hi,
                "stage": "largest_safe_rg2_scaling",
                "reason": "raw_window_too_small",
            })
            continue

        basin_c, basin_stats = make_deu_basin_macro_causet(
            raw_c,
            sterile_mode=sterile_mode,
            relation_mode="direct",
        )

        rg2_c, rg2_stats = rg_coarse_grain_once_safe(
            basin_c,
            epoch_width=rg_epoch_width,
            max_block_size=rg_max_block_size,
            signature_mode=rg_signature_mode,
            relation_mode="direct",
            include_role=False,
        )

        if keep_causets:
            causets_by_window[wi] = rg2_c

        diamond_df = largest_epoch_separated_diamonds(
            rg2_c,
            n_endpoint_samples=n_endpoint_samples,
            keep_top=keep_top,
            early_frac=0.25,
            late_frac=0.25,
            min_epoch_gap=min_epoch_gap,
            min_size=min_size,
            seed=seed + 1000 + wi,
        )
        diamond_tables[wi] = diamond_df

        mm_row, _sample = mm_from_diamond_table(
            rg2_c,
            diamond_df,
            label=f"largest_safeRG2_ep{ep_lo}_{ep_hi}",
            target_dim=target_dim,
            seed=seed + 2000 + wi,
        )

        scaling_df = diamond_scaling_table(
            rg2_c,
            diamond_df,
            max_diamonds=max_scaling_diamonds,
        )
        scaling_tables[wi] = scaling_df

        scaling_row = summarize_causal_scaling(
            scaling_df,
            label=f"scaling_safeRG2_ep{ep_lo}_{ep_hi}",
        )

        row = dict(mm_row)
        row.update(scaling_row)
        row.update({
            "window_id": wi,
            "stage": "largest_safe_rg2_scaling",
            "ep_lo": ep_lo,
            "ep_hi": ep_hi,
            "raw_events": len(raw_c.events),
            "basin_events": len(basin_c.events),
            "rg2_events": len(rg2_c.events),
            "scc_merged": rg2_stats.get("scc_merged", 0),
            "scc_largest": rg2_stats.get("scc_largest", 1),
            "n_diamonds_found": (
                len(diamond_df)
                if diamond_df is not None and "interval_bits" in diamond_df.columns
                else 0
            ),
        })
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if keep_causets:
        return out_df, diamond_tables, scaling_tables, causets_by_window
    return out_df, diamond_tables, scaling_tables


def plot_causal_scaling(scaling_df, title_prefix="safe-RG2 diamonds"):
    """
    Optional quick-look plots. Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    if scaling_df is None or len(scaling_df) == 0:
        print("No scaling rows to plot.")
        return

    # Volume growth plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scaling_df["h_inclusive"], scaling_df["V_strict"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("longest-chain height h, inclusive endpoints")
    ax.set_ylabel("strict interval volume V")
    ax.set_title(f"{title_prefix}: V vs h")
    plt.show()

    # Mid/max antichain plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scaling_df["h_inclusive"], scaling_df["mid_layer_size"], label="mid layer")
    ax.scatter(scaling_df["h_inclusive"], scaling_df["max_layer_size"], marker="x", label="max layer")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("longest-chain height h, inclusive endpoints")
    ax.set_ylabel("layer / antichain size")
    ax.set_title(f"{title_prefix}: layer scaling")
    ax.legend()
    plt.show()





# --- refinement: /mnt/data/deu_exp4_56_refinement.py ---

# === Experiment 4b/5b/6b refinements after first spatial + causal-scaling readout ===

from dataclasses import dataclass
from collections import defaultdict, Counter, deque
import heapq
import itertools
import numpy as np
import pandas as pd


def diagnose_causal_scaling_windows(df, target_dim=3.0):
    """
    Diagnose whether per-window causal volume/antichain power-law fits are meaningful.

    The earlier width-8 runs produce good largest-diamond MM rows, but most windows
    have h_inclusive concentrated in only a few values.  Power-law slopes should not
    be interpreted unless there is a genuine height lever arm.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if "h_max" in out and "h_med" in out:
        out["height_lever_hint"] = np.where(
            (out["h_max"] >= 2.0 * out["h_med"]) | (out["h_max"] >= 16),
            "usable/inspect",
            "height-starved",
        )
    else:
        out["height_lever_hint"] = "unknown"

    if "volume_fit_r2" in out:
        out["volume_fit_hint"] = np.where(
            (out["height_lever_hint"] != "height-starved") & (out["volume_fit_r2"] >= 0.80),
            "potentially meaningful",
            "do not interpret slope",
        )

    if "mid_layer_fit_r2" in out:
        out["mid_layer_fit_hint"] = np.where(
            (out["height_lever_hint"] != "height-starved") & (out["mid_layer_fit_r2"] >= 0.80),
            "potentially meaningful",
            "do not interpret slope",
        )

    out["MM_abs_err_to_target"] = (out.get("MM_med", np.nan) - target_dim).abs()
    return out


# ---------------------------------------------------------------------------
# Fast full/broad-run basin compression without building the full raw bitset causet
# ---------------------------------------------------------------------------

def _deu_direct_block_key(e, role, kind, basin, epoch, sterile_mode="singleton", include_root=False):
    if role == "ROOT" and not include_root:
        return None

    if basin is not None and basin >= 0 and role in {"T", "S", "I", "G"}:
        return ("basin", int(basin))

    if kind == "sterile_tick" or role == "T":
        if sterile_mode == "singleton":
            return ("sterile", int(e))
        if sterile_mode == "epoch":
            return ("sterile_epoch", int(epoch))
        if sterile_mode == "drop":
            return None
        raise ValueError("Unknown sterile_mode.")

    return ("other", int(e))


def make_deu_basin_macro_causet_direct_from_source(
    source,
    sterile_mode="singleton",
    ep_lo=None,
    ep_hi=None,
    max_raw_events=None,
    seed=0,
):
    """
    Direct-edge basin compression from a FoamRun-like source, without first building
    a raw transitive-closure bitset causet.

    This is useful for Experiment 5/6 because broad windows need a large height range.
    Building the full raw 40k+ event bitset can be memory-heavy; compressing first
    is much cheaper and preserves direct macro-causal edges.
    """
    rng = np.random.default_rng(seed)
    elements, children, event_role, event_kind, event_basin, event_epoch = extract_source_fields(source)

    selected = []
    for e in elements:
        ep = event_epoch.get(e, np.nan)
        if ep_lo is not None and ep < ep_lo:
            continue
        if ep_hi is not None and ep > ep_hi:
            continue
        selected.append(e)

    selected_before_cap = len(selected)
    if max_raw_events is not None and len(selected) > int(max_raw_events):
        selected = list(rng.choice(np.array(selected, dtype=object), size=int(max_raw_events), replace=False))
    selected = set(selected)

    raw_blocks = defaultdict(list)
    for e in selected:
        role = event_role.get(e, "?")
        kind = event_kind.get(e, "?")
        basin = event_basin.get(e, -1)
        epoch = event_epoch.get(e, 0)
        key = _deu_direct_block_key(e, role, kind, basin, epoch, sterile_mode=sterile_mode)
        if key is not None:
            raw_blocks[key].append(e)

    def block_sort_key(item):
        key, es = item
        eps = [event_epoch.get(e, 0.0) for e in es]
        return (float(np.mean(eps)) if eps else 0.0, min(es), str(key))

    block_items = sorted(raw_blocks.items(), key=block_sort_key)
    key_to_macro = {key: m for m, (key, es) in enumerate(block_items)}
    macro_to_events = {key_to_macro[key]: list(es) for key, es in block_items}
    event_to_macro = {}
    for m, es in macro_to_events.items():
        for e in es:
            event_to_macro[e] = m

    M = len(macro_to_events)
    macro_children = {m: set() for m in range(M)}
    crossing_edges = 0

    for p in selected:
        mp = event_to_macro.get(p, None)
        if mp is None:
            continue
        for c in children.get(p, ()):  # direct raw edge only
            if c not in selected:
                continue
            mc = event_to_macro.get(c, None)
            if mc is None or mc == mp:
                continue
            macro_children[mp].add(mc)
            crossing_edges += 1

    condensed_elements, condensed_children, old_to_new, scc_stats = condense_children_if_needed(
        set(range(M)),
        macro_children,
    )

    condensed_to_old = defaultdict(list)
    for old, new in old_to_new.items():
        condensed_to_old[new].append(old)

    macro_role = {}
    macro_kind = {}
    macro_basin = {}
    macro_epoch = {}

    for new_m in condensed_elements:
        old_ms = condensed_to_old[new_m]
        es = []
        for old_m in old_ms:
            es.extend(macro_to_events.get(old_m, []))
        eps = [event_epoch.get(e, 0.0) for e in es]
        basins = {event_basin.get(e, -1) for e in es if event_basin.get(e, -1) is not None and event_basin.get(e, -1) >= 0}
        roles = Counter(event_role.get(e, "?") for e in es)
        kinds = Counter(event_kind.get(e, "?") for e in es)
        macro_role[new_m] = "MACRO"
        macro_kind[new_m] = (
            f"direct_basin_block[n={len(es)},basins={len(basins)},"
            f"roles={'+'.join(f'{k}:{v}' for k, v in sorted(roles.items()))}]"
        )
        macro_basin[new_m] = -1
        macro_epoch[new_m] = float(np.mean(eps)) if eps else 0.0

    macro_causet = build_bitset_causet_robust(
        condensed_elements,
        condensed_children,
        event_role=macro_role,
        event_kind=macro_kind,
        event_basin=macro_basin,
        event_epoch=macro_epoch,
    )

    stats = {
        "coarse_grain_step": "direct_source_deu_basin_compression",
        "sterile_mode": sterile_mode,
        "selected_events_before_cap": selected_before_cap,
        "selected_events_after_cap": len(selected),
        "raw_blocks": M,
        "macro_events": len(macro_causet.events),
        "crossing_direct_edges": crossing_edges,
        "ep_lo": ep_lo,
        "ep_hi": ep_hi,
        "max_raw_events": max_raw_events,
        **scc_stats,
    }
    return macro_causet, stats


def make_broad_safe_rg2_causet_from_source(
    source,
    sterile_mode="singleton",
    ep_lo=None,
    ep_hi=None,
    max_raw_events=None,
    rg_epoch_width=1,
    rg_max_block_size=4,
    rg_signature_mode="degree",
    seed=0,
):
    """
    Build a broad-window/full-run safe-RG2 causet for multi-gap scaling.
    """
    basin_c, basin_stats = make_deu_basin_macro_causet_direct_from_source(
        source,
        sterile_mode=sterile_mode,
        ep_lo=ep_lo,
        ep_hi=ep_hi,
        max_raw_events=max_raw_events,
        seed=seed,
    )

    if "rg_coarse_grain_once_safe" in globals():
        rg2_c, rg2_stats = rg_coarse_grain_once_safe(
            basin_c,
            epoch_width=rg_epoch_width,
            max_block_size=rg_max_block_size,
            signature_mode=rg_signature_mode,
            relation_mode="direct",
            include_role=False,
        )
    else:
        rg2_c, rg2_stats = rg_coarse_grain_once(
            basin_c,
            epoch_width=rg_epoch_width,
            max_block_size=rg_max_block_size,
            relation_mode="direct",
            include_role=False,
        )

    stats = {
        "basin_events": len(basin_c.events),
        "rg2_events": len(rg2_c.events),
        "rg_epoch_width": rg_epoch_width,
        "rg_max_block_size": rg_max_block_size,
        "rg_signature_mode": rg_signature_mode,
    }
    stats.update({f"basin_{k}": v for k, v in basin_stats.items()})
    stats.update({f"rg2_{k}": v for k, v in rg2_stats.items()})
    return rg2_c, stats, basin_c


# ---------------------------------------------------------------------------
# Multi-gap causal-diamond sampling for real V~h^d and A~h^(d-1) tests
# ---------------------------------------------------------------------------

def _random_set_bit(bits, rng):
    bits = int(bits)
    n = bits.bit_count()
    if n <= 0:
        return None
    target = int(rng.integers(n))
    for k, i in enumerate(iter_bits(bits)):
        if k == target:
            return int(i)
    return None


def _epoch_range_bit_indexer(causet):
    eps = np.array([float(causet.event_epoch.get(e, 0.0)) for e in causet.events], dtype=float)
    order = np.argsort(eps, kind="mergesort")
    sorted_eps = eps[order]
    prefix = [0] * (len(order) + 1)
    bits = 0
    for k, idx in enumerate(order):
        bits |= bit_at(int(idx))
        prefix[k + 1] = bits

    def range_bits(lo, hi):
        left = int(np.searchsorted(sorted_eps, float(lo), side="left"))
        right = int(np.searchsorted(sorted_eps, float(hi), side="right"))
        if right <= left:
            return 0
        return int(prefix[right] & ~prefix[left])

    return eps, range_bits


def sample_diamonds_by_epoch_gaps(
    causet,
    gap_bins=((2, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 48), (48, 64)),
    n_endpoint_samples_per_bin=25000,
    keep_top_per_bin=200,
    min_size=2,
    seed=0,
):
    """
    Sample causal diamonds over a broad causet in several epoch-gap bins.

    This fixes the width-window problem: instead of fitting many diamonds whose
    heights are all 6--8, it intentionally builds a height ladder and then fits
    medians across gap bins.
    """
    rng = np.random.default_rng(seed)
    eps, range_bits = _epoch_range_bit_indexer(causet)
    ep_max = float(np.nanmax(eps)) if len(eps) else 0.0

    all_kept = []
    for gi, (gap_lo, gap_hi) in enumerate(gap_bins):
        gap_lo = float(gap_lo)
        gap_hi = float(gap_hi)
        if gap_hi < gap_lo:
            raise ValueError("gap bins must have gap_hi >= gap_lo")

        eligible_a = np.where(eps <= ep_max - gap_lo)[0]
        rows = []
        if len(eligible_a) == 0:
            continue

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

            rows.append({
                "gap_bin": gi,
                "gap_lo": gap_lo,
                "gap_hi": gap_hi,
                "gap_mid": 0.5 * (gap_lo + gap_hi),
                "a_index": a,
                "b_index": b,
                "a_event": causet.events[a],
                "b_event": causet.events[b],
                "ep_a": float(eps[a]),
                "ep_b": float(eps[b]),
                "epoch_gap": float(eps[b] - eps[a]),
                "interval_bits": strict_bits,
                "V_strict_prescan": V,
            })

        if rows:
            sub = pd.DataFrame(rows)
            sub = (
                sub.sort_values("V_strict_prescan", ascending=False)
                   .drop_duplicates(["a_index", "b_index"])
                   .head(keep_top_per_bin)
                   .copy()
            )
            all_kept.append(sub)

    if not all_kept:
        return pd.DataFrame()

    out = pd.concat(all_kept, ignore_index=True)
    out = out.sort_values(["gap_bin", "V_strict_prescan"], ascending=[True, False]).reset_index(drop=True)
    return out


def scaling_table_for_gap_diamonds(causet, diamond_df, max_diamonds=None):
    """
    Run the existing diamond_scaling_table and reattach gap-bin metadata.
    """
    if diamond_df is None or len(diamond_df) == 0:
        return pd.DataFrame()

    scaling = diamond_scaling_table(causet, diamond_df, max_diamonds=max_diamonds)
    if len(scaling) == 0:
        return scaling

    meta_cols = [
        "a_index", "b_index", "gap_bin", "gap_lo", "gap_hi", "gap_mid", "V_strict_prescan",
    ]
    meta = diamond_df[meta_cols].drop_duplicates(["a_index", "b_index"])
    scaling = scaling.merge(meta, on=["a_index", "b_index"], how="left")
    return scaling


def summarize_multigap_scaling(scaling_df, target_dim=3.0):
    """
    Fit volume and antichain scaling from bin medians over a real height ladder.
    Returns (summary_dict, binned_dataframe).
    """
    if scaling_df is None or len(scaling_df) == 0:
        return {"n_diamonds": 0, "status": "no diamonds"}, pd.DataFrame()

    rows = []
    for gap_bin, sub in scaling_df.groupby("gap_bin"):
        h = sub["h_inclusive"].to_numpy(dtype=float)
        V = sub["V_strict"].to_numpy(dtype=float)
        Amid = sub["mid_layer_size"].to_numpy(dtype=float)
        Amax = sub["max_layer_size"].to_numpy(dtype=float)
        rows.append({
            "gap_bin": int(gap_bin),
            "gap_lo": float(np.nanmedian(sub["gap_lo"])),
            "gap_hi": float(np.nanmedian(sub["gap_hi"])),
            "gap_mid": float(np.nanmedian(sub["gap_mid"])),
            "n": int(len(sub)),
            "h_med": float(np.nanmedian(h)),
            "h_q25": float(np.nanquantile(h, 0.25)),
            "h_q75": float(np.nanquantile(h, 0.75)),
            "h_min": int(np.nanmin(h)),
            "h_max": int(np.nanmax(h)),
            "V_med": float(np.nanmedian(V)),
            "V_q25": float(np.nanquantile(V, 0.25)),
            "V_q75": float(np.nanquantile(V, 0.75)),
            "V_max": int(np.nanmax(V)),
            "mid_layer_med": float(np.nanmedian(Amid)),
            "mid_layer_q75": float(np.nanquantile(Amid, 0.75)),
            "max_layer_med": float(np.nanmedian(Amax)),
            "MM_med": float(np.nanmedian(sub["MM_dim_strict"])),
            "r_med": float(np.nanmedian(sub["r_strict"])),
        })

    binned = pd.DataFrame(rows).sort_values("gap_mid").reset_index(drop=True)
    fit_mask = (
        np.isfinite(binned["h_med"]) & np.isfinite(binned["V_med"]) &
        (binned["h_med"] > 1) & (binned["V_med"] > 0) & (binned["n"] >= 3)
    )

    vol_fit_bin = loglog_fit(binned.loc[fit_mask, "h_med"], binned.loc[fit_mask, "V_med"], min_points=3)
    mid_fit_bin = loglog_fit(binned.loc[fit_mask, "h_med"], binned.loc[fit_mask, "mid_layer_med"], min_points=3)
    max_fit_bin = loglog_fit(binned.loc[fit_mask, "h_med"], binned.loc[fit_mask, "max_layer_med"], min_points=3)

    # Individual-diamond fits are secondary because top-diamond selection can bias them.
    h_all = scaling_df["h_inclusive"].to_numpy(dtype=float)
    V_all = scaling_df["V_strict"].to_numpy(dtype=float)
    Amid_all = scaling_df["mid_layer_size"].to_numpy(dtype=float)
    Amax_all = scaling_df["max_layer_size"].to_numpy(dtype=float)
    individual_mask = np.isfinite(h_all) & (h_all > 1) & np.isfinite(V_all) & (V_all > 0)
    vol_fit_ind = loglog_fit(h_all[individual_mask], V_all[individual_mask], min_points=6)
    mid_fit_ind = loglog_fit(h_all[individual_mask], Amid_all[individual_mask], min_points=6)
    max_fit_ind = loglog_fit(h_all[individual_mask], Amax_all[individual_mask], min_points=6)

    unique_heights = np.unique(h_all[np.isfinite(h_all)])
    height_lever_ok = (len(unique_heights) >= 5) and (np.nanmax(h_all) >= 2.0 * max(1.0, np.nanmin(h_all)))

    summary = {
        "n_diamonds": int(len(scaling_df)),
        "n_gap_bins": int(len(binned)),
        "n_fit_bins": int(fit_mask.sum()),
        "height_unique": int(len(unique_heights)),
        "h_min": int(np.nanmin(h_all)) if len(h_all) else 0,
        "h_med": float(np.nanmedian(h_all)) if len(h_all) else np.nan,
        "h_max": int(np.nanmax(h_all)) if len(h_all) else 0,
        "height_lever_ok": bool(height_lever_ok),
        "MM_med_all": float(np.nanmedian(scaling_df["MM_dim_strict"])),
        "r_med_all": float(np.nanmedian(scaling_df["r_strict"])),
        "target_dim": float(target_dim),
        "target_r": float(r_of_d(target_dim)) if "r_of_d" in globals() else np.nan,
        "binned_volume_slope": vol_fit_bin["slope"],
        "binned_volume_r2": vol_fit_bin["r2"],
        "binned_mid_layer_slope": mid_fit_bin["slope"],
        "binned_mid_layer_r2": mid_fit_bin["r2"],
        "binned_max_layer_slope": max_fit_bin["slope"],
        "binned_max_layer_r2": max_fit_bin["r2"],
        "individual_volume_slope": vol_fit_ind["slope"],
        "individual_volume_r2": vol_fit_ind["r2"],
        "individual_mid_layer_slope": mid_fit_ind["slope"],
        "individual_mid_layer_r2": mid_fit_ind["r2"],
        "individual_max_layer_slope": max_fit_ind["slope"],
        "individual_max_layer_r2": max_fit_ind["r2"],
        "status": "usable height ladder" if height_lever_ok else "still height-starved",
    }
    return summary, binned


def run_broad_multigap_causal_scaling(
    source,
    ep_lo=None,
    ep_hi=None,
    sterile_mode="singleton",
    rg_epoch_width=1,
    rg_max_block_size=4,
    rg_signature_mode="degree",
    gap_bins=((2, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 48), (48, 64)),
    n_endpoint_samples_per_bin=25000,
    keep_top_per_bin=200,
    min_size=2,
    seed=0,
    target_dim=3.0,
):
    """
    End-to-end broad-run Experiment 5/6 pipeline.
    """
    rg2_c, rg_stats, basin_c = make_broad_safe_rg2_causet_from_source(
        source,
        sterile_mode=sterile_mode,
        ep_lo=ep_lo,
        ep_hi=ep_hi,
        rg_epoch_width=rg_epoch_width,
        rg_max_block_size=rg_max_block_size,
        rg_signature_mode=rg_signature_mode,
        seed=seed,
    )

    diamond_df = sample_diamonds_by_epoch_gaps(
        rg2_c,
        gap_bins=gap_bins,
        n_endpoint_samples_per_bin=n_endpoint_samples_per_bin,
        keep_top_per_bin=keep_top_per_bin,
        min_size=min_size,
        seed=seed + 1000,
    )

    scaling_df = scaling_table_for_gap_diamonds(rg2_c, diamond_df)
    summary, binned = summarize_multigap_scaling(scaling_df, target_dim=target_dim)
    summary.update(rg_stats)
    return summary, binned, scaling_df, diamond_df, rg2_c, basin_c


# ---------------------------------------------------------------------------
# Experiment 4b: spatial metric refinement with face-depth / area weighting
# ---------------------------------------------------------------------------

@dataclass
class FoamSpatialDepthSnapshot:
    epoch: int
    active_faces: set
    face_nodes: dict
    face_types: dict
    face_depth: dict
    face_neighbors: dict
    stats: dict


@dataclass
class SpatialDepthRun:
    stats: dict
    spatial_snapshots: dict
    epoch_log: pd.DataFrame


def grow_typed_foam_spatial_depth_only(
    target_basin_splits=10000,
    seed=0,
    scheduler="capped",
    max_epochs=None,
    max_splits_per_epoch=256,
    max_ticks_per_epoch=256,
    unbounded_ticks=True,
    snapshot_epochs=None,
    snapshot_every=None,
    record_initial=True,
    record_final=True,
):
    """
    Spatial-only replay of the same capped/unbounded typed triangular scheduler.

    It does not build causal events.  It records face_depth, where each 1->3 split
    assigns child_depth = parent_depth + 1.  If a split is interpreted as dividing
    triangle area by three, a face has area weight 3^-depth and linear scale
    3^(-depth/2).  This lets us test a refinement-aware spatial metric rather than
    treating every tiny descendant triangle as unit graph length.
    """
    if scheduler not in {"capped", "unbounded"}:
        raise ValueError("scheduler must be 'capped' or 'unbounded'")

    snapshot_epochs = None if snapshot_epochs is None else {int(e) for e in snapshot_epochs}
    rng = np.random.default_rng(seed)
    faces = {}
    face_types = {}
    face_depth = {}
    edge_to_faces = defaultdict(set)
    active = set()
    next_face = 0
    next_node = 6
    epoch = 0
    stats = Counter()
    epoch_log = []
    spatial_snapshots = {}

    def add_face(nodes, ftype, depth):
        nonlocal next_face
        fid = next_face
        next_face += 1
        nodes = frozenset(int(x) for x in nodes)
        faces[fid] = nodes
        face_types[fid] = ftype
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

    def get_neighbors(fid):
        ns = set()
        for e in itertools.combinations(sorted(faces[fid]), 2):
            ns |= edge_to_faces[frozenset(e)]
        ns.discard(fid)
        return ns

    def snapshot_raw():
        active0 = set(active)
        neigh0 = {fid: get_neighbors(fid) & active0 for fid in active0}
        return active0, dict(face_types), dict(face_depth), dict(faces), neigh0

    def record_snapshot(ep):
        active0, types0, depth0, faces0, neigh0 = snapshot_raw()
        spatial_snapshots[int(ep)] = FoamSpatialDepthSnapshot(
            epoch=int(ep),
            active_faces=active0,
            face_nodes=faces0,
            face_types=types0,
            face_depth=depth0,
            face_neighbors=neigh0,
            stats=dict(stats),
        )

    def should_record(ep):
        if snapshot_epochs is not None and int(ep) in snapshot_epochs:
            return True
        if snapshot_every is not None and snapshot_every > 0 and int(ep) % int(snapshot_every) == 0:
            return True
        return False

    def is_frustrated0(fid, types0, neigh0):
        if types0[fid] != "S":
            return False
        nts = {types0[n] for n in neigh0[fid]}
        return ("G" in nts) and ("I" not in nts)

    # Same open seed patch used by the causal generator.
    add_face((0, 1, 2), "S", 0)
    add_face((0, 1, 3), "G", 0)
    add_face((2, 4, 5), "I", 0)
    add_face((3, 4, 5), "S", 0)

    if record_initial or should_record(0):
        record_snapshot(0)

    if max_epochs is None:
        factor = 10 if scheduler == "unbounded" else 5
        max_epochs = max(10, factor * target_basin_splits)

    epoch = 1
    while stats["basin_splits"] < target_basin_splits and epoch <= max_epochs:
        active0, types0, depth0, faces0, neigh0 = snapshot_raw()
        frustrated = [fid for fid in active0 if is_frustrated0(fid, types0, neigh0)]
        frontier_size = len(frustrated)
        stats["frontier_max"] = max(stats.get("frontier_max", 0), frontier_size)

        if frustrated:
            rng.shuffle(frustrated)
            remaining = target_basin_splits - stats["basin_splits"]
            if scheduler == "unbounded":
                selected = list(frustrated)
            else:
                selected = frustrated[: min(frontier_size, max_splits_per_epoch, remaining)]

            actual_splits = 0
            for fid in selected:
                if fid not in active:
                    stats["selected_already_removed"] += 1
                    continue
                old_nodes = sorted(faces0[fid])
                old_depth = depth0[fid]
                a_node, b_node, c_node = old_nodes
                new_node = next_node
                next_node += 1
                remove_face(fid)
                add_face((new_node, a_node, b_node), "S", old_depth + 1)
                add_face((new_node, a_node, c_node), "I", old_depth + 1)
                add_face((new_node, b_node, c_node), "G", old_depth + 1)
                stats["basin_splits"] += 1
                actual_splits += 1

            stats["split_epochs"] += 1
            stats["max_splits_in_epoch"] = max(stats.get("max_splits_in_epoch", 0), actual_splits)
            epoch_log.append({
                "epoch": epoch,
                "kind": "split",
                "frontier_size": frontier_size,
                "actual_splits": actual_splits,
                "ticks": 0,
                "active_faces": len(active),
                "basin_splits_total": stats["basin_splits"],
            })
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
            if scheduler == "unbounded" and unbounded_ticks:
                selected = list(candidates)
            else:
                selected = candidates[: min(len(candidates), max_ticks_per_epoch)]

            actual_ticks = 0
            for fid in selected:
                if fid not in active:
                    continue
                old_type = face_types[fid]
                if old_type == "I":
                    face_types[fid] = "S"
                elif old_type == "G":
                    face_types[fid] = "I"
                else:
                    face_types[fid] = "G"
                stats["sterile_ticks"] += 1
                stats[f"sterile_{old_type}_to_{face_types[fid]}"] += 1
                actual_ticks += 1

            stats["tick_epochs"] += 1
            stats["max_ticks_in_epoch"] = max(stats.get("max_ticks_in_epoch", 0), actual_ticks)
            epoch_log.append({
                "epoch": epoch,
                "kind": "tick",
                "frontier_size": 0,
                "actual_splits": 0,
                "ticks": actual_ticks,
                "active_faces": len(active),
                "basin_splits_total": stats["basin_splits"],
            })
            if actual_ticks == 0:
                stats["sterile_starved"] += 1
                if should_record(epoch):
                    record_snapshot(epoch)
                break

        if should_record(epoch):
            record_snapshot(epoch)
        epoch += 1

    stats["epochs"] = epoch - 1
    stats["final_active_faces"] = len(active)
    stats["final_nodes"] = next_node
    stats["scheduler"] = scheduler
    if record_final and int(stats["epochs"]) not in spatial_snapshots:
        record_snapshot(int(stats["epochs"]))

    return SpatialDepthRun(
        stats=dict(stats),
        spatial_snapshots=dict(sorted(spatial_snapshots.items())),
        epoch_log=pd.DataFrame(epoch_log),
    )


def face_depth_summary(snapshot):
    depths = np.array(list(snapshot.face_depth.values()), dtype=float)
    if len(depths) == 0:
        return {"epoch": snapshot.epoch, "faces": 0}
    return {
        "epoch": snapshot.epoch,
        "faces": int(len(depths)),
        "depth_min": int(np.min(depths)),
        "depth_med": float(np.median(depths)),
        "depth_mean": float(np.mean(depths)),
        "depth_q90": float(np.quantile(depths, 0.90)),
        "depth_max": int(np.max(depths)),
        "area_sum_if_split_area_thirds": float(np.sum(np.power(3.0, -depths))),
    }


def _largest_component_nodes(adj):
    comps = connected_components_from_adj(adj)
    return comps[0] if comps else set()


def _weighted_face_adj(snapshot, component="largest", length_base=3.0 ** -0.5):
    adj0 = snapshot_face_adj(snapshot, component="all")
    if component == "largest":
        keep = _largest_component_nodes(adj0)
        adj0 = induced_adj(adj0, keep)
    depths = {f: int(snapshot.face_depth.get(f, 0)) for f in adj0}
    lengths = {f: float(length_base ** depths[f]) for f in adj0}
    areas = {f: float(3.0 ** (-depths[f])) for f in adj0}
    wadj = {}
    for f, ns in adj0.items():
        wadj[f] = [(g, 0.5 * (lengths[f] + lengths[g])) for g in ns if g in adj0]
    return wadj, areas, depths


def _dijkstra_limited(wadj, source, max_distance=np.inf):
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist.get(u, np.inf):
            continue
        if d > max_distance:
            continue
        for v, w in wadj.get(u, ()): 
            nd = d + float(w)
            if nd <= max_distance and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def estimate_weighted_spatial_dimension(
    snapshot,
    n_centers=32,
    seed=0,
    radius_grid=None,
    fit_min_volume_fraction=1e-4,
    fit_max_volume_fraction=0.25,
    component="largest",
):
    """
    Refinement-aware spatial dimension estimate.

    Edge length between adjacent faces is 0.5*(3^-depth_f/2 + 3^-depth_g/2),
    and ball volume is sum(3^-depth).  This is not a proof of a smooth embedding;
    it is a check of whether the 1->3 refinement metric rescues ordinary 2D
    area-radius scaling from the raw dual-graph's exponential growth.
    """
    rng = np.random.default_rng(seed)
    wadj, areas, depths = _weighted_face_adj(snapshot, component=component)
    nodes = list(wadj)
    n = len(nodes)
    if n == 0:
        return {"epoch": snapshot.epoch, "weighted_dim": np.nan, "component_size": 0}, pd.DataFrame()

    n_centers = int(min(max(1, n_centers), n))
    centers = list(rng.choice(np.array(nodes, dtype=object), size=n_centers, replace=False))
    total_area = float(sum(areas.values()))

    if radius_grid is None:
        positive_lengths = []
        for f in nodes[: min(n, 1000)]:
            positive_lengths.extend([w for _, w in wadj.get(f, ())])
        min_step = float(np.quantile(positive_lengths, 0.10)) if positive_lengths else 1.0
        # Total area is finite; use a grid that starts above the tiny-face scale and
        # stops before guaranteed saturation of the original seed patch.
        radius_grid = np.geomspace(max(min_step, 1e-12), 2.0, 28)
    radius_grid = np.asarray(radius_grid, dtype=float)

    volume_curves = []
    count_curves = []
    max_r = float(np.nanmax(radius_grid))
    for c in centers:
        dist = _dijkstra_limited(wadj, c, max_distance=max_r)
        ds = np.array(list(dist.values()), dtype=float)
        fs = list(dist.keys())
        area_vals = np.array([areas[f] for f in fs], dtype=float)
        order = np.argsort(ds)
        ds = ds[order]
        area_prefix = np.cumsum(area_vals[order])
        vols = []
        counts = []
        for R in radius_grid:
            k = int(np.searchsorted(ds, R, side="right"))
            counts.append(k)
            vols.append(float(area_prefix[k - 1]) if k > 0 else 0.0)
        volume_curves.append(vols)
        count_curves.append(counts)

    volume_curves = np.asarray(volume_curves, dtype=float)
    count_curves = np.asarray(count_curves, dtype=float)
    mean_volume = volume_curves.mean(axis=0)
    mean_count = count_curves.mean(axis=0)

    eligible = (
        (mean_volume > max(1e-15, fit_min_volume_fraction * total_area)) &
        (mean_volume < fit_max_volume_fraction * total_area) &
        (radius_grid > 0)
    )
    if eligible.sum() < 3:
        eligible = (mean_volume > 0) & (mean_volume < 0.75 * total_area) & (radius_grid > 0)

    fit = loglog_fit(radius_grid[eligible], mean_volume[eligible], min_points=3)
    exp_fit = semilog_growth_fit(radius_grid[eligible], mean_volume[eligible], min_points=3)

    curve = pd.DataFrame({
        "epoch": snapshot.epoch,
        "radius": radius_grid,
        "mean_weighted_area": mean_volume,
        "mean_face_count": mean_count,
        "fit_used": eligible,
        "total_area_largest_component": total_area,
    })

    summary = {
        "epoch": snapshot.epoch,
        "component_size": n,
        "weighted_dim": fit["slope"],
        "weighted_fit_r2": fit["r2"],
        "weighted_fit_n": fit["n_fit"],
        "weighted_exp_rate": exp_fit["rate"],
        "weighted_exp_r2": exp_fit["r2"],
        "total_weighted_area": total_area,
        "n_centers": n_centers,
        "depth_min": int(min(depths.values())) if depths else 0,
        "depth_med": float(np.median(list(depths.values()))) if depths else np.nan,
        "depth_max": int(max(depths.values())) if depths else 0,
        "fit_radius_min": float(np.min(radius_grid[eligible])) if eligible.sum() else np.nan,
        "fit_radius_max": float(np.max(radius_grid[eligible])) if eligible.sum() else np.nan,
    }
    return summary, curve


def audit_weighted_spatial_snapshots(depth_run, epochs=None, n_centers=32, seed=0):
    if epochs is None:
        epochs = sorted(depth_run.spatial_snapshots)
    rows = []
    curves = {}
    depth_rows = []
    for k, ep in enumerate(epochs):
        snap = depth_run.spatial_snapshots[int(ep)]
        depth_rows.append(face_depth_summary(snap))
        summary, curve = estimate_weighted_spatial_dimension(
            snap,
            n_centers=n_centers,
            seed=seed + 1000 * k + int(ep),
        )
        rows.append(summary)
        curves[int(ep)] = curve
    return pd.DataFrame(rows), pd.DataFrame(depth_rows), curves


