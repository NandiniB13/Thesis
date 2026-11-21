"""
Exact rectangle packer for rectilinear plots.

Problem: place axis-aligned rectangular rooms (each may be rotated) in a rectilinear plot so that
  * rooms lie inside the plot and have disjoint interiors,
  * two rooms touch (share a segment of positive length) exactly when their pair is required,
  * rooms in `interior_ids` do not touch the plot boundary at all (not even at a point),
  * every other room may or may not touch the boundary,
  * the leftover space is arbitrary (any shape, any number of pieces).
Positions are whole units: the grid step is min(1, gcd of the plot coordinates and room sides).

pack_cluster places rooms in free coordinates, each touching a room already placed, and keeps the set of
plot translations under which all placed rooms still fit (the plot is allowed to move).
It returns (found, placements, nodes); found is True / False / None (time limit).
With no time limit, False means that no valid layout exists.
"""
import time
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, Optional


class _Timeout(Exception):
    pass


def _lattice(polygon, sizes):
    vals = [abs(c) for p in polygon for c in p] + [s for wh in sizes.values() for s in wh]
    if any(v > 1000 for v in vals) or any(s <= 0 for wh in sizes.values() for s in wh):
        raise ValueError("coordinates must be at most 1000 in absolute value and room sides positive")
    fr = []
    for v in [c for p in polygon for c in p] + [s for wh in sizes.values() for s in wh]:
        f = Fraction(v).limit_denominator(1000)
        if abs(float(f) - v) > 1e-11:
            raise ValueError("a coordinate is not within 1e-11 of a rational with denominator <= 1000")
        fr.append(f)
    den = 1
    for f in fr:
        den = den * f.denominator // gcd(den, f.denominator)
    nums = [int(f * den) for f in fr]
    g = 0
    for n in nums:
        g = gcd(g, abs(n))
    return Fraction(g, den), den


class _Budget(Exception):
    pass


def _cluster_search(polygon, rects: Dict[int, tuple], required_adjacency=None, *,
                    interior_ids: Iterable[int] = (), t_end=None, rule="mrv", node_budget=None,
                    on_solution=None, per_arrangement=4):
    """One complete cluster-first search. rule = "mrv": next room = fewest legal positions;
    rule = "area": fewest legal positions per unit of the room's area. Returns (found, placements, nodes)
    with found True / False / None (time ran out) / "budget" (node_budget used up, nothing decided)."""
    nodes = [0]
    ids = list(rects)
    interior = set(interior_ids)
    partners = {i: set() for i in ids}
    for a, b in (required_adjacency or ()):
        if a == b or a not in partners or b not in partners:
            raise ValueError("bad required pair")
        partners[a].add(b)
        partners[b].add(a)

    g, den = _lattice(polygon, rects)
    g = min(g, Fraction(1))
    sc = lambda v: int(round(Fraction(v).limit_denominator(1000) / g))
    poly = [(sc(x), sc(y)) for x, y in polygon]
    x0 = min(p[0] for p in poly)
    y0 = min(p[1] for p in poly)
    poly = [(x - x0, y - y0) for x, y in poly]
    size = {i: (sc(rects[i][0]), sc(rects[i][1])) for i in ids}
    W = max(p[0] for p in poly)
    H = max(p[1] for p in poly)
    n = len(poly)
    for k in range(n):
        (ax, ay), (bx, by) = poly[k], poly[(k + 1) % n]
        if ax != bx and ay != by:
            raise ValueError("the plot must be rectilinear")

    # ---- cells of the plot (stride W) -----------------------------------------------------------
    inside = 0
    for j in range(H):
        yc2 = 2 * j + 1
        cross = sorted(poly[k][0] for k in range(n)
                       if poly[k][0] == poly[(k + 1) % n][0]
                       and 2 * min(poly[k][1], poly[(k + 1) % n][1]) < yc2 < 2 * max(poly[k][1], poly[(k + 1) % n][1]))
        for a in range(0, len(cross) - 1, 2):
            for i in range(cross[a], cross[a + 1]):
                inside |= 1 << (j * W + i)
    if sum(size[i][0] * size[i][1] for i in ids) > bin(inside).count("1"):
        return False, None, 0
    ones = [0] * (H + 3)
    for h in range(1, H + 3):
        ones[h] = ones[h - 1] | (1 << (W * (h - 1)))
    rmask = lambda x, y, w, h: (((1 << w) - 1) * ones[h]) << (y * W + x)

    # ---- translation masks ------------------------------------------------------------------------
    S = 2 * W + 1                               # stride of the translation grid; u = plot position of the seed room,
    full = 0                                    # so only ux in [0, W), uy in [0, H) matter (columns >= W absorb the wrap)
    for uy in range(H):
        full |= ((1 << W) - 1) << (uy * S)
    Vcache = {}

    def V(w, h, inner):
        """bit (a + b*S) set when a w x h room at (a, b) of the plot is allowed (inner: strictly inside)"""
        key = (w, h, inner)
        if key not in Vcache:
            m = 0
            for b in range(H - h + 1):
                for a in range(W - w + 1):
                    if inner:
                        if a < 1 or b < 1 or a + w + 1 > W or b + h + 1 > H:
                            continue
                        r = rmask(a - 1, b - 1, w + 2, h + 2)
                    else:
                        r = rmask(a, b, w, h)
                    if inside & r == r:
                        m |= 1 << (a + b * S)
            Vcache[key] = m
        return Vcache[key]

    def Tof(T, r, i):
        x, y, w, h = r
        s = x + y * S
        v = V(w, h, i in interior)
        return T & ((v >> s) if s >= 0 else (v << -s))

    def touch(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        if ax + aw == bx or bx + bw == ax:
            return min(ay + ah, by + bh) - max(ay, by) > 0
        if ay + ah == by or by + bh == ay:
            return min(ax + aw, bx + bw) - max(ax, bx) > 0
        return False

    def overlap(a, b):
        return (min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0]) and
                min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1]))

    placed: Dict[int, tuple] = {}
    orient = {i: ([size[i]] if size[i][0] == size[i][1] else [size[i], (size[i][1], size[i][0])]) for i in ids}

    def legal(i, r, T):
        """new translation set if room i may go at r, else 0"""
        if abs(r[0]) > W or abs(r[1]) > H:
            return 0
        pi = partners[i]
        rx, ry, rw, rh = r
        rx2 = rx + rw
        ry2 = ry + rh
        for j, (qx, qy, qw, qh) in placed.items():
            qx2 = qx + qw
            qy2 = qy + qh
            ox = rx2 > qx and qx2 > rx          # positive overlap of the x ranges
            oy = ry2 > qy and qy2 > ry
            if ox and oy:
                return 0                        # the rooms overlap
            if ((oy and (rx2 == qx or qx2 == rx)) or (ox and (ry2 == qy or qy2 == ry))) != (j in pi):
                return 0                        # touching (shared wall of positive length) must equal "required"
        return Tof(T, r, i)

    def candidates(i, aj):
        out = set()
        if aj is not None:
            px, py, pw, ph = placed[aj]
            for (w, h) in orient[i]:
                for yy in range(py - h + 1, py + ph):
                    out.add((px + pw, yy, w, h)); out.add((px - w, yy, w, h))
                for xx in range(px - w + 1, px + pw):
                    out.add((xx, py + ph, w, h)); out.add((xx, py - h, w, h))
        else:
            for (w, h) in orient[i]:
                for xx in range(-W, W + 1):
                    for yy in range(-H, H + 1):
                        out.add((xx, yy, w, h))
        return sorted(out)

    def rec(T, prev=None, pr=None):
        # prev: room -> (anchor, options) computed at the parent; pr = (room, position) just placed.
        # An option list stays valid for the child after one more check against the new room and one AND with T.
        nodes[0] += 1
        if node_budget is not None and nodes[0] > node_budget:
            raise _Budget
        if t_end is not None and time.perf_counter() > t_end:
            raise _Timeout
        rest = [i for i in ids if i not in placed]
        if not rest:
            if on_solution is None:
                return T
            # collecting mode: report this arrangement at (up to per_arrangement of) its plot positions, keep searching
            idxs = []
            t = T
            while t:
                low = t & -t
                idxs.append(low.bit_length() - 1)
                t ^= low
            if len(idxs) > per_arrangement:
                step = (len(idxs) - 1) / max(per_arrangement - 1, 1)
                idxs = [idxs[round(k * step)] for k in range(per_arrangement)] if per_arrangement > 1 else idxs[:1]
            for idx in idxs:
                ux, uy = idx % S, idx // S
                lay = {i: (float((placed[i][0] + ux + x0) * g), float((placed[i][1] + uy + y0) * g),
                           float(placed[i][2] * g), float(placed[i][3] * g)) for i in ids}
                if on_solution(lay):
                    raise _Budget
            return 0
        front = [i for i in rest if any(j in placed for j in partners[i])]
        pool = front if front else [max(rest, key=lambda i: size[i][0] * size[i][1])]
        # next room: every room touching the cluster is checked, and one with no legal position cuts the branch
        cand_list = []
        for i in pool:
            aj = next((j for j in partners[i] if j in placed), None)
            pc = prev.get(i) if prev else None
            if pc is not None and aj is not None and pc[0] == aj:
                p, (qx, qy, qw, qh) = pr
                pin = p in partners[i]
                qx2 = qx + qw
                qy2 = qy + qh
                opts = []
                for r, T2 in pc[1]:
                    rx, ry, rw, rh = r
                    rx2 = rx + rw
                    ry2 = ry + rh
                    ox = rx2 > qx and qx2 > rx
                    oy = ry2 > qy and qy2 > ry
                    if ox and oy:
                        continue
                    if ((oy and (rx2 == qx or qx2 == rx)) or (ox and (ry2 == qy or qy2 == ry))) != pin:
                        continue
                    T3 = T2 & T
                    if T3:
                        opts.append((r, T3))
            else:
                opts = []
                for r in candidates(i, aj):
                    T2 = legal(i, r, T)
                    if T2:
                        opts.append((r, T2))
            if not opts:
                return 0
            cand_list.append((i, opts, aj))
            if rule == "mrv" and len(opts) == 1:
                break
        if rule == "mrv":
            i, opts, _a = min(cand_list, key=lambda t: len(t[1]))
        else:
            i, opts, _a = min(cand_list, key=lambda t: len(t[1]) / (size[t[0]][0] * size[t[0]][1]))
        child_prev = {k: (a, o) for (k, o, a) in cand_list if k != i}
        for r, T2 in opts:
            placed[i] = r
            res = rec(T2, child_prev, (i, r))
            if res:
                return res
            del placed[i]
        return 0

    # seed: most required neighbours; ties go to the smallest area (any room works as the seed, this only affects speed)
    first = max(ids, key=lambda i: (len(partners[i]), -size[i][0] * size[i][1]))
    final = 0
    try:
        for (w, h) in orient[first]:
            r = (0, 0, w, h)
            T = Tof(full, r, first)
            if not T:
                continue
            placed[first] = r
            final = rec(T)
            if final:
                break
            del placed[first]
    except _Timeout:
        return None, None, nodes[0]
    except _Budget:
        return "budget", None, nodes[0]
    if not final:
        return False, None, nodes[0]
    idx = (final & -final).bit_length() - 1
    ux, uy = idx % S, idx // S
    return True, {i: (float((placed[i][0] + ux + x0) * g), float((placed[i][1] + uy + y0) * g),
                      float(placed[i][2] * g), float(placed[i][3] * g)) for i in ids}, nodes[0]


def _worker(rule, polygon, rects, required_adjacency, interior_ids, remaining, q):
    """child process: one full search with the given rule, result put on the queue"""
    t_end = None if remaining is None else time.perf_counter() + remaining
    found, sol, n = _cluster_search(polygon, rects, required_adjacency, interior_ids=interior_ids,
                                    t_end=t_end, rule=rule)
    q.put((rule, found, sol, n))


def _pack_cluster_parallel(polygon, rects, required_adjacency, interior_ids, t_end):
    """both rules at once in two processes; the first one to decide wins and the other is stopped"""
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context()
    q = ctx.Queue()
    remaining = None if t_end is None else max(0.0, t_end - time.perf_counter())
    procs = [ctx.Process(target=_worker, daemon=True,
                         args=(rule, polygon, rects, required_adjacency, list(interior_ids), remaining, q))
             for rule in ("mrv", "area")]
    for p in procs:
        p.start()
    got = 0
    nodes = 0
    try:
        while got < 2:
            wait = None if t_end is None else max(0.05, t_end - time.perf_counter() + 3.0)
            try:
                rule, found, sol, n = q.get(timeout=wait)
            except _queue.Empty:
                return None, None, nodes
            got += 1
            nodes += n
            if found is not None:
                return found, sol, nodes
        return None, None, nodes
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2)


def _pack_cluster_sequential(polygon, rects, required_adjacency, interior_ids, t_end, first_budget):
    """one core: both rules in turn with doubling node budgets (restarts); costs several times the better rule"""
    total = 0
    budget = first_budget
    while True:
        for rule in ("mrv", "area"):
            found, sol, nodes = _cluster_search(polygon, rects, required_adjacency, interior_ids=interior_ids,
                                                t_end=t_end, rule=rule, node_budget=budget)
            total += nodes
            if found != "budget":
                return found, sol, total
        budget *= 2


def find_layouts(polygon, rects: Dict[int, tuple], required_adjacency=None, *,
                 interior_ids: Iterable[int] = (), max_layouts=100, time_limit=60.0, per_arrangement=4, rule="mrv"):
    """Collect several valid layouts instead of one. Returns (layouts, complete, nodes).
    Every distinct arrangement of the rooms (relative positions and orientations) is visited by the search; each
    arrangement is reported at up to `per_arrangement` positions of the plot (spread evenly over all positions where
    it fits; pass a huge number for all of them). complete is True when the search ran to the end (so no
    arrangement is missing), False when it stopped at max_layouts or at the time limit."""
    out = []

    def cb(lay):
        out.append(lay)
        return len(out) >= max_layouts

    found, _, nodes = _cluster_search(polygon, rects, required_adjacency, interior_ids=interior_ids,
                                      t_end=time.perf_counter() + time_limit, rule=rule,
                                      on_solution=cb, per_arrangement=per_arrangement)
    return out, found is False, nodes


def diverse_layouts(polygon, rects, required_adjacency=None, *, interior_ids=(), k=10, pool=1500,
                    time_limit=60.0, min_diff=3, rule="mrv"):
    """Up to k layouts that differ from each other as much as possible. The search collects a pool of up to `pool`
    layouts (one per arrangement; time_limit seconds at most), then picks greedily: first the first layout, then
    each time the layout whose smallest difference to the ones already picked is largest. The difference of two
    layouts is the number of rooms whose position relative to the room with id 0 is not the same (a shifted
    copy of the same arrangement counts as the same layout). A layout is only picked while it differs from every
    picked one in at least min_diff rooms, so fewer than k layouts can come back. Returns
    (chosen, pool_size, complete, nodes); complete is True when the pool holds every arrangement."""
    ids = sorted(rects)
    pooled = []

    def cb(lay):
        pooled.append(lay)
        return len(pooled) >= pool

    found, _, nodes = _cluster_search(polygon, rects, required_adjacency, interior_ids=interior_ids,
                                      t_end=time.perf_counter() + time_limit, rule=rule,
                                      on_solution=cb, per_arrangement=1)
    rel = [{i: (lay[i][0] - lay[ids[0]][0], lay[i][1] - lay[ids[0]][1], lay[i][2], lay[i][3]) for i in ids}
           for lay in pooled]

    def diff(a, b):
        return sum(1 for i in ids if a[i] != b[i])

    chosen = []
    if pooled:
        chosen = [0]
        best = [diff(rel[0], r) for r in rel]            # distance of each layout to the picked set
        while len(chosen) < k:
            j = max(range(len(rel)), key=lambda q: best[q])
            if best[j] < min_diff:
                break
            chosen.append(j)
            best = [min(best[q], diff(rel[j], rel[q])) for q in range(len(rel))]
    return [pooled[j] for j in chosen], len(pooled), found is False, nodes


def unused_regions(polygon, placements):
    """(number of separate unused pieces, number of those closed in by rooms, i.e. not touching the plot boundary)"""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    plot = Polygon(polygon)
    left = plot.difference(unary_union([box(x, y, x + w, y + h) for x, y, w, h in placements.values()]))
    parts = [gm for gm in (left.geoms if hasattr(left, "geoms") else [left])
             if gm.geom_type == "Polygon" and gm.area > 1e-9]
    return len(parts), sum(1 for gm in parts if not gm.intersects(plot.exterior))


def pack_cluster(polygon, rects: Dict[int, tuple], required_adjacency=None, *,
                 interior_ids: Iterable[int] = (), time_limit: Optional[float] = 60.0, first_budget: int = 3000):
    """Cluster-first solver. Step 1: the "fewest legal positions" rule alone for first_budget nodes, which
    settles most instances at once. Step 2 (hard instances): the same rule and an "fewest positions per unit
    of area" rule run at the same time in two processes, and the first search to finish decides the answer
    (a finished search is exhaustive, so "no solution" is a proper answer). On a one-core machine step 2
    falls back to running the two rules in turn. Returns (found, placements, nodes) with found True / False /
    None (time limit); nodes counts step 1 plus the winning search."""
    import os
    t_end = None if time_limit is None else time.perf_counter() + time_limit
    found, sol, n0 = _cluster_search(polygon, rects, required_adjacency, interior_ids=interior_ids,
                                     t_end=t_end, rule="mrv", node_budget=first_budget)
    if found != "budget":
        return found, sol, n0
    if (os.cpu_count() or 1) >= 2:
        found, sol, n1 = _pack_cluster_parallel(polygon, rects, required_adjacency, interior_ids, t_end)
    else:
        found, sol, n1 = _pack_cluster_sequential(polygon, rects, required_adjacency, interior_ids, t_end,
                                                  first_budget * 2)
    return found, sol, n0 + n1


# --------------------------------------------------------------------------
# Example plots
# --------------------------------------------------------------------------

# Each example may also carry an optional  labels={0: "Living", 1: "Kitchen", ...}
# (room id -> name). Labelled rooms are drawn with that name; others with their number.
EXAMPLES = {
    1: dict(
        polygon=[(0, 0), (24, 0), (24, 16), (16, 16), (16, 24), (0, 24)],
        rects=dict(enumerate([(24, 4), (8, 7), (8, 7), (13, 4), (7, 4), (4, 4), (16, 4), (16, 5), (7, 4), (8, 4), (1, 4)])),
        required_adjacency=[(0, 6), (0, 7), (1, 4), (1, 7), (2, 3), (2, 4), (3, 5), (3, 6), (5, 6),
                             (8, 3), (8, 6), (8, 7), (8, 0), (8, 9), (9, 3), (9, 2), (9, 4), (9, 7), (9, 10), (10, 4), (10, 1), (10, 7)],
        interior=[8, 9, 10],
    ),
    2: dict(
        polygon=[(0, 0), (35, 0), (35, 20), (22, 20), (22, 40), (0, 40)],
        rects=dict(enumerate([(8, 8), (13, 10), (14, 8), (17, 12), (12, 10), (5, 5), (5, 5),
                               (12, 10), (6, 5), (6, 5), (10, 10), (5, 7), (5, 5), (12, 10)])),
        required_adjacency=[(0, 2), (0, 12), (0, 3), (2, 3), (3, 12), (3, 11),
                            (12, 11), (3, 10), (11, 10), (10, 5), (10, 4), (5, 4), (5, 6), (6, 4), (4, 7), (7, 8),
                            (7, 9), (7, 1), (8, 9), (9, 1),
                            (13, 4), (13, 7), (13, 3), (13, 10), (13, 1)],
        interior=[],
    ),
    3: dict(
        polygon=[(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)],
        rects=dict(enumerate([(6, 8), (6, 6), (3, 7), (6, 4), (6, 2), (4, 6), (11, 4), (9, 2), (3, 8), (3, 3), (6, 4), (5, 5), (3, 9)])),
        required_adjacency=[(0, 3), (0, 5), (1, 2), (1, 4), (2, 6), (3, 4), (3, 5), (6, 8), (7, 8),
                             (9, 2), (9, 6), (9, 8), (9, 7), (9, 11), (10, 5), (10, 3), (10, 4), (10, 1), (10, 2), (10, 11), (11, 5), (11, 2), (11, 7), (11, 12), (12, 7)],
        interior=[9, 10],
    ),
    4: dict(
        polygon=[(0, 0), (20, 0), (20, 15), (10, 15), (10, 30), (0, 30)],
        rects=dict(enumerate([(6, 6), (11, 4), (1, 5), (2, 4), (9, 6), (6, 10), (5, 10), (9, 5), (10, 2), (4, 10), (5, 9), (3, 5), (4, 7)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0),
                             (9, 0), (9, 1), (9, 7), (9, 8), (9, 10), (9, 11), (10, 2), (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 1), (11, 2), (12, 3), (12, 4), (12, 5)],
        interior=[9, 10, 11, 12],
    ),
    5: dict(
        polygon=[(0, 0), (16, 0), (16, 6), (9, 6), (9, 14), (0, 14)],
        rects=dict(enumerate([(6, 4), (3, 3), (3, 4), (5, 3), (3, 8), (3, 6), (5, 2), (3, 4), (2, 3), (6, 2), (4, 5), (3, 1), (1, 3)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                             (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 7), (11, 8), (11, 9), (11, 12), (12, 3), (12, 9)],
        interior=[10, 11, 12],
    ),
    6: dict(
        polygon=[(0, 7), (0, 13), (20, 13), (20, 0), (13, 0), (13, 7)],
        rects=dict(enumerate([(7, 4), (7, 3), (2, 3), (4, 4), (2, 3), (2, 2), (2, 5), (2, 2), (2, 4), (4, 5), (3, 4), (3, 4), (11, 1), (3, 3), (1, 2)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9), (9, 10), (10, 11),
                             (12, 7), (12, 9), (12, 6), (12, 5), (12, 4), (12, 3), (12, 10), (12, 11), (12, 13), (13, 3), (13, 0), (13, 11), (13, 14), (14, 3), (14, 2), (14, 0)],
        interior=[12],
    ),
    7: dict(
        polygon=[(0, 0), (18, 0), (18, 13), (10, 13), (10, 24), (0, 24)],
        rects=dict(enumerate([(9, 6),(4, 4),(4, 4),(6, 4),(4, 4),(5, 6),(3, 4),(2, 4),(4, 4),(3, 4),(5, 5),(7, 7),(5, 3), (2, 7), (7, 2), (1, 3), (5, 2), (2, 5)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                             (13, 3), (13, 4), (13, 5), (13, 8), (13, 9), (13, 10), (13, 15), (14, 0), (14, 2), (14, 3), (14, 10), (14, 11), (14, 15), (14, 16), (15, 3), (15, 10), (16, 0), (16, 11), (16, 12), (16, 17), (17, 0), (17, 12)],
        interior=[14, 15, 16],
    ),
    8: dict(
        # the 18-room plan with an 8x8 atrium hole; interior rooms 8, 14, 15, 17
        polygon=[(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)],
        rects=dict(enumerate([(10, 10), (10, 10), (10, 5), (10, 5), (10, 5), (10, 5), (10, 10), (10, 10), (10, 5), (10, 5), (10, 5), (10, 5), (20, 6), (6, 14), (4, 6), (4, 6), (6, 6), (6, 8)])),
        required_adjacency=[(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 13), (4, 14), (5, 15), (5, 16), (6, 7), (6, 8), (6, 16), (6, 17), (7, 9), (8, 9), (8, 10), (8, 12), (8, 17), (9, 11), (10, 11), (10, 12), (12, 13), (12, 17), (13, 14), (14, 15), (15, 16), (16, 17)],
        interior=[],
    ),
    9: dict(   # 11 rooms, wall only 57% covered, 2 hole(s)
        polygon=[(0, 0), (12, 0), (12, 5), (6, 5), (6, 11), (0, 11)],
        rects={0: (3, 3), 1: (3, 4), 2: (2, 2), 3: (2, 3), 4: (4, 2), 5: (4, 3), 6: (2, 2), 7: (2, 3), 8: (2, 2), 9: (2, 3), 10: (3, 2)},
        required_adjacency=[(0, 3), (0, 6), (0, 9), (0, 10), (1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8), (3, 5), (3, 10), (4, 9), (4, 10), (5, 6), (6, 7), (7, 8)],
        interior=[],
    ),
    10: dict(   # two-connected, sparse, 15 rooms
        polygon=[(0, 0), (17, 0), (17, 13), (3, 13), (3, 16), (0, 16)],
        rects={0: (5, 2), 1: (3, 6), 2: (3, 6), 3: (3, 2), 4: (2, 2), 5: (5, 2), 6: (6, 4), 7: (5, 3), 8: (3, 2), 9: (2, 2), 10: (2, 5), 11: (4, 2), 12: (3, 5), 13: (2, 3), 14: (2, 4)},
        required_adjacency=[(0, 2), (0, 6), (0, 9), (1, 3), (1, 7), (1, 8), (1, 13), (1, 14), (2, 10), (3, 10), (3, 13), (4, 6), (4, 9), (4, 11), (5, 6), (5, 12), (5, 14), (6, 8), (6, 14), (7, 12), (8, 14), (9, 11), (12, 14)],
        interior=[],
    ),
    11: dict(   # two-connected, sparse, 12 rooms
        polygon=[(0, 0), (14, 0), (14, 7), (7, 7), (7, 15), (0, 15)],
        rects={0: (5, 5), 1: (6, 3), 2: (2, 2), 3: (4, 4), 4: (4, 4), 5: (2, 5), 6: (6, 2), 7: (2, 2), 8: (4, 2), 9: (2, 2), 10: (2, 2), 11: (2, 2)},
        required_adjacency=[(0, 3), (0, 5), (1, 4), (1, 6), (1, 7), (1, 8), (2, 9), (2, 11), (3, 5), (3, 10), (3, 11), (4, 7), (4, 8), (4, 9), (4, 10), (5, 11), (6, 7), (7, 10), (10, 11)],
        interior=[],
    ),
    12: dict(   # two-connected, dense, 12 rooms
        polygon=[(0, 0), (13, 0), (13, 4), (7, 4), (7, 13), (0, 13)],
        rects={0: (4, 3), 1: (2, 3), 2: (4, 2), 3: (3, 3), 4: (4, 2), 5: (2, 3), 6: (5, 2), 7: (2, 3), 8: (4, 5), 9: (4, 2), 10: (3, 2), 11: (2, 2)},
        required_adjacency=[(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (4, 6), (5, 6), (5, 7), (6, 8), (7, 8), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11)],
        interior=[],
    ),
    13: dict(   # two-connected, dense, 12 rooms
        polygon=[(0, 0), (11, 0), (11, 6), (7, 6), (7, 11), (0, 11)],
        rects={0: (5, 2), 1: (4, 2), 2: (2, 4), 3: (4, 3), 4: (3, 2), 5: (5, 2), 6: (4, 3), 7: (2, 4), 8: (2, 3), 9: (2, 2), 10: (2, 3), 11: (2, 2)},
        required_adjacency=[(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (1, 9), (2, 9), (2, 11), (3, 4), (3, 5), (3, 6), (4, 5), (4, 9), (5, 6), (5, 7), (5, 9), (5, 11), (6, 7), (6, 8), (6, 10), (7, 10), (8, 10)],
        interior=[4],
    ),
    14: dict(   # two-connected, dense, 14 rooms
        polygon=[(0, 0), (13, 0), (13, 8), (7, 8), (7, 11), (0, 11)],
        rects={0: (2, 5), 1: (2, 4), 2: (2, 2), 3: (3, 5), 4: (4, 3), 5: (3, 2), 6: (3, 5), 7: (2, 3), 8: (3, 2), 9: (5, 2), 10: (2, 2), 11: (3, 2), 12: (3, 2), 13: (2, 2)},
        required_adjacency=[(0, 1), (0, 6), (0, 7), (1, 2), (1, 6), (1, 10), (2, 3), (2, 10), (3, 4), (3, 5), (3, 8), (3, 9), (3, 10), (4, 5), (5, 9), (6, 7), (6, 8), (6, 10), (6, 11), (6, 12), (6, 13), (8, 9), (8, 13), (11, 12), (12, 13)],
        interior=[],
    ),
    15: dict(   # two-connected, sparse, 13 rooms
        polygon=[(0, 0), (18, 0), (18, 8), (7, 8), (7, 16), (0, 16)],
        rects={0: (3, 5), 1: (5, 4), 2: (6, 6), 3: (3, 2), 4: (3, 6), 5: (3, 5), 6: (2, 3), 7: (3, 5), 8: (6, 2), 9: (3, 2), 10: (2, 3), 11: (2, 2), 12: (2, 2)},
        required_adjacency=[(0, 1), (0, 4), (0, 10), (0, 11), (0, 12), (1, 7), (1, 8), (1, 10), (1, 11), (2, 3), (2, 6), (3, 5), (4, 6), (4, 9), (4, 12), (5, 9), (6, 12), (7, 8), (8, 10), (11, 12)],
        interior=[],
    ),
    16: dict(   # two-connected, dense, 13 rooms
        polygon=[(0, 0), (11, 0), (11, 10), (5, 10), (5, 13), (0, 13)],
        rects={0: (5, 2), 1: (2, 3), 2: (2, 4), 3: (3, 5), 4: (3, 3), 5: (5, 2), 6: (2, 4), 7: (3, 5), 8: (2, 5), 9: (3, 2), 10: (3, 4), 11: (2, 2), 12: (2, 2)},
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (2, 6), (3, 4), (3, 7), (3, 8), (4, 5), (4, 8), (4, 9), (5, 6), (5, 9), (5, 10), (7, 8), (7, 12), (8, 9), (8, 11), (8, 12), (9, 10), (9, 11)],
        interior=[],
    ),
    17: dict(   # two-connected, sparse, 13 rooms
        polygon=[(0, 0), (14, 0), (14, 4), (8, 4), (8, 15), (0, 15)],
        rects={0: (4, 6), 1: (6, 2), 2: (2, 4), 3: (6, 4), 4: (6, 2), 5: (4, 2), 6: (2, 4), 7: (2, 2), 8: (2, 2), 9: (2, 2), 10: (2, 3), 11: (3, 2), 12: (2, 3)},
        required_adjacency=[(0, 2), (0, 6), (0, 7), (0, 10), (0, 11), (0, 12), (1, 3), (1, 4), (2, 4), (2, 12), (3, 8), (4, 11), (5, 9), (5, 10), (6, 9), (6, 12), (7, 8), (8, 11)],
        interior=[],
    ),
    18: dict(   # two-connected, sparse, 12 rooms
        polygon=[(0, 0), (17, 0), (17, 9), (7, 9), (7, 14), (0, 14)],
        rects={0: (5, 5), 1: (3, 6), 2: (6, 3), 3: (5, 2), 4: (5, 5), 5: (5, 2), 6: (5, 4), 7: (2, 2), 8: (2, 2), 9: (2, 4), 10: (4, 2), 11: (3, 2)},
        required_adjacency=[(0, 1), (0, 4), (0, 7), (0, 9), (1, 3), (1, 5), (1, 9), (1, 11), (2, 4), (2, 6), (2, 8), (3, 11), (4, 6), (4, 7), (4, 10), (5, 7), (5, 8), (5, 11), (6, 10)],
        interior=[],
    ),
    19: dict(   # two-connected, dense, 12 rooms
        polygon=[(0, 0), (13, 0), (13, 7), (7, 7), (7, 11), (0, 11)],
        rects={0: (2, 4), 1: (5, 2), 2: (4, 2), 3: (4, 3), 4: (2, 3), 5: (3, 5), 6: (5, 4), 7: (4, 2), 8: (5, 2), 9: (2, 3), 10: (2, 2), 11: (2, 2)},
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (3, 5), (3, 6), (4, 7), (5, 7), (5, 9), (5, 10), (6, 8), (6, 9), (7, 10), (7, 11), (8, 9), (10, 11)],
        interior=[],
    ),
    20: dict(   # two-connected, sparse, 13 rooms
        polygon=[(0, 0), (14, 0), (14, 4), (11, 4), (11, 14), (0, 14)],
        rects={0: (3, 5), 1: (6, 6), 2: (2, 3), 3: (2, 6), 4: (6, 3), 5: (2, 2), 6: (2, 5), 7: (2, 2), 8: (2, 3), 9: (3, 2), 10: (4, 2), 11: (4, 2), 12: (2, 2)},
        required_adjacency=[(0, 1), (0, 4), (0, 9), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11), (1, 12), (2, 8), (3, 6), (3, 9), (4, 12), (5, 7), (5, 10), (6, 11), (7, 10), (7, 11), (8, 10)],
        interior=[],
    ),
    21: dict(   # two-connected, dense, 12 rooms
        polygon=[(0, 0), (11, 0), (11, 7), (8, 7), (8, 11), (0, 11)],
        rects={0: (2, 2), 1: (2, 5), 2: (2, 4), 3: (3, 2), 4: (5, 3), 5: (5, 2), 6: (3, 3), 7: (4, 3), 8: (4, 3), 9: (2, 3), 10: (2, 2), 11: (3, 2)},
        required_adjacency=[(0, 1), (0, 9), (1, 2), (1, 5), (1, 9), (2, 3), (2, 4), (3, 4), (3, 10), (4, 6), (4, 10), (4, 11), (5, 6), (5, 7), (5, 9), (6, 8), (6, 11), (7, 8)],
        interior=[],
    ),
    22: dict(   # two-connected, sparse, 14 rooms
        polygon=[(0, 0), (16, 0), (16, 10), (6, 10), (6, 14), (0, 14)],
        rects={0: (5, 2), 1: (5, 4), 2: (6, 3), 3: (2, 4), 4: (4, 4), 5: (2, 4), 6: (2, 2), 7: (4, 3), 8: (5, 2), 9: (2, 4), 10: (3, 2), 11: (2, 5), 12: (2, 2), 13: (2, 3)},
        required_adjacency=[(0, 8), (0, 10), (1, 8), (1, 10), (1, 11), (2, 3), (2, 6), (2, 10), (2, 12), (3, 9), (3, 12), (4, 5), (4, 7), (4, 9), (4, 13), (5, 13), (6, 7), (6, 9), (7, 9), (7, 11), (8, 10), (9, 13), (10, 11)],
        interior=[],
    ),
    23: dict(   # two-connected, dense, 13 rooms
        polygon=[(0, 0), (13, 0), (13, 8), (7, 8), (7, 13), (0, 13)],
        rects={0: (4, 2), 1: (2, 2), 2: (4, 4), 3: (5, 2), 4: (3, 4), 5: (3, 5), 6: (2, 4), 7: (3, 3), 8: (2, 3), 9: (2, 3), 10: (5, 3), 11: (2, 4), 12: (2, 2)},
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (2, 4), (2, 7), (2, 8), (3, 5), (3, 6), (4, 9), (5, 6), (5, 10), (6, 7), (6, 11), (7, 8), (7, 11), (8, 9), (10, 11), (10, 12), (11, 12)],
        interior=[],
    ),
    24: dict(   # two-connected, dense, 13 rooms
        polygon=[(0, 0), (13, 0), (13, 7), (9, 7), (9, 13), (0, 13)],
        rects={0: (5, 2), 1: (4, 2), 2: (4, 3), 3: (2, 4), 4: (4, 3), 5: (5, 4), 6: (5, 2), 7: (4, 3), 8: (4, 2), 9: (4, 3), 10: (5, 2), 11: (2, 2), 12: (2, 2)},
        required_adjacency=[(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 5), (2, 11), (3, 4), (3, 6), (4, 5), (5, 7), (5, 11), (5, 12), (6, 7), (6, 8), (7, 10), (8, 9), (9, 10), (11, 12)],
        interior=[],
    ),
    25: dict(   # two-connected, dense, 14 rooms
        polygon=[(0, 0), (13, 0), (13, 9), (10, 9), (10, 13), (0, 13)],
        rects={0: (4, 4), 1: (5, 4), 2: (4, 2), 3: (5, 2), 4: (4, 2), 5: (4, 5), 6: (3, 2), 7: (4, 5), 8: (2, 3), 9: (4, 2), 10: (2, 4), 11: (4, 2), 12: (3, 2), 13: (2, 3)},
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (3, 7), (4, 6), (5, 6), (5, 8), (5, 9), (5, 10), (6, 8), (6, 13), (7, 9), (7, 11), (8, 10), (8, 13), (9, 10), (9, 12), (11, 12)],
        interior=[],
    ),
    26: dict(   # two-connected, sparse, 13 rooms
        polygon=[(0, 0), (14, 0), (14, 11), (6, 11), (6, 16), (0, 16)],
        rects={0: (3, 2), 1: (6, 6), 2: (2, 2), 3: (3, 2), 4: (2, 4), 5: (4, 5), 6: (2, 4), 7: (5, 3), 8: (5, 4), 9: (2, 5), 10: (3, 2), 11: (2, 2), 12: (2, 2)},
        required_adjacency=[(0, 4), (0, 12), (1, 2), (1, 3), (1, 4), (1, 11), (1, 12), (2, 8), (2, 10), (3, 6), (3, 11), (4, 7), (5, 9), (5, 10), (5, 12), (6, 7), (8, 10), (9, 12)],
        interior=[],
    ),
    27: dict(   # two-connected, sparse, 15 rooms
        polygon=[(0, 0), (16, 0), (16, 11), (13, 11), (13, 14), (0, 14)],
        rects={0: (6, 3), 1: (5, 2), 2: (2, 4), 3: (4, 4), 4: (3, 3), 5: (5, 4), 6: (3, 4), 7: (5, 2), 8: (2, 2), 9: (3, 4), 10: (3, 5), 11: (2, 2), 12: (5, 2), 13: (2, 2), 14: (2, 3)},
        required_adjacency=[(0, 6), (0, 7), (0, 13), (0, 14), (1, 4), (1, 9), (1, 11), (2, 3), (2, 14), (3, 8), (3, 10), (3, 14), (4, 9), (4, 13), (5, 7), (5, 13), (6, 10), (8, 12), (11, 12)],
        interior=[],
    ),
    28: dict(   # two-connected, sparse, 15 rooms
        polygon=[(0, 0), (15, 0), (15, 8), (8, 8), (8, 14), (0, 14)],
        rects={0: (3, 4), 1: (3, 6), 2: (6, 5), 3: (3, 2), 4: (2, 5), 5: (6, 2), 6: (2, 3), 7: (2, 4), 8: (2, 3), 9: (2, 2), 10: (2, 2), 11: (3, 3), 12: (2, 2), 13: (2, 2), 14: (2, 2)},
        required_adjacency=[(0, 1), (0, 5), (0, 11), (1, 7), (1, 9), (1, 11), (2, 3), (2, 5), (2, 6), (2, 10), (2, 12), (3, 8), (3, 12), (3, 14), (4, 6), (4, 7), (4, 10), (4, 13), (5, 8), (6, 13), (7, 9), (8, 12), (8, 14)],
        interior=[],
    ),
    29: dict(   # two-connected, sparse, 14 rooms
        polygon=[(0, 0), (16, 0), (16, 11), (9, 11), (9, 15), (0, 15)],
        rects={0: (4, 3), 1: (5, 2), 2: (2, 2), 3: (6, 5), 4: (5, 5), 5: (4, 4), 6: (2, 5), 7: (5, 2), 8: (5, 2), 9: (5, 2), 10: (3, 3), 11: (2, 2), 12: (3, 2), 13: (2, 4)},
        required_adjacency=[(0, 3), (0, 10), (1, 7), (1, 8), (1, 10), (2, 4), (2, 7), (3, 9), (3, 10), (3, 12), (3, 13), (4, 7), (4, 9), (5, 6), (5, 9), (6, 11), (6, 12), (8, 10), (11, 12), (12, 13)],
        interior=[],
    ),
    30: dict(   # two-connected, sparse, 15 rooms
        polygon=[(0, 0), (14, 0), (14, 9), (11, 9), (11, 17), (0, 17)],
        rects={0: (6, 4), 1: (6, 6), 2: (5, 3), 3: (6, 4), 4: (2, 3), 5: (2, 6), 6: (2, 2), 7: (3, 2), 8: (2, 5), 9: (2, 2), 10: (2, 3), 11: (2, 4), 12: (2, 3), 13: (3, 2), 14: (2, 2)},
        required_adjacency=[(0, 2), (0, 6), (0, 13), (1, 3), (1, 8), (1, 11), (1, 13), (2, 6), (2, 9), (2, 12), (3, 7), (3, 8), (3, 11), (4, 8), (4, 9), (5, 9), (5, 12), (7, 10), (8, 10), (8, 14), (10, 14)],
        interior=[],
    ),
    31: dict(   # two-connected, sparse, 15 rooms
        polygon=[(0, 0), (14, 0), (14, 11), (8, 11), (8, 14), (0, 14)],
        rects={0: (4, 6), 1: (5, 3), 2: (2, 2), 3: (6, 2), 4: (2, 3), 5: (3, 4), 6: (3, 3), 7: (3, 2), 8: (3, 3), 9: (2, 3), 10: (2, 3), 11: (2, 2), 12: (4, 2), 13: (2, 2), 14: (3, 2)},
        required_adjacency=[(0, 6), (0, 8), (0, 11), (0, 14), (1, 3), (1, 9), (1, 12), (2, 11), (2, 14), (3, 5), (3, 9), (4, 12), (4, 13), (5, 8), (6, 12), (6, 13), (7, 9), (7, 10), (7, 12), (8, 14), (9, 10)],
        interior=[],
    ),
    32: dict(   # two-connected, sparse, 14 rooms
        polygon=[(0, 0), (16, 0), (16, 10), (12, 10), (12, 15), (0, 15)],
        rects={0: (5, 5), 1: (3, 4), 2: (2, 6), 3: (4, 6), 4: (6, 3), 5: (4, 3), 6: (4, 6), 7: (2, 6), 8: (3, 2), 9: (3, 2), 10: (3, 2), 11: (2, 2), 12: (2, 2), 13: (3, 2)},
        required_adjacency=[(0, 8), (0, 10), (0, 12), (1, 3), (1, 8), (1, 10), (2, 4), (2, 5), (2, 6), (2, 13), (3, 9), (3, 11), (4, 5), (4, 6), (4, 7), (5, 7), (6, 13), (7, 12), (9, 11), (11, 13)],
        interior=[],
    ),
    33: dict(   # two-connected, sparse, 14 rooms
        polygon=[(0, 0), (16, 0), (16, 10), (12, 10), (12, 14), (0, 14)],
        rects={0: (2, 6), 1: (2, 5), 2: (4, 4), 3: (4, 6), 4: (6, 4), 5: (2, 2), 6: (5, 2), 7: (5, 4), 8: (4, 3), 9: (3, 2), 10: (3, 2), 11: (3, 2), 12: (2, 3), 13: (2, 2)},
        required_adjacency=[(0, 1), (0, 8), (0, 12), (0, 13), (1, 10), (1, 13), (2, 4), (2, 5), (2, 6), (2, 12), (3, 5), (3, 7), (3, 8), (3, 9), (4, 6), (4, 11), (5, 12), (6, 10), (7, 11), (8, 9)],
        interior=[],
    ),
    34: dict(   # house plan 1: 20 rooms, plot 60 x 66 with a 24 x 26 corner removed, two-connected
        polygon=[(24, 0), (60, 0), (60, 66), (0, 66), (0, 26), (24, 26)],
        rects={0: (12, 10), 1: (10, 5), 2: (10, 5), 3: (20, 10), 4: (12, 10), 5: (12, 10), 6: (30, 20), 7: (18, 10), 8: (18, 10), 9: (12, 10), 10: (12, 10), 11: (10, 10), 12: (18, 10), 13: (10, 5), 14: (10, 5), 15: (18, 10), 16: (18, 10), 17: (18, 6), 18: (18, 8), 19: (18, 14)},
        required_adjacency=[(0, 1), (0, 2), (0, 5), (0, 6), (1, 2), (1, 3), (2, 3), (2, 6), (3, 4), (3, 6), (3, 7), (4, 7), (5, 6), (5, 9), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (8, 12), (8, 13), (9, 10), (10, 11), (11, 12), (11, 15), (12, 13), (12, 14), (12, 15), (12, 18), (13, 14), (14, 18), (15, 16), (15, 18), (15, 19), (16, 17), (16, 19), (17, 19), (18, 19)],
        interior=[2, 6, 12],
        labels={0: 'Jill', 1: 'WC', 2: 'Bath', 3: 'Terrace', 4: 'Chef', 5: 'Jack', 6: 'Great room', 7: 'Internet / study', 8: 'Games', 9: 'Hall', 10: 'Utility', 11: 'Laundry', 12: 'Vestibule', 13: 'Powder', 14: 'Linen', 15: 'Loggia', 16: 'Courtyard', 17: 'Planter box', 18: 'Ensuite', 19: 'Boudoir'},
    ),
    35: dict(   # house plan 2: 22 rooms, 100 x 68 L plot (68 x 14 corner removed), two-connected
        polygon=[(0, 14), (68, 14), (68, 0), (100, 0), (100, 68), (0, 68)],
        rects={0: (17, 17), 1: (9, 11), 2: (15, 17), 3: (8, 17), 4: (10, 17), 5: (9, 11), 6: (7, 17), 7: (13, 17), 8: (8, 11), 9: (80, 6), 10: (8, 6), 11: (17, 17), 12: (6, 11), 13: (36, 26), 14: (9, 15), 15: (7, 5), 16: (13, 5), 17: (20, 12), 18: (9, 6), 19: (20, 9), 20: (20, 15), 21: (8, 33)},
        required_adjacency=[(0, 1), (0, 9), (1, 2), (1, 9), (2, 3), (2, 9), (3, 4), (3, 9), (4, 5), (4, 9), (5, 6), (5, 9), (6, 7), (6, 9), (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (10, 16), (10, 21), (11, 12), (12, 13), (13, 14), (13, 18), (14, 15), (14, 17), (14, 18), (15, 16), (15, 17), (16, 17), (16, 21), (17, 18), (17, 19), (17, 21), (18, 19), (19, 20), (19, 21), (20, 21)],
        interior=[14, 15, 16, 17],
        labels={0: 'Bedroom 3', 1: 'Porch 1', 2: 'Master bed', 3: 'Closet', 4: 'M. bath', 5: 'Porch 2', 6: 'Office', 7: 'Bedroom 1', 8: 'Porch 3', 9: 'Hall', 10: 'Foyer', 11: 'Bedroom 2', 12: 'Mech.', 13: 'Garage', 14: 'Mudroom', 15: 'Linen', 16: 'Pantry', 17: 'Kitchen', 18: 'Bath 4', 19: 'Dining', 20: 'Living', 21: 'Porch (east)'},
    ),
    36: dict(   # floor plan 1 (exact dimensions): 11 rooms, 72 x 64 L plot
        polygon=[(0, 64), (72, 64), (72, 0), (28, 0), (28, 19), (0, 19)],
        rects={0: (21, 14), 1: (21, 21), 2: (20, 14), 3: (15, 10), 4: (21, 6), 5: (14, 20), 6: (21, 15), 7: (10, 12), 8: (11, 12), 9: (14, 19), 10: (24, 24)},
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 4), (3, 6), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 10), (6, 7), (7, 8), (7, 9), (8, 9), (8, 10), (9, 10)],
        interior=[4, 8],
        labels={0: 'Bed 1', 1: 'Living', 2: 'Dining', 3: 'Bath 1', 4: 'Hallway', 5: 'Kitchen', 6: 'Bed 2', 7: 'Bath 2', 8: 'Utility', 9: 'Courtyard', 10: 'Garage'},
    ),
    37: dict(   # floor plan 2 (exact dimensions): 22 rooms, 67 x 62 L plot
        polygon=[(0, 62), (67, 62), (67, 14), (48, 14), (48, 0), (0, 0)],
        rects={0: (7, 16), 1: (14, 16), 2: (15, 16), 3: (7, 10), 4: (7, 8), 5: (11, 14), 6: (7, 8), 7: (7, 6), 8: (11, 4), 9: (7, 4), 10: (18, 7), 11: (12, 7), 12: (14, 16), 13: (17, 16), 14: (18, 15), 15: (12, 9), 16: (7, 8), 17: (9, 8), 18: (15, 8), 19: (16, 14), 20: (15, 10), 21: (11, 15)},
        required_adjacency=[(0, 1), (0, 12), (1, 2), (1, 12), (1, 13), (2, 3), (2, 4), (2, 11), (2, 13), (3, 4), (3, 5), (4, 5), (4, 8), (4, 11), (5, 6), (5, 7), (5, 8), (6, 7), (7, 9), (8, 9), (8, 10), (9, 10), (10, 11), (10, 14), (10, 15), (11, 13), (11, 14), (12, 13), (12, 16), (12, 17), (13, 14), (13, 17), (13, 18), (14, 15), (14, 18), (14, 21), (16, 17), (16, 19), (17, 18), (17, 19), (18, 20), (18, 21), (19, 20), (20, 21)],
        interior=[4, 8, 11, 13, 17, 18],
        labels={0: 'Outdoor Kitchen', 1: 'Covered Lanai', 2: 'Family Rm', 3: 'Bath 2', 4: 'W.I.C. (north)', 5: 'Bedroom 2', 6: 'Office', 7: 'P.R.', 8: 'Pantry', 9: 'Linen', 10: 'Wetbar', 11: 'Dinette', 12: 'Master', 13: 'Living Rm', 14: 'Kitchen', 15: 'Stairs', 16: 'Util.', 17: 'W.I.C. (south)', 18: 'Foyer', 19: 'Master Bath', 20: 'Entry', 21: 'Dining Rm'},
    ),
    38: dict(   # floor plan 3 (exact dimensions): 12 rooms, 74 x 75 L plot
        polygon=[(0, 75), (74, 75), (74, 10), (25, 10), (25, 0), (0, 0)],
        rects={0: (19, 17), 1: (12, 12), 2: (7, 12), 3: (19, 20), 4: (14, 10), 5: (19, 10), 6: (26, 11), 7: (12, 11), 8: (26, 30), 9: (19, 18), 10: (26, 11), 11: (23, 30)},
        required_adjacency=[(0, 1), (0, 2), (0, 6), (0, 8), (1, 2), (1, 3), (2, 3), (2, 8), (3, 4), (3, 8), (3, 10), (4, 5), (4, 10), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10), (8, 11), (9, 11), (10, 11)],
        interior=[2, 8],
        labels={0: 'Bedroom 1', 1: 'Bath', 2: 'Hallway', 3: 'Bedroom 2', 4: 'Utility', 5: 'Courtyard', 6: 'Study', 7: 'Store', 8: 'Liv-Din', 9: 'Kitchen', 10: 'Foyer', 11: 'Garage'},
    ),
    39: dict(   # floor plan 4 (exact dimensions): 16 rooms, 51 x 61 L plot
        polygon=[(0, 61), (51, 61), (51, 0), (18, 0), (18, 8), (0, 8)],
        rects={0: (7, 6), 1: (7, 6), 2: (11, 12), 3: (10, 7), 4: (4, 7), 5: (4, 27), 6: (14, 10), 7: (14, 10), 8: (18, 8), 9: (19, 14), 10: (19, 13), 11: (5, 8), 12: (7, 8), 13: (7, 17), 14: (7, 7), 15: (20, 20)},
        required_adjacency=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (2, 9), (3, 4), (3, 6), (4, 5), (4, 6), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 13), (6, 7), (7, 8), (8, 13), (8, 14), (9, 10), (10, 11), (10, 12), (10, 13), (11, 12), (11, 13), (11, 15), (12, 15), (13, 14), (13, 15), (14, 15)],
        interior=[4, 5, 11, 13],
        labels={0: 'WC 1', 1: 'Store', 2: 'Bedroom 1', 3: 'Bathroom', 4: 'WC 2', 5: 'Hall 1', 6: 'Bedroom 2', 7: 'Bedroom 3', 8: 'Bedroom 4', 9: 'Living Room', 10: 'Kitchen - Dining', 11: 'Boiler', 12: 'Pantry', 13: 'Hall 2', 14: 'Entry', 15: 'Garage'},
    ),
    40: dict(   # graph 1 (16.json): 7 rooms, 12 pairs, 30 x 40 plot; solvable
        polygon=[(0, 0), (30, 0), (30, 40), (0, 40)],
        rects={1: (16, 13), 2: (6, 11), 3: (11, 11), 4: (15, 15), 5: (13, 13), 6: (12, 9), 7: (13, 13)},
        required_adjacency=[(1, 2), (1, 3), (1, 7), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (5, 6), (6, 7)],
        interior=[3],
    ),
    41: dict(   # graph 2 (17.json): 8 rooms, 13 pairs, 25 x 30 plot; solvable
        polygon=[(0, 0), (25, 0), (25, 30), (0, 30)],
        rects={1: (5, 5), 2: (10, 10), 3: (6, 6), 4: (5, 10), 5: (8, 10), 6: (10, 10), 7: (10, 10), 8: (7, 7)},
        required_adjacency=[(1, 2), (1, 4), (2, 3), (2, 5), (2, 6), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8)],
        interior=[5],
    ),
    42: dict(   # graph 3 (18.json): 15 rooms, 28 pairs, 70 x 65 plot; solvable
        polygon=[(0, 0), (70, 0), (70, 65), (0, 65)],
        rects={1: (11, 12), 2: (7, 6), 3: (11, 12), 4: (11, 5), 5: (21, 34), 6: (11, 13), 7: (9, 13), 8: (20, 13), 9: (15, 26), 10: (17, 11), 11: (14, 26), 12: (14, 16), 13: (12, 14), 14: (9, 14), 15: (21, 16)},
        required_adjacency=[(1, 2), (1, 6), (1, 7), (2, 3), (2, 9), (3, 4), (3, 9), (3, 11), (4, 5), (4, 11), (5, 11), (5, 13), (5, 14), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 12), (11, 12), (11, 13), (12, 15), (13, 14), (13, 15), (14, 15)],
        interior=[7, 9, 11, 13],
    ),
    43: dict(   # graph 4 (19.json): 8 rooms, 12 pairs, 40 x 35 plot; solvable
        polygon=[(0, 0), (40, 0), (40, 35), (0, 35)],
        rects={1: (15, 5), 2: (11, 13), 3: (6, 8), 4: (11, 14), 5: (24, 11), 6: (4, 11), 7: (6, 5), 8: (6, 5)},
        required_adjacency=[(1, 2), (1, 3), (1, 4), (2, 3), (2, 8), (3, 4), (3, 7), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8)],
        interior=[3, 7],
    ),
    44: dict(   # graph 5 (20.json): 8 rooms, 13 pairs, 20 x 35 plot; NO SOLUTION (search finished)
        polygon=[(0, 0), (20, 0), (20, 35), (0, 35)],
        rects={1: (5, 5), 2: (10, 10), 3: (6, 6), 4: (5, 10), 5: (5, 10), 6: (10, 10), 7: (10, 10), 8: (7, 7)},
        required_adjacency=[(1, 2), (1, 4), (2, 3), (2, 5), (2, 6), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8)],
        interior=[5],
    ),
    45: dict(   # graph 6 (eg1_1.json): 8 rooms, 14 pairs, 30 x 40 plot; solvable
        polygon=[(0, 0), (30, 0), (30, 40), (0, 40)],
        rects={1: (14, 13), 2: (12, 11), 3: (9, 10), 4: (5, 7), 5: (4, 6), 6: (11, 12), 7: (5, 8), 8: (13, 12)},
        required_adjacency=[(1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 8), (7, 8)],
        interior=[4],
    ),
    46: dict(   # graph 7 (eg_1_2.json): 8 rooms, 14 pairs, 30 x 40 plot; solvable
        polygon=[(0, 0), (30, 0), (30, 40), (0, 40)],
        rects={1: (14, 13), 2: (12, 11), 3: (9, 10), 4: (5, 7), 5: (4, 6), 6: (11, 12), 7: (5, 8), 8: (13, 12)},
        required_adjacency=[(1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 8), (7, 8)],
        interior=[4],
    ),
    47: dict(   # graph 8 (eg1_3.json): 8 rooms, 14 pairs, 35 x 35 plot; NO SOLUTION (search finished)
        polygon=[(0, 0), (35, 0), (35, 35), (0, 35)],
        rects={1: (14, 13), 2: (12, 11), 3: (9, 10), 4: (5, 7), 5: (4, 6), 6: (11, 12), 7: (5, 8), 8: (13, 12)},
        required_adjacency=[(1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (4, 6), (6, 8), (7, 8)],
        interior=[3],
    ),
    48: dict(   # graph 9 (eg1_4.json): 8 rooms, 15 pairs, 40 x 35 plot; NO SOLUTION (search finished)
        polygon=[(0, 0), (40, 0), (40, 35), (0, 35)],
        rects={1: (14, 13), 2: (12, 11), 3: (9, 10), 4: (5, 7), 5: (4, 6), 6: (11, 12), 7: (5, 8), 8: (13, 12)},
        required_adjacency=[(1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (4, 6), (4, 8), (5, 6), (6, 8), (7, 8)],
        interior=[3, 4],
    ),
}


# --------------------------------------------------------------------------
# Interactive runner
# --------------------------------------------------------------------------
# In these examples `interior` lists the rooms that may NOT touch any wall. Every other room may or may
# not touch a wall, and the leftover space may have any shape (any number of holes).

def _verify(polygon, placements, rects, adjacency, interior):
    """Independent check with shapely (returns None if shapely is missing)."""
    try:
        from shapely.geometry import Polygon, box
    except ImportError:
        return None
    P = Polygon(polygon)
    bd = P.boundary
    bs = {i: box(x, y, x + w, y + h) for i, (x, y, w, h) in placements.items()}
    adj = {tuple(sorted(a)) for a in (adjacency or [])}
    outside = sum(1 for b in bs.values() if not P.buffer(1e-9).contains(b))
    ids = sorted(bs)
    overlaps = wrong_adj = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if bs[i].intersection(bs[j]).area > 1e-9:
                overlaps += 1
            if (bs[i].boundary.intersection(bs[j].boundary).length > 1e-9) != ((i, j) in adj):
                wrong_adj += 1
    bad_int = sum(1 for i in (interior or ()) if bs[i].intersects(bd))
    return overlaps, outside, wrong_adj, bad_int


def _draw_shapes(ax, polygon, placements, interior):
    from matplotlib.patches import Polygon as MPoly, Rectangle
    ax.add_patch(MPoly(polygon, closed=True, fill=False, ec="black", lw=2))
    for i, (x, y, w, h) in placements.items():
        ax.add_patch(Rectangle((x, y), w, h, fc="#f6c28b" if i in set(interior) else "#9ec5ee", ec="black"))
    xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
    ax.set_xlim(min(xs) - 1, max(xs) + 1); ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_aspect("equal"); ax.axis("off")


def _draw_labels(fig, ax, placements, labels, max_font=9.0):
    """Room name + width x height, shrunk and wrapped to fit inside each room (call after fig.canvas.draw())."""
    import textwrap
    renderer = fig.canvas.get_renderer()
    for i, (x, y, w, h) in placements.items():
        label = str(labels.get(i, i))
        dims = f"{w:g}\u00d7{h:g}"                  # width x height as placed
        (px0, py0), (px1, py1) = ax.transData.transform((x, y)), ax.transData.transform((x + w, y + h))
        bw, bh = abs(px1 - px0) * 0.9, abs(py1 - py0) * 0.85          # usable pixel box inside the room
        best = None
        for n in sorted({len(label)} | set(range(1, len(label) + 1))):
            text = "\n".join(textwrap.wrap(label, n, break_long_words=False) or [label]) + "\n" + dims
            if best is not None and text == best[1]:
                continue
            t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
            ext = t.get_window_extent(renderer)
            t.remove()
            f = min(max_font, 10.0 * min(bw / max(ext.width, 1e-6), bh / max(ext.height, 1e-6)))
            if best is None or f > best[0]:
                best = (f, text)
        ax.text(x + w / 2, y + h / 2, best[1], ha="center", va="center", fontsize=max(best[0], 2.0), linespacing=1.0)


def render(polygon, placements, save_path, interior=(), labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    side = 6 if len(placements) <= 12 else 9
    fig, ax = plt.subplots(figsize=(side, side))
    _draw_shapes(ax, polygon, placements, interior)
    fig.canvas.draw()                              # fixes the data -> pixel scale so the names can be fitted
    _draw_labels(fig, ax, placements, labels or {})
    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def render_many(polygon, layouts, save_path, interior=(), labels=None, titles=None):
    """All the given layouts in one picture (a grid, at most 3 per row)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(layouts)
    cols = min(3, n); rows = math.ceil(n / cols)
    side = 7 if len(layouts[0]) <= 12 else 9
    pw = max(p[0] for p in polygon) - min(p[0] for p in polygon) + 2
    ph = max(p[1] for p in polygon) - min(p[1] for p in polygon) + 2
    fig, axes = plt.subplots(rows, cols, figsize=(side * cols, side * rows * min(1.0, ph / pw + 0.12)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, lay in enumerate(layouts):
        _draw_shapes(axes.flat[k], polygon, lay, interior)
        if titles:
            axes.flat[k].set_title(titles[k], fontsize=14)
    fig.canvas.draw()
    for k, lay in enumerate(layouts):
        _draw_labels(fig, axes.flat[k], lay, labels or {}, max_font=11.0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close(fig)


def layout_distance(a, b):
    """How different two layouts are: the number of rooms whose position differs, measured relative to the
    best-matching reference room (so sliding a whole layout sideways does not count as a difference)."""
    ids = list(a)
    best = len(ids)
    for r in ids:
        ax_, ay_ = a[r][0], a[r][1]; bx_, by_ = b[r][0], b[r][1]
        d = sum(1 for i in ids if (a[i][0] - ax_, a[i][1] - ay_, a[i][2], a[i][3]) !=
                (b[i][0] - bx_, b[i][1] - by_, b[i][2], b[i][3]))
        if d < best:
            best = d
    return best


def pick_diverse(layouts, k=10, min_distance=None):
    """Up to k layouts that are far apart from each other (farthest-first). Stops early when every remaining
    layout is closer than min_distance (rooms in a different place; default a quarter of the rooms, at least 4) to one already chosen, so it may return fewer."""
    if not layouts:
        return []
    if min_distance is None:                 # a quarter of the rooms (at least 4) must be somewhere else
        min_distance = max(4, -(-len(layouts[0]) // 4))
    chosen = [0]
    dist = [layout_distance(layouts[0], l) for l in layouts]
    while len(chosen) < k:
        j = max(range(len(layouts)), key=lambda t: dist[t])
        if dist[j] < min_distance:
            break
        chosen.append(j)
        for t, l in enumerate(layouts):
            d = layout_distance(layouts[j], l)
            if d < dist[t]:
                dist[t] = d
    return [layouts[j] for j in chosen]


def render_text(polygon, placements, interior=()):
    """Plain-text picture (no matplotlib needed): one character per unit cell, top row first.
    Rooms are drawn with their id (0-9, then a-z, A-Z), '.' is empty space inside the plot."""
    if any(abs(v - round(v)) > 1e-9 for p in polygon for v in p) or \
       any(abs(v - round(v)) > 1e-9 for r in placements.values() for v in r):
        return "(text picture needs whole-unit coordinates)"
    glyph = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    W = int(round(max(p[0] for p in polygon))); H = int(round(max(p[1] for p in polygon)))
    n = len(polygon)

    def inside(cx, cy):
        c = False
        for k in range(n):
            (ax, ay), (bx, by) = polygon[k], polygon[(k + 1) % n]
            if ax == bx and min(ay, by) < cy < max(ay, by) and ax > cx:
                c = not c
        return c
    rows = []
    for j in range(H - 1, -1, -1):
        line = ""
        for i in range(W):
            ch = " "
            if inside(i + 0.5, j + 0.5):
                ch = "."
                for r, (x, y, w, h) in placements.items():
                    if x <= i < x + w and y <= j < y + h:
                        ch = glyph[r % len(glyph)]
            line += ch
        rows.append(line)
    return "\n".join(rows)


def _solve_and_report(label, polygon, rects, required_adjacency, time_limit, interior_ids=(), labels=None):
    print(f"\nRunning example {label}: {len(rects)} rects, {len(required_adjacency or [])} adjacencies")
    print(f"  polygon: {polygon}")
    print(f"  rects:   {rects}")
    if required_adjacency:
        print(f"  required_adjacency (PTPG): {sorted(required_adjacency)}")
    if interior_ids:
        print(f"  interior rects (touch no wall): {sorted(interior_ids)}")
    t0 = time.perf_counter()
    found, placements, nodes = pack_cluster(polygon, rects, required_adjacency,
                                            interior_ids=interior_ids, time_limit=time_limit)
    dt = time.perf_counter() - t0
    status = "FEASIBLE" if found else "INFEASIBLE" if found is False else "INCONCLUSIVE"
    print(f"\n  solve: {status}   nodes explored: {nodes}   time: {dt:.3f}s")
    if found:
        v = _verify(polygon, placements, rects, required_adjacency, interior_ids)
        if v:
            print(f"    overlaps={v[0]}  outside={v[1]}  wrong-adjacency pairs={v[2]}  interior rooms touching a wall={v[3]}")
        save_path = f"layout_ex_{label}.png"
        try:
            render(polygon, placements, save_path, interior_ids, labels)
            print(f"  saved to {save_path}")
        except BaseException as exc:      # a broken / slow matplotlib must not hide the result
            print(f"  (could not draw the picture: {type(exc).__name__}: {exc}; text picture instead)")
            print("\n" + render_text(polygon, placements, interior_ids) + "\n")
    elif found is False:
        print("  no solution exists under these constraints")
    else:
        print(f"  hit time_limit ({time_limit}s) -- inconclusive")
    return found, nodes, dt


def run_example(choice, time_limit=60.0, required_adjacency=None):
    """`required_adjacency=None` uses the example's own PTPG; pass an iterable (or []) to override."""
    ex = EXAMPLES[choice]
    adj = ex.get("required_adjacency") if required_adjacency is None else required_adjacency
    return _solve_and_report(str(choice), ex["polygon"], ex["rects"], adj, time_limit,
                             interior_ids=ex.get("interior", ()), labels=ex.get("labels"))[0]


def select_diverse(layouts, k):
    """Pick k layouts that differ from each other as much as possible (farthest-point choice). The distance of two
    layouts is the sum over rooms of how far the room's corner moved plus any change of the room's shape (rotation)."""
    if len(layouts) <= k:
        return list(layouts)
    def dist(p, q):
        return sum(abs(p[i][0] - q[i][0]) + abs(p[i][1] - q[i][1]) + abs(p[i][2] - q[i][2]) + abs(p[i][3] - q[i][3])
                   for i in p)
    chosen = [0]
    d = [dist(layouts[0], l) for l in layouts]
    while len(chosen) < k:
        nxt = max(range(len(layouts)), key=lambda i: d[i])
        chosen.append(nxt)
        d = [min(d[i], dist(layouts[nxt], layouts[i])) for i in range(len(layouts))]
    return [layouts[i] for i in chosen]


def _more_layouts(label, polygon, rects, adj, interior, labels=None):
    raw = input("\nFind more layouts? Enter how many (up to 10, Enter = no): ").strip()
    if not raw:
        return
    try:
        n = int(raw)
        if n < 1:
            raise ValueError
    except ValueError:
        print(f"Invalid number: {raw!r}")
        return
    n = min(n, 10)
    t0 = time.perf_counter()
    pool, complete, nodes = find_layouts(polygon, rects, adj, interior_ids=interior, max_layouts=300,
                                         time_limit=60.0, per_arrangement=1)
    chosen = pick_diverse(pool, k=n)
    dt = time.perf_counter() - t0
    print(f"  collected {len(pool)} layouts in {dt:.1f}s ({nodes} nodes; "
          + ("the search ran to the end" if complete else "stopped at the limit before the search ended")
          + f"); {len(chosen)} clearly different ones chosen")
    if not chosen:
        return
    titles = []
    for k, lay in enumerate(chosen):
        pcs, enc = unused_regions(polygon, lay)
        titles.append(f"Layout {k + 1}: {pcs} unused piece" + ("" if pcs == 1 else "s") + (f" ({enc} closed in)" if enc else ""))
    path = f"layouts_ex_{label}.png"
    try:
        render_many(polygon, chosen, path, interior, labels, titles)
        print(f"  saved all {len(chosen)} in one picture: {path}")
    except BaseException as exc:
        print(f"  (could not draw {path}: {type(exc).__name__}: {exc})")



def run_custom_example(time_limit=60.0):
    import ast
    print("\nEnter your own floorplan. Each value is a Python literal, e.g.:")
    print("  polygon: [(0,0), (10,0), (10,10), (0,10)]")
    print("  rects:   {0: (4,4), 1: (6,4), 2: (10,6)}")
    print("  required_adjacency: [(0,1), (1,2)]   (or [] for none)")
    print("  interior rects (touch no wall): [2]   (or [] for none)")
    try:
        polygon = ast.literal_eval(input("\npolygon: ").strip())
        rects = {int(k): tuple(v) for k, v in ast.literal_eval(input("rects: ").strip()).items()}
        raw = input("required_adjacency (blank for none): ").strip()
        adj = ast.literal_eval(raw) if raw else []
        raw = input("interior rects (blank for none): ").strip()
        interior = ast.literal_eval(raw) if raw else []
    except (ValueError, SyntaxError) as exc:
        print(f"Could not parse that: {exc}")
        return
    found = _solve_and_report("custom", polygon, rects, adj, time_limit, interior_ids=interior)[0]
    if found:
        _more_layouts("custom", polygon, rects, adj, interior)


def main():
    print("Available examples:")
    for key, ex in EXAMPLES.items():
        print(f"  {key}. {len(ex['rects'])} rects, {len(ex.get('required_adjacency') or [])} adjacencies")
    print("  a. run all examples one by one")
    print("  c. enter your own custom example")
    raw = input(f"\nChoose an example [{min(EXAMPLES)}-{max(EXAMPLES)}], 'a' for all or 'c': ").strip()
    if raw.lower() == "a":
        rows = []
        for key, ex in EXAMPLES.items():
            ex_ = ex
            res = _solve_and_report(str(key), ex_["polygon"], ex_["rects"], ex_.get("required_adjacency"), 60.0,
                                    interior_ids=ex_.get("interior", ()), labels=ex_.get("labels"))
            rows.append((key, len(ex_["rects"]), *res))
        print("\nSummary")
        print("  ex  rooms  result        nodes     time")
        for key, n, found, nodes, dt in rows:
            st = "FEASIBLE" if found else "INFEASIBLE" if found is False else "INCONCLUSIVE"
            print(f"  {key:>2}  {n:>5}  {st:<12} {nodes:>7}  {dt:7.2f}s")
        print(f"  total time {sum(r[4] for r in rows):.1f}s")
        return
    if raw.lower() == "c":
        run_custom_example()
        return
    try:
        choice = int(raw)
        if choice not in EXAMPLES:
            raise ValueError
    except ValueError:
        print(f"Invalid choice: {raw!r}")
        return
    found = run_example(choice)
    if found:
        ex = EXAMPLES[choice]
        _more_layouts(str(choice), ex["polygon"], ex["rects"], ex.get("required_adjacency"),
                      ex.get("interior", ()), ex.get("labels"))


if __name__ == "__main__":
    main()