"""
Exact backtracking packer: places rectangles inside a rectangular or L-shaped plot (a
rectilinear polygon with at most one reflex vertex) with no overlaps and no rectangle crossing the
boundary.

`pack_rectangles_frontier` is the sole solver: a frontier search built around
`required_adjacency` (a set of (id_a, id_b) pairs over the same keys as `rects`: which
rooms must touch; any two rooms that touch must be such a pair). It grows outward from
placed rectangles using that graph to decide what to try next.

The leftover (plot minus rooms) is empty or one connected interior piece whose closure
misses the plot boundary altogether. Every non-interior room meets the boundary (a point
is enough); `interior_ids` rooms do not meet it at all. Coordinates are on one lattice.
The completeness proof is the "Completeness proof" doc: with time_limit=None a False
result means no valid layout exists.
"""
# ---------------------------------------------------------------------------
# interior_nosh.py -- interior_narrow.py with shapely replaced by integer bitmasks.
# Free space is one Python int, one bit per unit cell of the lattice grid (step g);
# fit / difference / components / vertices / boundary contact are exact bit operations.
# Search logic is identical to interior_narrow.py.
# ---------------------------------------------------------------------------
import ast
import time
from typing import Dict, Iterable, List, Optional, Tuple




# --------------------------------------------------------------------------
# Bitmask grid (replaces shapely)
# --------------------------------------------------------------------------

class _Grid:
    pass


_G = _Grid()


def _poly_area(polygon) -> float:
    n = len(polygon)
    return abs(sum(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
                   for i in range(n))) / 2.0


def _setup_grid(polygon, g: float):
    """Rasterise the rectilinear plot onto cells of side g. Cell (i, j) is bit j*stride+i
    (stride = W+1; the extra column is padding that is never free). Lattice POINT (i, j)
    uses the same indexing, with i in 0..W and j in 0..H."""
    G = _G
    G.g = g
    xs = [int(round(p[0] / g)) for p in polygon]
    ys = [int(round(p[1] / g)) for p in polygon]
    G.ox, G.oy = min(xs), min(ys)
    G.W, G.H = max(xs) - G.ox, max(ys) - G.oy
    G.stride = s = G.W + 1
    pts = [(x - G.ox, y - G.oy) for x, y in zip(xs, ys)]
    n = len(pts)
    cells = 0
    for j in range(G.H):
        yc = j + 0.5
        cross = sorted(pts[k][0] for k in range(n)
                       if pts[k][0] == pts[(k + 1) % n][0]
                       and min(pts[k][1], pts[(k + 1) % n][1]) < yc < max(pts[k][1], pts[(k + 1) % n][1]))
        for a in range(0, len(cross) - 1, 2):
            for i in range(cross[a], cross[a + 1]):
                cells |= 1 << (j * s + i)
    G.inside = cells
    bp = 0
    for k in range(n):
        (x1, y1), (x2, y2) = pts[k], pts[(k + 1) % n]
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                bp |= 1 << (y * s + x1)
        elif y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                bp |= 1 << (y1 * s + x)
        else:
            raise ValueError("interior_nosh.py needs a rectilinear plot")
    G.BP = bp
    G.rowfull = (1 << s) - 1
    G.PM = (1 << ((G.H + 1) * s)) - 1                       # all lattice points
    G.cellmask = sum(((1 << G.W) - 1) << (j * s) for j in range(G.H))
    # cells whose closed square meets the plot boundary
    G.TB = (bp | (bp >> 1) | (bp >> s) | (bp >> (s + 1))) & G.cellmask
    G.rm = {}


def _rm(cw: int, ch: int) -> int:
    """Mask of a cw x ch block with its lower-left bit at 0."""
    k = (cw, ch)
    m = _G.rm.get(k)
    if m is None:
        s = _G.stride
        m = ((1 << cw) - 1) * (((1 << (s * ch)) - 1) // ((1 << s) - 1))
        _G.rm[k] = m
    return m


def _cells_of(rx, ry, w, h):
    g = _G.g
    return int(round(rx / g)) - _G.ox, int(round(ry / g)) - _G.oy, int(round(w / g)), int(round(h / g))


def _rect_mask(rx, ry, w, h):
    """Cell mask of the rectangle, or None if it leaves the plot's bounding box."""
    i, j, cw, ch = _cells_of(rx, ry, w, h)
    if cw <= 0 or ch <= 0 or i < 0 or j < 0 or i + cw > _G.W or j + ch > _G.H:
        return None
    return _rm(cw, ch) << (j * _G.stride + i)


def _diff(free, rx, ry, w, h) -> int:
    m = _rect_mask(rx, ry, w, h)
    return free if m is None else free & ~m


def _bits(m: int):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _vertices_mask(F: int) -> int:
    """Lattice points that are corners of the region F (1 or 3 of the 4 cells around the
    point free, or 2 diagonal ones)."""
    s = _G.stride
    A, B, C, D = F << (s + 1), F << s, F << 1, F
    return ((A ^ B ^ C ^ D) | (A & D & ~B & ~C) | (~A & ~D & B & C)) & _G.PM


def _polygon_vertices(F: int) -> List[Tuple[float, float]]:
    s, g, ox, oy = _G.stride, _G.g, _G.ox, _G.oy
    return [(((idx % s) + ox) * g, ((idx // s) + oy) * g) for idx in _bits(_vertices_mask(F))]


def _bounds(F: int):
    s, g = _G.stride, _G.g
    jmin = ((F & -F).bit_length() - 1) // s
    jmax = (F.bit_length() - 1) // s
    acc = 0
    for j in range(jmin, jmax + 1):
        acc |= (F >> (j * s)) & _G.rowfull
    imin = (acc & -acc).bit_length() - 1
    imax = acc.bit_length() - 1
    return ((imin + _G.ox) * g, (jmin + _G.oy) * g, (imax + 1 + _G.ox) * g, (jmax + 1 + _G.oy) * g)


def _carea(F: int) -> float:
    return F.bit_count() * _G.g * _G.g


def _on_bnd(p) -> bool:
    g = _G.g
    i, j = int(round(p[0] / g)) - _G.ox, int(round(p[1] / g)) - _G.oy
    return 0 <= i <= _G.W and 0 <= j <= _G.H and bool((_G.BP >> (j * _G.stride + i)) & 1)


class _SearchTimeout(Exception):
    """Raised by `_check_time` once `time_limit` is exceeded; caught once
    at the bottom of `pack_rectangles_frontier` for an immediate abort."""

Size = Tuple[float, float]
Placement = Tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _polygon_reflex_vertices(polygon: List[Tuple[float, float]], tol: float = 1e-9) -> set:
    """Concave (reflex) vertices of the plot boundary, e.g. the inner
    corner(s) of an L-shaped plot. Handles either winding order via the
    polygon's signed area."""
    n = len(polygon)
    if n < 4:
        return set()
    area2 = sum(
        polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
        for i in range(n)
    )
    ccw = area2 > 0
    out = set()
    for i in range(n):
        ax, ay = polygon[i - 1]
        bx, by = polygon[i]
        cx, cy = polygon[(i + 1) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        is_reflex = (cross < -tol) if ccw else (cross > tol)
        if is_reflex:
            out.add((bx, by))
    return out


def _polygon_convex_vertices(polygon: List[Tuple[float, float]], tol: float = 1e-9) -> set:
    """Convex vertices of the plot boundary (collinear vertices excluded)."""
    n = len(polygon)
    if n < 4:
        return set()
    area2 = sum(
        polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
        for i in range(n)
    )
    ccw = area2 > 0
    out = set()
    for i in range(n):
        ax, ay = polygon[i - 1]
        bx, by = polygon[i]
        cx, cy = polygon[(i + 1) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        is_convex = (cross > tol) if ccw else (cross < -tol)
        if is_convex:
            out.add((bx, by))
    return out


def _at_plot_corner(rx: float, ry: float, w: float, h: float, convex_vertices, tol: float = 1e-9) -> bool:
    """True if one of the rect's corners sits on a convex plot vertex."""
    corners = ((rx, ry), (rx + w, ry), (rx, ry + h), (rx + w, ry + h))
    return any(
        abs(cx - vx) < tol and abs(cy - vy) < tol
        for (cx, cy) in corners for (vx, vy) in convex_vertices
    )


def check_area_match(polygon, rects: Dict[int, Size], tol: float = 1e-6) -> Tuple[bool, str]:
    """(True, "") if the rects' total area is <= the plot area, else
    (False, message). Total area < plot area leaves a hole."""
    plot_area = _poly_area(polygon)
    rect_area = sum(w * h for w, h in rects.values())
    if rect_area > plot_area + tol:
        return False, (f"total rect area {rect_area:g} > plot area {plot_area:g} "
                       f"(excess {rect_area - plot_area:+g})")
    return True, ""


def _lattice_step(polygon, rects) -> float:
    """Largest step that divides every plot coordinate and room side."""
    from fractions import Fraction
    from math import gcd
    vals = [abs(v) for pt in polygon for v in pt] + [v for wh in rects.values() for v in wh]
    fr = [Fraction(v).limit_denominator(1000) for v in vals if v]
    if not fr:
        return 1.0
    den = 1
    for f in fr:
        den = den * f.denominator // gcd(den, f.denominator)
    g = 0
    for f in fr:
        g = gcd(g, int(f * den))
    return float(Fraction(g, den)) if g else 1.0


def check_no_isolated(rects: Dict[int, Size], required_adjacency, interior_ids=()) -> Tuple[bool, str]:
    """(True, "") if no rect is isolated, else (False, message).

    1. Every rect needs at least one required neighbour. Adjacency is exact
       (two rects touch iff required), so a rect with no required neighbour
       would have to sit with nothing touching it.
    2. The required adjacency must form ONE connected graph. Two separate
       groups would be unrelated floorplans sharing a plot, and a group
       with no link to the rest has nothing to be anchored against.
    3. At least one rect must be a boundary rect (not in `interior_ids`);
       together with 2 this means every rect is chained to the boundary.
       (A group made only of interior rects would have to float inside the
       hole, which the corner-anchored search cannot place.)

    A single-rect instance has nobody to be adjacent to and is allowed."""
    if len(rects) <= 1:
        return True, ""
    parent = {i: i for i in rects}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    linked = set()
    for a, b in (required_adjacency or []):
        if a in rects and b in rects and a != b:
            linked.add(a)
            linked.add(b)
            parent[find(a)] = find(b)
    isolated = sorted(i for i in rects if i not in linked)
    if isolated:
        return False, (f"isolated rect(s) {isolated}: every rect needs at least one "
                       f"required neighbour")
    groups: Dict[int, List[int]] = {}
    for i in rects:
        groups.setdefault(find(i), []).append(i)
    if len(groups) > 1:
        parts = sorted(sorted(g) for g in groups.values())
        return False, (f"required adjacency is not one connected graph: {len(parts)} separate "
                       f"groups {parts}")
    interior = set(interior_ids)
    if all(i in interior for i in rects):
        return False, "every rect is interior: at least one rect must touch the plot boundary"
    return True, ""


def _touches_boundary_edge(shape: int) -> bool:
    """True if the cell piece `shape` meets the plot boundary AT ALL (an edge or a point)."""
    return bool(shape & _G.TB)


def _touches_boundary(rx: float, ry: float, w: float, h: float) -> bool:
    """True if the closed rectangle meets the plot boundary at all (an edge segment or a
    single point). A rect that does NOT is an interior rect. Inside the plot the boundary
    lies on lattice lines, so it meets the closed rectangle iff a boundary lattice point
    does."""
    i, j, cw, ch = _cells_of(rx, ry, w, h)
    i0, j0, i1, j1 = max(i, 0), max(j, 0), min(i + cw, _G.W), min(j + ch, _G.H)
    if i0 > i1 or j0 > j1:
        return False
    return bool(_G.BP & (_rm(i1 - i0 + 1, j1 - j0 + 1) << (j0 * _G.stride + i0)))


def _rect_fits(free: int, rx: float, ry: float, w: float, h: float) -> bool:
    if not free:
        return False
    m = _rect_mask(rx, ry, w, h)
    return m is not None and (free & m) == m


def _rects_touch(a: Placement, b: Placement, tol: float = 1e-9) -> bool:
    """True if two axis-aligned placed rectangles share a positive-length edge."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if abs((ax + aw) - bx) < tol or abs((bx + bw) - ax) < tol:
        overlap = min(ay + ah, by + bh) - max(ay, by)
        if overlap > tol:
            return True
    if abs((ay + ah) - by) < tol or abs((by + bh) - ay) < tol:
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap > tol:
            return True
    return False


def _free_space_components(F: int):
    """4-connected pieces of free space (pieces meeting at a single point are separate),
    largest first."""
    s = _G.stride
    out, rest = [], F
    while rest:
        c = rest & -rest
        while True:
            n = (c | (c << 1) | (c >> 1) | (c << s) | (c >> s)) & F
            if n == c:
                break
            c = n
        out.append(c)
        rest &= ~c
    out.sort(key=lambda c: -c.bit_count())
    return out


_ORIENTATION_OFFSETS = [(0, 0), (-1, 0), (0, -1), (-1, -1)]  # multiplied by (w, h) per call


def _candidate_offsets(w, h):
    return [(ox * w, oy * h) for ox, oy in _ORIENTATION_OFFSETS]


# --------------------------------------------------------------------------
# Frontier / PTPG-driven search
#
# Grows outward from placed rectangles using the PTPG graph to decide what
# to try next, instead of rescanning the whole free boundary each step:
#
#  1. Seed: pick a rectangle via whole-boundary MRV, place it at one of
#     its live boundary anchor points.
#  2. That placement exposes at most 2 new anchor points (its corners
#     adjacent to the anchor, excluding the diagonal). At each, only
#     rects required to be adjacent to the exposing rect are considered,
#     ranked by local MRV.
#  3. Triangle shortcut: whenever the rect just placed and the rect that
#     exposed its anchor point share a common required partner, that
#     partner is placed mandatorily at the reflex corner of their L-union
#     (or, if they meet corner-to-corner, along the shared perpendicular
#     line). If no orientation fits or passes boundary-touch, the whole
#     placement backtracks.
#  4. Continues growing the frontier this way; once it's exhausted, a
#     fresh rectangle is seeded via step 1.
# --------------------------------------------------------------------------

def _point_on_rect_edge(vx: float, vy: float, rx: float, ry: float, w: float, h: float, tol: float = 1e-9) -> bool:
    """True if (vx, vy) lies anywhere along one of the rect's 4 edges."""
    on_vert_edge = (abs(vx - rx) < tol or abs(vx - rx - w) < tol) and ry - tol <= vy <= ry + h + tol
    on_horiz_edge = (abs(vy - ry) < tol or abs(vy - ry - h) < tol) and rx - tol <= vx <= rx + w + tol
    return on_vert_edge or on_horiz_edge


_REFLEX_VERTS: set = set()   # plot reflex vertices; one stays an anchor for the partners of a room placed there


def _new_frontier_points(rx, ry, w, h, used_point, free_shape=None):
    """The rectangle's exposed corners for further growth: the 2 corners
    adjacent to the one it was anchored at, plus the diagonal corner if
    `free_shape` is given and that corner is still a live vertex (the
    triangle shortcut anchors at reflex corners, where the diagonal can
    stay exposed). Also, when `free_shape` is given, any other live
    vertex lying along this rect's own edges (not one of its 4 corners) --
    e.g. a plot reflex corner sitting partway along the edge, which is a
    valid anchor for a required partner touching both boundary and rect."""
    corners = [(rx, ry), (rx + w, ry), (rx, ry + h), (rx + w, ry + h)]
    px, py = used_point
    adjacent = [
        c for c in corners
        if not (abs(c[0] - px) < 1e-9 and abs(c[1] - py) < 1e-9)
        and (abs(c[0] - px) < 1e-9 or abs(c[1] - py) < 1e-9)
    ]
    if free_shape is None:
        return adjacent
    diagonal = next(
        c for c in corners
        if abs(c[0] - px) > 1e-9 and abs(c[1] - py) > 1e-9
    )
    live_verts = _polygon_vertices(free_shape)
    result = list(adjacent)
    if (round(px, 6), round(py, 6)) in _REFLEX_VERTS:
        result.append((px, py))   # a plot reflex vertex stays an anchor for this room's partners
    if any(abs(diagonal[0] - vx) < 1e-9 and abs(diagonal[1] - vy) < 1e-9 for (vx, vy) in live_verts):
        result.append(diagonal)
    for (vx, vy) in live_verts:
        if any(abs(vx - cx) < 1e-9 and abs(vy - cy) < 1e-9 for (cx, cy) in corners):
            continue
        if _point_on_rect_edge(vx, vy, rx, ry, w, h):
            result.append((vx, vy))
    return result


def _reflex_corners(box_a: Placement, box_b: Placement, tol: float = 1e-9):
    """ALL inner (concave) corners of the union of two touching rectangles.
    An L-union has one; a stepped union (one rect overhangs the other, e.g.
    a room sitting across the end of a longer one) has two. A third room
    touching both can nestle into either."""
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    xs = [x1, x1 + w1, x2, x2 + w2]
    ys = [y1, y1 + h1, y2, y2 + h2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    corners = {
        (x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1),
        (x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2),
    }
    return sorted((cx, cy) for (cx, cy) in corners
                  if xmin + tol < cx < xmax - tol and ymin + tol < cy < ymax - tol)


def _union_is_rectangle(box_a: Placement, box_b: Placement, tol: float = 1e-9) -> bool:
    """True if two touching rectangles join into ONE rectangle (they share
    a full side, e.g. two rooms side by side forming a bar). The triangle
    shortcut is only meaningful when the union is NOT a rectangle: then the
    third room nestles into the reflex corner."""
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    bw = max(x1 + w1, x2 + w2) - min(x1, x2)
    bh = max(y1 + h1, y2 + h2) - min(y1, y2)
    return abs(bw * bh - (w1 * h1 + w2 * h2)) < tol


def _shared_corners(box_a: Placement, box_b: Placement, tol: float = 1e-9):
    """Points that are a corner of both box_a and box_b -- non-empty only
    when the two meet exactly corner-to-corner (flush along their touching
    edge) at one or both ends. Equal-sized touching edges are flush at
    both ends, giving two distinct shared corners."""
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    corners_a = {(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)}
    corners_b = {(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)}
    out = []
    for ca in corners_a:
        for cb in corners_b:
            if abs(ca[0] - cb[0]) < tol and abs(ca[1] - cb[1]) < tol:
                out.append(ca)
    return out


def _perpendicular_search_line(box_a: Placement, box_b: Placement, shared_corner, tol: float = 1e-6):
    """When two rects meet exactly at a shared corner, the axis they
    DON'T touch along, at the shared corner's coordinate, is a line both
    rects lie flush on -- a third rect touching both can anchor anywhere
    on it. Returns (axis, value), or None if they don't touch along a
    positive-length edge."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    y_overlap = min(ay + ah, by + bh) - max(ay, by)
    if abs(x_overlap) < tol and y_overlap > tol:
        return ("y", shared_corner[1])
    if abs(y_overlap) < tol and x_overlap > tol:
        return ("x", shared_corner[0])
    return None


def _flush_side_has_room(pa, pb, free_shape, t_size, step, tol=1e-9):
    """a and b meet flush on a straight line (they share a corner and have
    collinear edges). True if a room of size `t_size` (either orientation)
    can be placed on the FREE side of that line, with a side on the line,
    overlapping a's edge AND b's edge by a positive length, entirely inside
    the free space (not on a boundary or an already-placed rect)."""
    for sc in _shared_corners(pa, pb):
        ln = _perpendicular_search_line(pa, pb, sc)
        if ln is None:
            continue
        axis, v = ln
        i = 0 if axis == "x" else 1          # coordinate that is fixed (= v)
        j = 1 - i                            # coordinate along the line
        a_lo, a_hi = pa[j], pa[j] + pa[2 + j]
        b_lo, b_hi = pb[j], pb[j] + pb[2 + j]
        bodies_before = abs(pa[i] + pa[2 + i] - v) < tol   # a, b lie before the line
        for tw, th in {t_size, (t_size[1], t_size[0])}:
            t_across = (tw, th)[i]           # t's extent across the line
            t_along = (tw, th)[j]            # t's extent along the line
            lo = min(a_lo, b_lo) - t_along
            hi = max(a_hi, b_hi)
            n = int(round((hi - lo) / step))
            for k in range(n + 1):
                s0 = lo + k * step
                if min(s0 + t_along, a_hi) - max(s0, a_lo) <= tol:
                    continue
                if min(s0 + t_along, b_hi) - max(s0, b_lo) <= tol:
                    continue
                c0 = v if bodies_before else v - t_across
                if i == 0:
                    ok = _rect_fits(free_shape, c0, s0, t_across, t_along)
                else:
                    ok = _rect_fits(free_shape, s0, c0, t_along, t_across)
                if ok:
                    return True
    return False


def _forced_points(pa, pb, free_shape, t_size, step):
    """Anchor points where a room t that must touch BOTH placed rooms a and
    b is forced to have a corner, or None when nothing is forced.
    Not forced when: a,b join into a rectangle; or a,b meet flush on a
    line and t fits on the free side of that line (t may then sit anywhere
    along the line -- left to the normal search / lattice grid). Otherwise
    t can only touch both of them by having a corner on a reflex corner of
    their L-shaped/stepped union."""
    if _union_is_rectangle(pa, pb):
        return None
    if _flush_side_has_room(pa, pb, free_shape, t_size, step):
        return None
    pts = _reflex_corners(pa, pb)
    return pts or None


def pack_rectangles_frontier(
    polygon,
    rects: Dict[int, Size],
    required_adjacency=None,
    *,
    interior_ids: Iterable[int] = (),
    time_limit: Optional[float] = 60.0,
    collect_solutions: Optional[list] = None,
    max_solutions: Optional[int] = None,
):
    """Frontier search. Returns (found, placements_or_None, nodes_explored).
    found is True / False / None (feasible / proven infeasible / cut off by
    time_limit). Returns False immediately if the total room area exceeds the
    plot area, if some room has no required neighbour, if the required adjacency
    is not one connected graph, or if every room is interior.

    The leftover (plot area minus room area, zero for an exact tiling) must be
    empty or one connected piece that does not touch the plot boundary at all.
    `interior_ids` rooms must not touch the plot boundary (not even at a point);
    every other room must meet it (a point is enough). `time_limit` (seconds,
    wall-clock; None = unbounded) is checked in every hot loop.

    Pass `collect_solutions` (a list) to keep searching past the first valid
    layout: each new one is appended and the search backtracks for a genuinely
    different one, up to `max_solutions` (None = exhaustive). In this mode the
    second return value is always `collect_solutions` itself, and the first one
    is True only if `max_solutions` was reached. The completeness proof is for
    the default single-solution mode."""
    sizes = dict(rects)
    if not check_area_match(polygon, sizes)[0]:
        return False, collect_solutions, 0
    if not check_no_isolated(sizes, required_adjacency, interior_ids)[0]:
        return False, collect_solutions, 0

    inner_l_points = _polygon_reflex_vertices(polygon)
    convex_vertices = _polygon_convex_vertices(polygon)
    required_adjacency = required_adjacency or []
    interior = set(interior_ids)
    unknown = interior - set(sizes)
    if unknown:
        raise ValueError(f"interior_ids references unknown rect ids: {sorted(unknown)}")

    seen_layouts = {_layout_key(sol) for sol in collect_solutions} if collect_solutions else set()

    required_partners: Dict[int, List[int]] = {i: [] for i in sizes}
    for a, b in required_adjacency:
        if a == b:
            raise ValueError(f"required_adjacency has a self-pair: ({a!r}, {a!r})")
        if a not in sizes or b not in sizes:
            raise ValueError(f"required_adjacency references an unknown rect id ({a!r}, {b!r})")
        if b not in required_partners[a]:
            required_partners[a].append(b)
        if a not in required_partners[b]:
            required_partners[b].append(a)

    # Triangle-corner rule (3b): a boundary room with exactly 2 partners that touch
    # each other must have a corner on a convex plot vertex. `placements_of` rejects
    # such a room away from convex corners and `seed_step` tries them first.
    triangle_corner_ids = {
        i for i, ps in required_partners.items()
        if len(ps) == 2 and ps[1] in required_partners.get(ps[0], [])
        and i not in interior
    }

    if len(inner_l_points) > 1:
        raise ValueError("the plot may have at most one reflex vertex (a rectangle or an L-shape)")
    _REFLEX_VERTS.clear()
    _REFLEX_VERTS.update((round(float(v[0]), 6), round(float(v[1]), 6)) for v in inner_l_points)
    placements: Dict[int, Placement] = {}
    corner_ban: set = set()   # rooms already refuted on a convex plot corner (corner sweep)
    nodes = [0]
    deadline = time.perf_counter() + time_limit if time_limit is not None else None

    def _check_time():
        if deadline is not None and time.perf_counter() > deadline:
            raise _SearchTimeout()

    def required_adjacency_ok(idx, rx, ry, w, h) -> bool:
        cand = (rx, ry, w, h)
        partners = required_partners[idx]
        for j, placed in placements.items():
            if _rects_touch(cand, placed) != (j in partners):
                return False
        return True

    def boundary_ok(idx, rx, ry, w, h) -> bool:
        touches = _touches_boundary(rx, ry, w, h)
        if idx in interior:
            return not touches
        return touches

    def placements_of(idx, point, free_shape):
        """All valid (rx, ry, w, h, new_free) placements of rect `idx`
        anchored at `point`, both orientations, all 4 quadrant offsets.
        Cheap plain-Python checks (`boundary_ok`, `required_adjacency_ok`)
        run before the shapely `.difference()`."""
        px, py = point
        w0, h0 = sizes[idx]
        out = []
        for uw, uh in {(w0, h0), (h0, w0)}:
            for ox, oy in _candidate_offsets(uw, uh):
                rx, ry = px + ox, py + oy
                if not _rect_fits(free_shape, rx, ry, uw, uh):
                    continue
                if corner_ban and idx in corner_ban and _at_plot_corner(rx, ry, uw, uh, convex_vertices):
                    continue
                if idx in triangle_corner_ids and not _at_plot_corner(rx, ry, uw, uh, convex_vertices):
                    continue
                if not boundary_ok(idx, rx, ry, uw, uh):
                    continue
                if not required_adjacency_ok(idx, rx, ry, uw, uh):
                    continue
                new_free = _diff(free_shape, rx, ry, uw, uh)
                out.append((rx, ry, uw, uh, new_free))
        return out

    def _rect_still_placeable(q_idx, free_shape) -> bool:
        """Whether the not-yet-placed `q_idx` could be placed anywhere in
        the current free space, checked across every live vertex.
        `required_adjacency_ok` checks exactly against every placed rect,
        so this existence check is sound: free space only shrinks, so a
        rect unplaceable now stays unplaceable later in this branch (a
        boundary stretch can still split into a new vertex later, but
        that's an accepted limitation of the anchor-at-a-vertex convention).

        Interior rects are always reported placeable: they touch no
        boundary, so until their neighbours are placed none of their
        corners is on a live vertex and this scan would wrongly rule them
        out."""
        if q_idx in interior:
            return True
        for v in _polygon_vertices(free_shape):
            _check_time()
            if placements_of(q_idx, v, free_shape):
                return True
        # exact check: a room may slide along a wall and have no corner
        # on a free-space corner yet. Look for ANY lattice position where
        # it fits and matches the required adjacency exactly (cheap
        # adjacency / boundary tests first, shapely fit last).
        w0, h0 = sizes[q_idx]
        if not free_shape:
            return False
        fminx, fminy, fmaxx, fmaxy = _bounds(free_shape)
        for uw, uh in {(w0, h0), (h0, w0)}:
            nx = int(round((fmaxx - uw - fminx) / grid_step))
            ny = int(round((fmaxy - uh - fminy) / grid_step))
            for ix in range(nx + 1):
                rx = fminx + ix * grid_step
                for iy in range(ny + 1):
                    ry = fminy + iy * grid_step
                    if not required_adjacency_ok(q_idx, rx, ry, uw, uh):
                        continue
                    if not boundary_ok(q_idx, rx, ry, uw, uh):
                        continue
                    _check_time()
                    if _rect_fits(free_shape, rx, ry, uw, uh):
                        return True
        return False

    def _slot_count(opts, point):
        """MRV constrainedness: number of distinct quadrant slots among
        `opts`. Fitting both orientations in the same quadrant doesn't
        count as extra freedom, so this counts slots, not raw
        orientation x quadrant combinations."""
        px, py = point
        slots = set()
        for (rx, ry, w, h, _new_free) in opts:
            qx = 0 if abs(rx - px) < 1e-9 else 1
            qy = 0 if abs(ry - py) < 1e-9 else 1
            slots.add((qx, qy))
        return len(slots)

    def local_score(idx, point, free_shape):
        """`_slot_count` computed fresh; use only when `placements_of`'s
        result isn't already in hand elsewhere."""
        return _slot_count(placements_of(idx, point, free_shape), point)

    def point_score(point, originator, free_shape):
        """MRV score for a frontier point: score of its most constrained
        PTPG-eligible candidate, +inf if none."""
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids() and eligible(originator, i)]
        if not candidates:
            return float("inf")
        return min(local_score(i, point, free_shape) for i in candidates)

    def order_points(points_with_originator, free_shape):
        """Sort newly-exposed (point, originator) pairs by MRV."""
        return sorted(points_with_originator, key=lambda po: point_score(po[0], po[1], free_shape))

    def seed_step(free_shape):
        """Seed selection when the frontier is empty (single free region).

        While any triangle-corner rect is unplaced: every layout has each
        such rect on a free convex plot corner, so ONE of them (fewest
        corner placements) is seeded at every such corner, most
        constrained corner first. If none of those seeds works, no layout
        exists and the node is a dead end (empty result).

        Otherwise: fix one anchor point, rank every remaining id placeable
        there by degree in required_adjacency (corner rooms tend to be
        lowest-degree), then geometric local_score, then largest area, and
        exhaust every tied candidate at that point before trying a
        different point. Among degree-1 candidates, a point that's also one
        of the plot's own inner-L corners (`inner_l_points`) is preferred,
        since a degree-1 room tends to sit in a plot's inward notch on real
        floorplans."""
        verts = _polygon_vertices(free_shape)
        anchor_verts = [p for p in verts if _on_bnd(p)]
        all_anchor_verts = list(anchor_verts)   # the corner clause below needs EVERY live convex corner
        # every valid layout has exactly one room with a corner at each convex plot
        # vertex, so ONE live convex vertex with EVERY room tried there covers every layout
        conv = [p for p in anchor_verts if any(abs(p[0] - vx) < 1e-9 and abs(p[1] - vy) < 1e-9 for (vx, vy) in convex_vertices)]
        if conv:
            anchor_verts = [min(conv)]
        remaining = remaining_ids()
        # exhaustive corner clause: every layout has each unplaced
        # triangle-corner rect on a free convex plot corner. Seed ONE of
        # them (the one with the fewest corner placements) at every such
        # corner; if all of those seeds fail, no layout exists: stop.
        corner_ids = [i for i in remaining if i in triangle_corner_ids]
        if corner_ids:
            corner_anchors = [
                p for p in all_anchor_verts
                if any(abs(p[0] - vx) < 1e-9 and abs(p[1] - vy) < 1e-9 for (vx, vy) in convex_vertices)
            ]
            table = {}
            for idx in corner_ids:
                seen, per_pt = set(), {}
                for p in corner_anchors:
                    _check_time()
                    opts = [o for o in placements_of(idx, p, free_shape) if o[:4] not in seen]
                    seen.update(o[:4] for o in opts)
                    if opts:
                        per_pt[p] = opts
                table[idx] = per_pt
            idx = min(corner_ids, key=lambda i: (sum(len(o) for o in table[i].values()),
                                                 -(sizes[i][0] * sizes[i][1]), i))

            def corner_key(p):
                # most constrained corner first: fewest slots over ALL
                # triangle rects that fit at p, then the witness's own slots
                best = min(_slot_count(table[i][p], p) for i in corner_ids if p in table[i])
                return (best, _slot_count(table[idx][p], p), p)

            return [(0, corner_key(p)[0], p, [(idx, table[idx][p])]) for p in sorted(table[idx], key=corner_key)]

        # corner sweep: take the rooms in a decided order (fewest corner placements first, then
        # fewest partners, then larger area); try ONE room at EVERY live convex corner; if that
        # room fails everywhere, move on to the next room and from then on forbid the failed
        # rooms from sitting on a convex plot corner (their branch refuted every layout that
        # has them there). Empty result: no room can sit on any convex corner -> no layout.
        conv_all = [p for p in all_anchor_verts
                    if any(abs(p[0] - vx) < 1e-9 and abs(p[1] - vy) < 1e-9 for (vx, vy) in convex_vertices)]
        if conv_all:
            rows = []
            for idx in remaining:
                seen, per_pt = set(), {}
                for p in conv_all:
                    _check_time()
                    opts = [o for o in placements_of(idx, p, free_shape) if o[:4] not in seen]
                    seen.update(o[:4] for o in opts)
                    if opts:
                        per_pt[p] = opts
                if per_pt:
                    w, h = sizes[idx]
                    rows.append((len(required_partners.get(idx, [])), -(w * h), sum(len(o) for o in per_pt.values()), idx, per_pt))
            rows.sort(key=lambda r: r[:4])
            out, banned = [], []
            for (deg, _a, _n, idx, per_pt) in rows:
                for p in sorted(per_pt):
                    out.append((deg, 0, p, [(idx, per_pt[p])], tuple(banned)))
                banned.append(idx)
            return out
        per_point = []
        for p in anchor_verts:
            _check_time()
            scored = []
            for idx in remaining:
                opts = placements_of(idx, p, free_shape)
                if not opts:
                    continue
                deg = len(required_partners.get(idx, []))
                w, h = sizes[idx]
                scored.append((deg, _slot_count(opts, p), -(w * h), idx, opts))
            if not scored:
                continue
            # one entry per degree level, so a failed lowest-degree seed
            # falls back to the next level rather than giving up
            for level in sorted({d for d, _, _, _, _ in scored}):
                level_score = min(sc for d, sc, _, _, _ in scored if d == level)
                tied = sorted(
                    ((idx, opts) for d, sc, _, idx, opts in scored if d == level),
                    key=lambda t: -(sizes[t[0]][0] * sizes[t[0]][1]),
                )
                is_inner_l = 0 if (level == 1 and p in inner_l_points) else 1
                per_point.append((level, is_inner_l, level_score, p, tied))
        if not per_point:
            return None
        per_point.sort(key=lambda t: t[:3])
        return [(deg, score, p, tied) for (deg, _il, score, p, tied) in per_point]

    border_ids = [i for i in sizes if i not in interior]
    grid_step = _lattice_step(polygon, sizes)
    _setup_grid(polygon, grid_step)

    def eligible(originator, idx) -> bool:
        """May `idx` be anchored at a frontier point exposed by `originator`?
        An interior room's corners anchor only other interior rooms."""
        return originator not in interior or idx in interior

    def remaining_ids():
        return [i for i in sizes if i not in placements]

    def leftover_ok(free_shape) -> bool:
        """All rects placed: the leftover must be an interior hole -- no
        piece touches the plot boundary, and it is one connected piece."""
        if not free_shape:
            return True
        comps = _free_space_components(free_shape)
        if len(comps) > 1:
            return False
        return not any(_touches_boundary_edge(c) for c in comps)

    def hole_prune_ok(free_shape, remaining) -> bool:
        """Hole-mode pruning (sound). (1) Free area is plot area - placed
        area, so the leftover is always the fixed budget. (2) Once the free
        space has split into several pieces, a piece that shares an edge
        with the plot boundary can never end up as the hole, so remaining
        rects must fill it except for at most the hole budget H (subset-sum
        window on area, and it must be big enough for the smallest
        remaining rect). (3) More than one
        interior piece that cannot be filled at all violates the single-hole condition
        only if their combined area exceeds the budget -- already implied
        by (1), so not re-checked."""
        comps = _free_space_components(free_shape)
        if len(comps) <= 1:
            return True
        areas = [sizes[i][0] * sizes[i][1] for i in remaining]
        integral = all(abs(a - round(a)) < 1e-9 for a in areas)
        bits = None
        if integral:
            bits = 1
            for a in areas:
                bits |= bits << int(round(a))
        min_side = min((min(sizes[i]) for i in remaining), default=0)
        # the hole may still lie inside any piece that is not yet split off,
        # so a boundary piece may keep up to H of unfilled area (H = total
        # free area - remaining rect area, the fixed hole budget)
        H = max(0.0, _carea(free_shape) - sum(areas))
        need = 0.0
        for c in comps:
            if not _touches_boundary_edge(c):
                continue
            ca = _carea(c)
            need += ca
            if not remaining:
                return False
            minx, miny, maxx, maxy = _bounds(c)
            if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
                return False
            if bits is not None and abs(ca - round(ca)) < 1e-6 and abs(H - round(H)) < 1e-6:
                hi = int(round(ca)); lo = max(0, hi - int(round(H)))
                window = (bits >> lo) & ((1 << (hi - lo + 1)) - 1)
                if not window:
                    return False
        if need - H > sum(areas) + 1e-6:
            return False
        return True

    def grid_options(idx, free_shape):
        """All lattice placements of interior rect `idx` that fit, stay off
        the plot boundary and match the required adjacency exactly against
        every placed rect."""
        w0, h0 = sizes[idx]
        out = []
        if not free_shape:
            return out
        fminx, fminy, fmaxx, fmaxy = _bounds(free_shape)
        # general triangle shortcut where it applies: two placed required
        # partners of idx that require each other and whose union is not a
        # rectangle -> idx nestles at their reflex corner / shared line
        ps = [q for q in required_partners.get(idx, []) if q in placements]
        pts = None
        for i1 in range(len(ps)):
            for i2 in range(i1 + 1, len(ps)):
                a, b = ps[i1], ps[i2]
                if b not in required_partners.get(a, []):
                    continue
                fp = _forced_points(placements[a], placements[b], free_shape, sizes[idx], grid_step)
                if fp is None:
                    continue
                pts = (pts or [])
                pts.extend(fp)
        if pts is not None:
            seen = set()
            for pt in pts:
                for (rx, ry, uw, uh, _nf) in placements_of(idx, pt, free_shape):
                    k = (round(rx, 6), round(ry, 6), uw, uh)
                    if k in seen:
                        continue
                    seen.add(k)
                    if boundary_ok(idx, rx, ry, uw, uh) and required_adjacency_ok(idx, rx, ry, uw, uh):
                        out.append((rx, ry, uw, uh))
            return out
        for uw, uh in {(w0, h0), (h0, w0)}:
            nx = int(round((fmaxx - uw - fminx) / grid_step))
            ny = int(round((fmaxy - uh - fminy) / grid_step))
            for ix in range(nx + 1):
                rx = fminx + ix * grid_step
                for iy in range(ny + 1):
                    ry = fminy + iy * grid_step
                    _check_time()
                    if not _rect_fits(free_shape, rx, ry, uw, uh):
                        continue
                    if not boundary_ok(idx, rx, ry, uw, uh):
                        continue
                    if not required_adjacency_ok(idx, rx, ry, uw, uh):
                        continue
                    out.append((rx, ry, uw, uh))
        return out

    def grid_fill(free_shape, remaining):
        """Place the remaining interior rects anywhere on the lattice."""
        nodes[0] += 1
        _check_time()
        if not remaining:
            return solve(free_shape, [])
        best = None
        for idx in remaining:
            opts = grid_options(idx, free_shape)
            if not opts:
                return False
            key = (len(opts), -(sizes[idx][0] * sizes[idx][1]), idx)
            if best is None or key < best[0]:
                best = (key, idx, opts)
        _, idx, opts = best
        for (rx, ry, uw, uh) in opts:
            _check_time()
            placements[idx] = (rx, ry, uw, uh)
            result = grid_fill(_diff(free_shape, rx, ry, uw, uh),
                               [i for i in remaining if i != idx])
            if result is True or result is None:
                return result
            del placements[idx]
        return False

    _edges = list(zip(polygon, list(polygon[1:]) + [polygon[0]]))
    _vert_set = {(round(float(x), 6), round(float(y), 6)) for (x, y) in polygon}

    def _on_wall(px, py):
        for (x1, y1), (x2, y2) in _edges:
            if abs(x1 - x2) < 1e-9:
                if abs(px - x1) < 1e-9 and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9:
                    return True
            elif abs(py - y1) < 1e-9 and min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9:
                return True
        return False

    def _partner_fits_at(q, v, free_shape):
        """necessary conditions only: fits in the free space with a corner at v,
        touches the boundary, exact adjacency to every placed room"""
        w0, h0 = sizes[q]
        for uw, uh in {(w0, h0), (h0, w0)}:
            for ox, oy in _candidate_offsets(uw, uh):
                rx, ry = v[0] + ox, v[1] + oy
                if (_rect_fits(free_shape, rx, ry, uw, uh) and boundary_ok(q, rx, ry, uw, uh)
                        and required_adjacency_ok(q, rx, ry, uw, uh)):
                    return True
        return False

    def wall_corner_ok(free_shape):
        """every corner v of a placed boundary room that lies on the plot
        boundary and is not a plot vertex needs a partner at v: placed, or
        still able to fit there"""
        for o, (ox_, oy_, ow, oh) in placements.items():
            if o in interior:
                continue
            ps = required_partners.get(o, [])
            for v in {(ox_, oy_), (ox_ + ow, oy_), (ox_, oy_ + oh), (ox_ + ow, oy_ + oh)}:
                if not _on_wall(v[0], v[1]) or (round(float(v[0]), 6), round(float(v[1]), 6)) in _vert_set:
                    continue
                done = False
                for j in ps:
                    pj = placements.get(j)
                    if pj is not None and pj[0] - 1e-9 <= v[0] <= pj[0] + pj[2] + 1e-9 and pj[1] - 1e-9 <= v[1] <= pj[1] + pj[3] + 1e-9:
                        done = True
                        break
                if done:
                    continue
                for q in ps:
                    if q not in placements and q not in interior and _partner_fits_at(q, v, free_shape):
                        done = True
                        break
                if not done:
                    return False
        return True

    def partner_space_ok(free_shape):
        """Every unplaced room q that has a placed partner o must still be able
        to touch o: some lattice position along o's four sides where q fits in
        the free space, is boundary-consistent and exactly adjacent to every
        placed room. Necessary conditions only."""
        for q in sizes:
            if q in placements:
                continue
            anchors = [o for o in required_partners.get(q, []) if o in placements]
            if not anchors:
                continue
            ox_, oy_, ow, oh = placements[anchors[0]]
            w0, h0 = sizes[q]
            found = False
            for uw, uh in {(w0, h0), (h0, w0)}:
                cand_pos = []
                n_y = int(round((oh + uh) / grid_step)) - 1
                n_x = int(round((ow + uw) / grid_step)) - 1
                for k in range(1, n_y + 1):
                    ry = oy_ - uh + k * grid_step
                    cand_pos.append((ox_ + ow, ry)); cand_pos.append((ox_ - uw, ry))
                for k in range(1, n_x + 1):
                    rx = ox_ - uw + k * grid_step
                    cand_pos.append((rx, oy_ + oh)); cand_pos.append((rx, oy_ - uh))
                for (rx, ry) in cand_pos:
                    if (_rect_fits(free_shape, rx, ry, uw, uh) and boundary_ok(q, rx, ry, uw, uh)
                            and required_adjacency_ok(q, rx, ry, uw, uh)):
                        found = True
                        break
                if found:
                    break
            if not found:
                return False
        return True

    def solve(free_shape, frontier):
        nodes[0] += 1
        _check_time()
        if not partner_space_ok(free_shape):
            return False
        if not wall_corner_ok(free_shape):
            return False
        remaining = remaining_ids()
        if (remaining and all(i in interior for i in remaining)
                and all(b in placements for b in border_ids)):
            # every boundary rect is down: free space still touching the plot
            # boundary can never be filled by interior rects
            if any(_touches_boundary_edge(c) for c in _free_space_components(free_shape)):
                return False
            return grid_fill(free_shape, remaining)
        if not remaining:
            if not leftover_ok(free_shape):
                return False
            if collect_solutions is not None:
                key = _layout_key(placements)
                if key not in seen_layouts:
                    seen_layouts.add(key)
                    collect_solutions.append(dict(placements))
                    if max_solutions is not None and len(collect_solutions) >= max_solutions:
                        return True
                # keep backtracking to find a genuinely different placement
                return False
            return True

        if not hole_prune_ok(free_shape, remaining):
            return False
        if frontier:
            # plain FIFO over the frontier
            (point, originator), *rest = frontier
            candidates = [i for i in required_partners.get(originator, []) if i in remaining and eligible(originator, i)]
            if not candidates:
                return solve(free_shape, rest)
            cand_opts = {i: placements_of(i, point, free_shape) for i in candidates}
            candidates.sort(key=lambda i: (_slot_count(cand_opts[i], point), -(sizes[i][0] * sizes[i][1])))

            for cand in candidates:
                for (rx, ry, w, h, new_free) in cand_opts[cand]:
                    _check_time()
                    placements[cand] = (rx, ry, w, h)
                    new_points = order_points([(p, cand) for p in _new_frontier_points(rx, ry, w, h, point, new_free)], new_free)

                    # forward-check: did this placement leave a required
                    # partner of `cand` or `originator` unplaceable?
                    dead = False
                    for q in required_partners.get(cand, []):
                        if q in remaining_ids() and not _rect_still_placeable(q, new_free):
                            dead = True
                            break
                    if not dead:
                        for q in required_partners.get(originator, []):
                            if q != cand and q in remaining_ids() and not _rect_still_placeable(q, new_free):
                                dead = True
                                break
                    if dead:
                        del placements[cand]
                        continue

                    # triangle shortcut: force in a required partner
                    # shared by two already-placed rects, at their L-union
                    # reflex corner or the shared perpendicular line
                    def resolve_chain(pending_pairs, extra_points, cur_free):
                        if not pending_pairs:
                            result = solve(cur_free, extra_points + rest)
                            if result is True or result is None:
                                return result
                            return False

                        (a, b), *rest_pairs = pending_pairs
                        rect_union = _union_is_rectangle(placements[a], placements[b])
                        if rect_union:
                            # a and b join into one rectangle: no reflex
                            # corner for a third room to nestle into, so
                            # the shortcut does not apply -- the shared
                            # partner is placed later by the normal search
                            return resolve_chain(rest_pairs, extra_points, cur_free)
                        common = [
                            t for t in required_partners.get(a, [])
                            if t in required_partners.get(b, []) and t in remaining_ids()
                            and eligible(a, t) and eligible(b, t)
                        ]
                        if not common:
                            return resolve_chain(rest_pairs, extra_points, cur_free)

                        forced = {}
                        for t_idx in common:
                            fp = _forced_points(placements[a], placements[b], cur_free, sizes[t_idx], grid_step)
                            if fp is not None:
                                forced[t_idx] = fp
                        if not forced:
                            # nothing forced: placed later by the normal search
                            return resolve_chain(rest_pairs, extra_points, cur_free)

                        for t_idx in forced:
                            for pt in forced[t_idx]:
                                for (tx, ty, tw, th, t_free) in placements_of(t_idx, pt, cur_free):
                                    _check_time()
                                    placements[t_idx] = (tx, ty, tw, th)
                                    t_points = order_points([(p, t_idx) for p in _new_frontier_points(tx, ty, tw, th, pt, t_free)], t_free)
                                    result = resolve_chain(rest_pairs, extra_points + t_points, t_free)
                                    if result is True or result is None:
                                        return result
                                    del placements[t_idx]
                        return False

                    result = resolve_chain([(originator, cand)], new_points, new_free)
                    if result is True or result is None:
                        return result
                    del placements[cand]

            # every candidate failed at this entry: skip the entry (no dead-end rule;
            # the wall-corner rule and the space check cut dead states at the top of solve)
            return solve(free_shape, rest)

        # empty stack with a boundary room unplaced: not on any S-path (Lemma 2, Steps 2-6 give an
        # entry while a boundary room is unplaced), so the branch is dead and nothing is reseeded.
        if placements and any(i not in interior for i in remaining):
            return False
        # frontier empty: seed a fresh rectangle, point-first
        per_point = seed_step(free_shape)
        if not per_point:
            return False
        outer_ban = set(corner_ban)
        for entry in per_point:
            _deg, _score, point, tied = entry[:4]
            corner_ban.clear(); corner_ban.update(outer_ban); corner_ban.update(entry[4] if len(entry) > 4 else ())
            for seed, opts in tied:
                if seed not in remaining_ids():
                    continue
                for (rx, ry, uw, uh, new_free) in opts:
                    placements[seed] = (rx, ry, uw, uh)
                    new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                    result = solve(new_free, new_points)
                    if result is True or result is None:
                        return result
                    del placements[seed]
        corner_ban.clear(); corner_ban.update(outer_ban)
        return False

    try:
        result = solve(_G.inside, [])
    except _SearchTimeout:
        result = None

    if collect_solutions is not None:
        return result, collect_solutions, nodes[0]
    if result is True:
        return True, dict(placements), nodes[0]
    if result is False:
        return False, None, nodes[0]
    return None, None, nodes[0]


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _layout_key(placements: Dict[int, Placement]) -> tuple:
    """Canonical geometry-only key for deduplicating layouts."""
    return tuple(sorted(
        (round(x, 6), round(y, 6), round(w, 6), round(h, 6))
        for x, y, w, h in placements.values()
    ))


def verify_placements(polygon, placements) -> Tuple[int, int, int]:
    """Returns (overlap_count, outside_boundary_count, hole_count), exactly, on the lattice."""
    g = _lattice_step(polygon, {k: (w, h) for k, (_x, _y, w, h) in placements.items()})
    _setup_grid(polygon, g)
    masks = []
    for (x, y, w, h) in placements.values():
        i, j, cw, ch = _cells_of(x, y, w, h)
        masks.append(_rm(cw, ch) << (j * _G.stride + i) if i >= 0 and j >= 0 else 1 << (10 ** 6))
    overlaps = sum(1 for a in range(len(masks)) for b in range(a + 1, len(masks)) if masks[a] & masks[b])
    outside = sum(1 for m in masks if m & ~_G.inside)
    occupied = 0
    for m in masks:
        occupied |= m
    leftover = _G.inside & ~occupied
    return overlaps, outside, len(_free_space_components(leftover))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_solution_grid(polygon, solutions, save_path, cols=5, labels=None):
    """Saves a grid with one panel per solution (a dict of placements).

    `labels` (optional) maps room id -> text drawn on that room, e.g.
    {0: "Living", 1: "Kitchen"}. A room without a label shows its number.
    The room size is printed under the label."""
    labels = labels or {}
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(solutions)
    if n == 0:
        raise ValueError("no solutions to render")
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    xs_all = [p[0] for p in polygon]
    ys_all = [p[1] for p in polygon]
    xlim = (min(xs_all) - 1, max(xs_all) + 1)
    ylim = (min(ys_all) - 1, max(ys_all) + 1)

    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = [axes] if n == 1 else axes.flatten()

    bx = [p[0] for p in polygon] + [polygon[0][0]]
    by = [p[1] for p in polygon] + [polygon[0][1]]

    for i, placements in enumerate(solutions):
        ax = axes[i]
        ax.plot(bx, by, "k-", linewidth=1.5)
        for idx, (x, y, w, h) in placements.items():
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=0.8, edgecolor="black",
                                            facecolor="tab:blue", alpha=0.65))
            ax.text(x + w / 2, y + h / 2, f"{labels.get(idx, idx)}\n{w:g}x{h:g}",
                    ha="center", va="center", fontsize=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"solution {i + 1}", fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Example plots
# --------------------------------------------------------------------------

# Each example may also carry an optional  labels={0: "Living", 1: "Kitchen", ...}
# (room id -> name). Labelled rooms are drawn with that name; others with their number.
EXAMPLES = {
    1: dict(
        polygon=[(0, 0), (25, 0), (25, 15), (15, 15), (15, 30), (0, 30)],
        rects=dict(enumerate([(15, 6), (10, 6), (15, 4), (15, 4), (8, 7), (7, 7), (5, 9), (16, 9), (4, 9)])),
        required_adjacency=[(0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 7), (2, 3),
                             (3, 6), (3, 7), (3, 8), (4, 5), (6, 7), (6, 8)],
    ),
    2: dict(
        polygon=[(0, 0), (24, 0), (24, 16), (16, 16), (16, 24), (0, 24)],
        rects=dict(enumerate([(24, 4), (8, 7), (8, 7), (13, 4), (7, 4), (4, 4), (16, 4), (16, 5), (7, 4), (8, 4), (1, 4)])),
        required_adjacency=[(0, 6), (0, 7), (1, 4), (1, 7), (2, 3), (2, 4), (3, 5), (3, 6), (5, 6),
                             (8, 3), (8, 6), (8, 7), (8, 0), (8, 9), (9, 3), (9, 2), (9, 4), (9, 7), (9, 10), (10, 4), (10, 1), (10, 7)],
        interior=[8, 9, 10],
    ),
    3: dict(
        polygon=[(0, 0), (35, 0), (35, 20), (22, 20), (22, 40), (0, 40)],
        rects=dict(enumerate([(8, 8), (13, 10), (14, 8), (17, 12), (12, 10), (5, 5), (5, 5),
                               (12, 10), (6, 5), (6, 5), (10, 10), (5, 7), (5, 5), (12, 10)])),
        required_adjacency=[(0, 2), (0, 12), (0, 3), (2, 3), (3, 12), (3, 11),
                            (12, 11), (3, 10), (11, 10), (10, 5), (10, 4), (5, 4), (5, 6), (6, 4), (4, 7), (7, 8),
                            (7, 9), (7, 1), (8, 9), (9, 1),
                            (13, 4), (13, 7), (13, 3), (13, 10), (13, 1)],
        interior=[],
    ),
    4: dict(
        polygon=[(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)],
        rects=dict(enumerate([(6, 8), (6, 6), (3, 7), (6, 4), (6, 2), (4, 6), (11, 4), (9, 2), (3, 8), (3, 3), (6, 4), (5, 5), (3, 9)])),
        required_adjacency=[(0, 3), (0, 5), (1, 2), (1, 4), (2, 6), (3, 4), (3, 5), (6, 8), (7, 8),
                             (9, 2), (9, 6), (9, 8), (9, 7), (9, 11), (10, 5), (10, 3), (10, 4), (10, 1), (10, 2), (10, 11), (11, 5), (11, 2), (11, 7), (11, 12), (12, 7)],
        interior=[9, 10],
    ),
    5: dict(
        polygon=[(0, 0), (20, 0), (20, 15), (10, 15), (10, 30), (0, 30)],
        rects=dict(enumerate([(6, 6), (11, 4), (1, 5), (2, 4), (9, 6), (6, 10), (5, 10), (9, 5), (10, 2), (4, 10), (5, 9), (3, 5), (4, 7)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0),
                             (9, 0), (9, 1), (9, 7), (9, 8), (9, 10), (9, 11), (10, 2), (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 1), (11, 2), (12, 3), (12, 4), (12, 5)],
        interior=[9, 10, 11, 12],
    ),
    6: dict(
        polygon=[(0, 0), (16, 0), (16, 6), (9, 6), (9, 14), (0, 14)],
        rects=dict(enumerate([(6, 4), (3, 3), (3, 4), (5, 3), (3, 8), (3, 6), (5, 2), (3, 4), (2, 3), (6, 2), (4, 5), (3, 1), (1, 3)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                             (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 7), (11, 8), (11, 9), (11, 12), (12, 3), (12, 9)],
        interior=[10, 11, 12],
    ),
    7: dict(
        polygon=[(0, 0), (16, 0), (16, 8), (6, 8), (6, 23), (0, 23)],
        rects=dict(enumerate([(3, 2), (3, 2), (6, 2), (3, 2), (3, 2), (4, 6),
                               (5, 6), (8, 5), (8, 3), (3, 2), (3, 2), (3, 4), (5, 4), (3, 4), (4, 2)])),
        required_adjacency=[
            (0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4),
            (1, 5), (2, 5), (4, 5), (5, 6), (6, 7), (6, 8),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8),
            (14, 9), (14, 10), (14, 12), (14, 13), (14, 8),
        ],
        interior=[14],
    ),
    8: dict(
        polygon=[(0, 7), (0, 13), (20, 13), (20, 0), (13, 0), (13, 7)],
        rects=dict(enumerate([(7, 4), (7, 3), (2, 3), (4, 4), (2, 3), (2, 2), (2, 5), (2, 2), (2, 4), (4, 5), (3, 4), (3, 4), (11, 1), (3, 3), (1, 2)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9), (9, 10), (10, 11),
                             (12, 7), (12, 9), (12, 6), (12, 5), (12, 4), (12, 3), (12, 10), (12, 11), (12, 13), (13, 3), (13, 0), (13, 11), (13, 14), (14, 3), (14, 2), (14, 0)],
        interior=[12, 14],
    ),
    9: dict(
        polygon=[(0, 0),(21,0),(21,6),(9,6),(9,18),(0,18)],
        rects=dict(enumerate([(6,8),(6,4),(2,2),(3,2),(4,5),(2,5),(3,5),(2,5),(3,3),(4,9), (6, 3), (4, 8), (2, 1)])),
        required_adjacency=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),
                             (10, 7), (10, 8), (10, 9), (10, 11), (11, 1), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 12), (12, 1), (12, 2), (12, 3)],
        interior=[12],
    ),
    10: dict(
        polygon=[(0, 0), (18, 0), (18, 13), (10, 13), (10, 24), (0, 24)],
        rects=dict(enumerate([(9, 6),(4, 4),(4, 4),(6, 4),(4, 4),(5, 6),(3, 4),(2, 4),(4, 4),(3, 4),(5, 5),(7, 7),(5, 3), (2, 7), (7, 2), (1, 3), (5, 2), (2, 5)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                             (13, 3), (13, 4), (13, 5), (13, 8), (13, 9), (13, 10), (13, 15), (14, 0), (14, 2), (14, 3), (14, 10), (14, 11), (14, 15), (14, 16), (15, 3), (15, 10), (16, 0), (16, 11), (16, 12), (16, 17), (17, 0), (17, 12)],
        interior=[13, 14, 15, 16],
    ),
    11: dict(
        # the 18-room plan with an 8x8 atrium hole; interior rooms 8, 14, 15, 17
        polygon=[(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)],
        rects=dict(enumerate([(10, 10), (10, 10), (10, 5), (10, 5), (10, 5), (10, 5), (10, 10), (10, 10), (10, 5), (10, 5), (10, 5), (10, 5), (20, 6), (6, 14), (4, 6), (4, 6), (6, 6), (6, 8)])),
        required_adjacency=[(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 13), (4, 14), (5, 15), (5, 16), (6, 7), (6, 8), (6, 16), (6, 17), (7, 9), (8, 9), (8, 10), (8, 12), (8, 17), (9, 11), (10, 11), (10, 12), (12, 13), (12, 17), (13, 14), (14, 15), (15, 16), (16, 17)],
        interior=[8, 14, 15, 17],
    ),
}


# --------------------------------------------------------------------------
# Interactive runner
# --------------------------------------------------------------------------

def run_example(
    choice: int,
    max_solutions: int = 5,
    time_limit: Optional[float] = 60.0,
    required_adjacency: Optional[Iterable[Tuple[int, int]]] = None,
):
    """
    `required_adjacency=None` uses the example's own PTPG. Pass an
    explicit iterable (including `[]`, no two rects may touch) to
    override it, or `False` to turn the adjacency constraint off entirely.
    """
    example = EXAMPLES[choice]
    polygon, rects = example["polygon"], example["rects"]
    if required_adjacency is False:
        required_adjacency = None
    elif required_adjacency is None:
        required_adjacency = example.get("required_adjacency")

    _solve_and_report(str(choice), polygon, rects, required_adjacency, max_solutions, time_limit,
                      interior_ids=example.get("interior", ()), labels=example.get("labels"))


def run_custom_example(max_solutions: int = 5, time_limit: Optional[float] = 60.0):
    """Prompt the user for their own polygon / rects / required_adjacency
    and run the same solve-and-report flow as `run_example`. Each value is
    typed as a plain Python literal and parsed with `ast.literal_eval`."""
    print("\nEnter your own floorplan. Each value is a Python literal, e.g.:")
    print("  polygon: [(0,0), (10,0), (10,10), (0,10)]")
    print("  rects:   {0: (4,4), 1: (6,4), 2: (10,6)}")
    print("  required_adjacency: [(0,1), (1,2)]   (or [] for none)")
    print("  interior rects (must not touch the boundary): [2]   (or [] for none)")
    print("  labels (optional): {0: 'Living', 1: 'Kitchen'}   (blank = show room numbers)")

    try:
        polygon = ast.literal_eval(input("\npolygon: ").strip())
        rects_raw = ast.literal_eval(input("rects: ").strip())
        rects = {int(k): tuple(v) for k, v in rects_raw.items()}
        adj_raw = input("required_adjacency (blank for none): ").strip()
        required_adjacency = ast.literal_eval(adj_raw) if adj_raw else []
        interior_raw = input("interior rects (blank for none): ").strip()
        interior_ids = ast.literal_eval(interior_raw) if interior_raw else []
        labels_raw = input("labels (blank for numbers): ").strip()
        labels = {int(k): str(v) for k, v in ast.literal_eval(labels_raw).items()} if labels_raw else None
    except (ValueError, SyntaxError) as exc:
        print(f"Could not parse that: {exc}")
        return

    _solve_and_report("custom", polygon, rects, required_adjacency, max_solutions, time_limit,
                      interior_ids=interior_ids, labels=labels)


def _solve_and_report(label, polygon, rects, required_adjacency, max_solutions, time_limit, interior_ids=(), labels=None):
    n_adj = len(required_adjacency or [])
    print(f"\nRunning example {label}: {len(rects)} rects, {n_adj} adjacencies")
    print(f"  polygon: {polygon}")
    print(f"  rects:   {rects}")
    if required_adjacency:
        print(f"  required_adjacency (PTPG): {sorted(required_adjacency)}")
    if interior_ids:
        print(f"  interior rects: {sorted(interior_ids)}")

    area_ok, area_msg = check_area_match(polygon, rects)
    if not area_ok:
        print(f"\n  infeasible: {area_msg}")
        return
    iso_ok, iso_msg = check_no_isolated(rects, required_adjacency, interior_ids)
    if not iso_ok:
        print(f"\n  infeasible: {iso_msg}")
        return

    t0 = time.perf_counter()
    found0, placements0 = None, None
    try:
        found0, placements0, nodes0 = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            interior_ids=interior_ids,
            time_limit=time_limit,
        )
        t1 = time.perf_counter()
        status = "FEASIBLE" if found0 else "INFEASIBLE" if found0 is False else "INCONCLUSIVE"
        print(f"\n  initial solve: {status}   nodes explored: {nodes0}   time: {t1 - t0:.3f}s")
        if found0 and placements0:
            overlaps0, outside0, holes0 = verify_placements(polygon, placements0)
            poly_area0 = _poly_area(polygon)
            coverage0 = sum(w * h for _, _, w, h in placements0.values()) / poly_area0 * 100
            print(f"    coverage={coverage0:.1f}%  overlaps={overlaps0}  outside={outside0}  holes={holes0}")
    except KeyboardInterrupt:
        print("\n  interrupted -- moving on to the multi-solution search")

    print(f"\n  searching for up to {max_solutions} distinct solutions... (Ctrl+C to stop early and keep what's found)")
    solutions: list = [placements0] if (found0 is True and placements0) else []
    interrupted = False
    try:
        found, solutions, nodes = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            interior_ids=interior_ids,
            time_limit=time_limit,
            collect_solutions=solutions,
            max_solutions=max_solutions,
        )
    except KeyboardInterrupt:
        interrupted = True
        found = None
        nodes = "?"

    if not solutions:
        if interrupted:
            print(f"  interrupted before finding any solution (nodes explored: {nodes})")
        elif found is False:
            print(f"  proven infeasible: no solution exists under these constraints (nodes explored: {nodes})")
        else:
            print(f"  hit time_limit ({time_limit}s) before finding any solution -- inconclusive (nodes explored: {nodes})")
        return

    if found is True:
        print(f"  found {len(solutions)} distinct solution(s), nodes explored: {nodes}")
    elif interrupted:
        print(f"  stopped by user -- showing {len(solutions)} of {max_solutions} distinct solution(s) found so far "
              f"(nodes explored: {nodes})")
    else:
        print(f"  hit time_limit ({time_limit}s) before finishing -- showing {len(solutions)} of {max_solutions} "
              f"distinct solution(s) found so far (nodes explored: {nodes})")

    poly_area = _poly_area(polygon)
    for i, sol in enumerate(solutions, start=1):
        overlaps, outside, holes = verify_placements(polygon, sol)
        coverage = sum(w * h for _, _, w, h in sol.values()) / poly_area * 100
        print(f"    solution {i}: coverage={coverage:.1f}%  overlaps={overlaps}  outside={outside}  holes={holes}")

    save_path = f"packing_multi_solutions_ex_{label}.png"
    render_solution_grid(polygon, solutions, save_path, labels=labels)
    print(f"  saved to {save_path}")


def main():
    print("Available examples:")
    for key, ex in EXAMPLES.items():
        n_adj = len(ex.get("required_adjacency") or [])
        print(f"  {key}. {len(ex['rects'])} rects, {n_adj} adjacencies")
    print("  c. enter your own custom example")

    raw = input(f"\nChoose an example [{min(EXAMPLES)}-{max(EXAMPLES)}] or 'c': ").strip()
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

    run_example(choice)


if __name__ == "__main__":
    main()