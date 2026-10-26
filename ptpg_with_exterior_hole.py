"""
Exact backtracking packer: places rectangles inside a rectilinear polygon
plot with no overlaps and no rectangle crossing the boundary.

`pack_rectangles_frontier` is the sole solver: a PTPG-driven frontier
search built around `required_adjacency` (a set of (id_a, id_b) pairs
over the same keys as `rects`, e.g. from a planar adjacency graph of
which rooms must touch). It grows outward from placed rectangles using
that graph to decide what to try next, instead of rescanning the whole
free boundary each step; see the "Frontier / PTPG-driven search" section
below.

Every candidate is anchored at a live vertex of the current free
polygon, and every placed rectangle must touch the plot boundary along
part of one edge (`require_boundary_touch`), except `interior_ids`
rects, which must not share any edge segment with it (touching at a
single point is allowed). A placement is rejected if
it fragments the remaining free space into more pieces than the
remaining rects can close, either by exact tiling or by being set aside
as the one leftover hole `require_single_hole` allows (see
`_fill_pockets_shortcut`). The final touch-graph must match
`required_adjacency` exactly, checked incrementally as each rect is
placed.

Rectangles are identified by explicit ids (`rects` is {id: (w, h)}), the
same ids used in `placements` and `required_adjacency`.

Multi-solution mode
--------------------
Passing `collect_solutions` (a list) keeps the search going after each
valid packing instead of stopping at the first one, returning several
distinct layouts (deduplicated by geometry).
"""
import ast
import time
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry


class _BudgetOut(Exception):
    """Internal: a root (first-placement) branch used up its node budget."""


class _SearchTimeout(Exception):
    """Raised by `_check_time` once `time_limit` is exceeded; caught once
    at the bottom of `pack_rectangles_frontier` for an immediate abort."""

Size = Tuple[float, float]
Placement = Tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _polygon_vertices(shape: BaseGeometry) -> List[Tuple[float, float]]:
    polys = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    pts = set()
    for p in polys:
        if p.is_empty:
            continue
        pts.update(p.exterior.coords[:-1])
        for interior in p.interiors:
            pts.update(interior.coords[:-1])
    return list(pts)


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


import shapely as _shp
from shapely.prepared import prep as _prep

_prep_cache = {}
_seg_cache = {}


def _prepared(shape):
    """Prepared version of `shape`, cached by identity (the shape is kept
    alive in the cache entry so its id cannot be reused)."""
    ent = _prep_cache.get(id(shape))
    if ent is None:
        if len(_prep_cache) > 512:
            _prep_cache.clear()
        ent = (shape, _prep(shape))
        _prep_cache[id(shape)] = ent
    return ent[1]


def _rect_fits_xywh(free_shape: BaseGeometry, x: float, y: float, w: float, h: float) -> bool:
    """True if the rectangle lies inside `free_shape` (1e-9 tolerance)."""
    if free_shape.is_empty:
        return False
    fminx, fminy, fmaxx, fmaxy = free_shape.bounds
    if x < fminx - 1e-9 or y < fminy - 1e-9 or x + w > fmaxx + 1e-9 or y + h > fmaxy + 1e-9:
        return False
    e = 1e-9
    return bool(_prepared(free_shape).covers(_shp.box(x + e, y + e, x + w - e, y + h - e)))


def _rect_fits(free_shape: BaseGeometry, candidate: BaseGeometry) -> bool:
    x0, y0, x1, y1 = candidate.bounds
    return _rect_fits_xywh(free_shape, x0, y0, x1 - x0, y1 - y0)


def _boundary_segments(boundary):
    """(horizontal, vertical) segments of a rectilinear boundary as
    (const, lo, hi) tuples, or None if it has a slanted segment."""
    ent = _seg_cache.get(id(boundary))
    if ent is not None:
        return ent[1]
    hs, vs = [], []
    ok = True
    geoms = boundary.geoms if hasattr(boundary, "geoms") else [boundary]
    for g in geoms:
        pts = list(g.coords)
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            if abs(ay - by) < 1e-12:
                hs.append((ay, min(ax, bx), max(ax, bx)))
            elif abs(ax - bx) < 1e-12:
                vs.append((ax, min(ay, by), max(ay, by)))
            else:
                ok = False
    res = (hs, vs) if ok else None
    _seg_cache[id(boundary)] = (boundary, res)
    return res


def _shares_boundary_edge(rx: float, ry: float, w: float, h: float, boundary, tol: float = 1e-9) -> bool:
    """True if any edge of the rectangle overlaps the plot boundary by more than a point."""
    segs = _boundary_segments(boundary)
    if segs is None:
        edges = [
            LineString([(rx, ry), (rx + w, ry)]),
            LineString([(rx, ry + h), (rx + w, ry + h)]),
            LineString([(rx, ry), (rx, ry + h)]),
            LineString([(rx + w, ry), (rx + w, ry + h)]),
        ]
        return any(edge.length >= tol and edge.intersection(boundary).length > tol for edge in edges)
    hs, vs = segs
    for yy in (ry, ry + h):
        tot = 0.0
        for (c, lo, hi) in hs:
            if abs(c - yy) < tol:
                ov = min(rx + w, hi) - max(rx, lo)
                if ov > 0:
                    tot += ov
        if tot > tol:
            return True
    for xx in (rx, rx + w):
        tot = 0.0
        for (c, lo, hi) in vs:
            if abs(c - xx) < tol:
                ov = min(ry + h, hi) - max(ry, lo)
                if ov > 0:
                    tot += ov
        if tot > tol:
            return True
    return False


def _touches_boundary_any(rx: float, ry: float, w: float, h: float, boundary, tol: float = 1e-9) -> bool:
    """True if the CLOSED rectangle meets the plot boundary anywhere, even at
    a single point (a corner resting on a wall or reflex corner counts)."""
    segs = _boundary_segments(boundary)
    if segs is None:
        return box(rx, ry, rx + w, ry + h).intersects(boundary)
    hs, vs = segs
    x1, y1 = rx + w, ry + h
    for (c, lo, hi) in hs:
        if ry - tol <= c <= y1 + tol and lo <= x1 + tol and hi >= rx - tol:
            return True
    for (c, lo, hi) in vs:
        if rx - tol <= c <= x1 + tol and lo <= y1 + tol and hi >= ry - tol:
            return True
    return False


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


def _num_components(shape: BaseGeometry, robust_tol: float = 1e-4) -> int:
    """Connected pieces of free space, robust to hairline slivers from float noise."""
    if shape.is_empty:
        return 0
    eroded = shape.buffer(-robust_tol)
    if eroded.is_empty:
        return 0
    return len(eroded.geoms) if eroded.geom_type == "MultiPolygon" else 1


def _free_space_components(shape: BaseGeometry):
    """Connected pieces of free space (exact boundaries), largest first."""
    if shape.is_empty:
        return []
    comps = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    return sorted(comps, key=lambda c: -c.area)


def _subset_sum_ok(areas, target, tol=1e-6):
    """Can some subset of `areas` sum to `target`? Scaled to integers
    (areas are products of short decimals); falls back to True if the
    scaling would be huge, so it can only ever prune, never wrongly cut."""
    scale = 1
    for k in range(7):
        if all(abs(a * scale - round(a * scale)) < 1e-6 for a in list(areas) + [target]):
            break
        scale *= 10
    else:
        return True
    t = round(target * scale)
    if t > 2_000_000:
        return True
    reach = 1
    mask = (1 << (t + 1)) - 1
    for a in areas:
        reach |= (reach << round(a * scale)) & mask
    return bool((reach >> t) & 1)


def _convex_vertices(poly):
    """Convex corners of a Polygon's exterior (free-space side), sorted
    lowest-then-leftmost first."""
    from shapely.geometry.polygon import orient
    c = list(orient(poly, 1.0).exterior.coords)[:-1]
    n = len(c)
    out = []
    for i in range(n):
        a, b, d = c[i - 1], c[i], c[(i + 1) % n]
        if (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]) > 1e-9:
            out.append(b)
    return sorted(out, key=lambda q: (q[1], q[0]))


def _best_subset_sum(areas, cap):
    """Largest sum <= cap reachable by a subset of `areas` (scaled to
    integers; returns `cap` -- i.e. no useful bound -- if scaling is
    awkward or huge, so it can only weaken a prune, never wrongly cut)."""
    scale = 1
    for _ in range(7):
        if all(abs(a * scale - round(a * scale)) < 1e-6 for a in list(areas) + [cap]) or abs(cap * scale - round(cap * scale)) < 1e-6 and all(abs(a * scale - round(a * scale)) < 1e-6 for a in areas):
            break
        scale *= 10
    else:
        return cap
    c = int(cap * scale + 1e-6)
    if c > 2_000_000:
        return cap
    reach, mask = 1, (1 << (c + 1)) - 1
    for a in areas:
        reach |= (reach << round(a * scale)) & mask
    return (reach.bit_length() - 1) / scale


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


_SLIDE = True               # last-resort pass, used only after the plain search says infeasible (triangle shortcut: partner slid along a free edge in whole grid steps (gap becomes a hole))
_LOCAL_INTERIOR_READY = True   # an interior room becomes placeable once its own border neighbours are placed
_DEFER_INTERIOR = True       # False = no border-rooms-first rule: interior rooms are ordinary candidates from the start
_PAIR_GAP_CHECK = True      # unplaced room required to touch two placed rooms must be able to bridge the gap between them
_SEALED_CHECK = False       # placed room must keep enough free edge for its unplaced required neighbours
_SURROUND_CHECK = True       # interior rect with all its neighbours placed but an edge not fully covered -> backtrack
_DEAD_CORNER_CHECK = False  # prune when an uncovered convex plot corner can't be covered and no hole budget is left
_MEMO = False               # remember states already proven to fail
_PRUNE_ANCHORS = True       # drop anchors that can never place anything
_ROOT_BUDGET = 1500         # nodes per first placement in round 1 (doubles each round); None = no restarts
_NODE_HOOK = None   # debug: called as hook(node_number, placements) on every search node
_ALL_CORNER_ANCHORS = True   # False = old rule (only the corners adjacent to the anchor)


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
    if _ALL_CORNER_ANCHORS and free_shape is not None:
        # every corner of a newly placed rect becomes an anchor point (the
        # one it was anchored at too, so other partners can still use it),
        # plus live vertices lying partway along its edges
        result = list(corners)
        for (vx, vy) in _polygon_vertices(free_shape):
            if any(abs(vx - cx) < 1e-9 and abs(vy - cy) < 1e-9 for (cx, cy) in corners):
                continue
            if _point_on_rect_edge(vx, vy, rx, ry, w, h):
                result.append((vx, vy))
        return result
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
    if any(abs(diagonal[0] - vx) < 1e-9 and abs(diagonal[1] - vy) < 1e-9 for (vx, vy) in live_verts):
        result.append(diagonal)
    for (vx, vy) in live_verts:
        if any(abs(vx - cx) < 1e-9 and abs(vy - cy) < 1e-9 for (cx, cy) in corners):
            continue
        if _point_on_rect_edge(vx, vy, rx, ry, w, h):
            result.append((vx, vy))
    return result


def _reflex_corner(box_a: Placement, box_b: Placement):
    """The inner (concave) corner of the L-union of two touching,
    perpendicular rectangles. None if they're collinear (no reflex corner)."""
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
    for (cx, cy) in corners:
        if xmin + 1e-9 < cx < xmax - 1e-9 and ymin + 1e-9 < cy < ymax - 1e-9:
            return (cx, cy)
    return None


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


def _anchor_points_on_line(free_shape, axis: str, value: float, tol: float = 1e-6):
    """Live vertices of the free polygon lying on x=value or y=value."""
    verts = _polygon_vertices(free_shape)
    idx = 0 if axis == "x" else 1
    return [p for p in verts if abs(p[idx] - value) < tol]


from itertools import combinations as _combinations


def pack_rectangles_frontier(
    polygon,
    rects: Dict[int, Size],
    required_adjacency=None,
    *,
    require_single_hole: bool = True,
    max_holes: int = 1,
    holes_touch_corner: bool = True,
    require_boundary_touch: bool = False,
    interior_ids: Iterable[int] = (),
    time_limit: Optional[float] = 60.0,
    collect_solutions: Optional[list] = None,
    max_solutions: Optional[int] = None,
):
    """PTPG-driven frontier search. Returns (found, placements_or_None,
    nodes_explored). found is True / False / None (feasible / proven
    infeasible / cut off by time_limit).

    `interior_ids` are rects that must not share an edge segment with the
    plot boundary (touching at a single point is allowed). Every other
    ("boundary") rect follows `require_boundary_touch`:
      * True  -- it must share an edge segment with the plot boundary (the
                 old strict rule);
      * False -- (default) it MAY be away from the boundary, as long as it
                 is not completely enclosed: at least one of its four edges
                 must be left (partly) uncovered by other rects, i.e. it is
                 exposed to the plot boundary or to a hole. Candidate
                 placements that touch the boundary are tried first.

    `time_limit` (seconds, wall-clock; None = unbounded) is checked via
    `_check_time` in every hot loop, not just `solve()`, since real work
    (a vertex scan, `_tile_pocket`'s own search) can happen without
    calling `solve()` again in between. Exceeding it raises
    `_SearchTimeout`, caught once at the bottom of this function.

    `required_adjacency` (an iterable of (id_a, id_b) pairs over the same
    ids as `rects`) is optional -- `None`/`[]` means no adjacency
    constraint (every rect is its own region, re-seeded independently;
    `require_single_hole`/`require_boundary_touch` still apply).

    Leftover space ("holes"): when `require_single_hole` is True (name kept
    for compatibility) the final layout may leave up to `max_holes`
    (default 4) separate empty regions, and -- if `holes_touch_corner` is
    True -- EVERY such hole must touch at least one vertex of the plot
    polygon (convex or reflex corner). Total room area <= plot area still
    holds, so the holes' areas add up to plot area - total room area.

    Pass `collect_solutions` (a list) to keep searching past the first
    valid placement: each is appended and the search backtracks for a
    genuinely different one, up to `max_solutions` (None = exhaustive).
    In this mode the second return value is always `collect_solutions`
    itself."""
    sizes = dict(rects)
    boundary = Polygon(polygon).boundary
    poly = Polygon(polygon)
    inner_l_points = _polygon_reflex_vertices(polygon)
    required_adjacency = required_adjacency or []
    interior = set(interior_ids)
    unknown = interior - set(sizes)
    if unknown:
        raise ValueError(f"interior_ids references unknown rect ids: {sorted(unknown)}")

    # grid for slid placements: gcd of every side length and plot coordinate
    from math import gcd as _gcd
    _vals = [abs(v) for wh in rects.values() for v in wh] + [abs(c) for p in polygon for c in p]
    try:
        _ints = [int(round(v)) for v in _vals]
        grid_step = 0
        if all(abs(v - r) < 1e-9 for v, r in zip(_vals, _ints)):
            for _r in _ints:
                grid_step = _gcd(grid_step, _r)
    except Exception:
        grid_step = 0
    _bx0, _by0, _bx1, _by1 = Polygon(polygon).bounds
    slide_max_steps = int(max(_bx1 - _bx0, _by1 - _by0) // grid_step) if grid_step else 0
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

    # A rect with exactly 2 required partners who are themselves required
    # partners of each other (a triangle in the adjacency graph) must sit
    # at the reflex/convex corner of that pair's L-union -- it has nowhere
    # else it could go. Sufficient, not necessary, for being a "corner"
    # rect (plenty of real corners aren't degree-2 triangles), so this is
    # only used to try these rects earlier in `seed_step`, not as a filter.
    triangle_corner_ids = {
        i for i, ps in required_partners.items()
        if len(ps) == 2 and ps[1] in required_partners.get(ps[0], [])
    }

    placements: Dict[int, Placement] = {}
    nodes = [0]
    deadline = time.perf_counter() + time_limit if time_limit is not None else None

    def _check_time():
        if deadline is not None and time.perf_counter() > deadline:
            raise _SearchTimeout()

    # The one pocket (if any) this branch has set aside as the final
    # leftover hole `require_single_hole` allows -- see
    # `_fill_pockets_shortcut`. Reset to None whenever that choice is undone.
    holes = []          # frozen leftover regions of this branch (shapes)
    plot_vertices = _polygon_vertices(poly)

    def _parts(shape):
        return list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]

    def hole_count():
        return sum(len(_parts(h)) for h in holes)

    def _touches_corner(shape) -> bool:
        """True if `shape` touches at least one plot vertex (convex or
        reflex). Free space only shrinks, so a region that does not touch a
        corner now never will: usable as a pruning test."""
        if not holes_touch_corner:
            return True
        return any(shape.distance(Point(v)) < 1e-7 for v in plot_vertices)

    def required_adjacency_ok(idx, rx, ry, w, h) -> bool:
        cand = (rx, ry, w, h)
        partners = required_partners[idx]
        for j, placed in placements.items():
            if _rects_touch(cand, placed) != (j in partners):
                return False
        return True

    def single_hole_ok(new_free) -> bool:
        """Cheap necessary pre-filter, not the real tiling check: a
        placement that splits off pockets is still viable if each pocket
        can be exactly tiled by the remaining rects (see `_tile_pocket`).
        Running that real search here would be paid on every candidate
        examined anywhere in the outer search, so this only checks area
        and minimum-dimension feasibility; the sufficient check is
        deferred to `_fill_pockets_shortcut`, run once a placement is
        actually committed.

        `require_single_hole` only requires the FINAL layout to have at
        most one leftover region, not that every pocket that splits off
        mid-search gets tiled -- so if this branch hasn't claimed its one
        allowed hole slot yet (`hole_count() < max_holes`), a failing pocket is
        still a live option. Only a second simultaneous failure is a sure
        reject."""
        if not require_single_hole:
            return True
        comps = _free_space_components(new_free)
        if len(comps) <= 1:
            return True
        remaining = remaining_ids()
        total_remaining_area = sum(sizes[i][0] * sizes[i][1] for i in remaining) if remaining else 0
        min_side = min(min(sizes[i]) for i in remaining) if remaining else 0
        freebies = max(0, max_holes - hole_count())
        for pocket in comps[1:]:
            ok = bool(remaining) and pocket.area <= total_remaining_area + 1e-6
            if ok:
                minx, miny, maxx, maxy = pocket.bounds
                if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
                    ok = False
            if not ok:
                # a pocket that cannot be filled must stay a hole: it needs a
                # free hole slot AND a plot corner to touch
                if freebies > 0 and _touches_corner(pocket):
                    freebies -= 1
                    continue
                return False
        return True

    def _edge_cover_lengths(r, others) -> list:
        """For rect r=(x,y,w,h): per edge (bottom, top, left, right), True if
        the rects in `others` that touch that edge cover all of it."""
        x, y, w, h = r
        cov = [0.0, 0.0, 0.0, 0.0]
        for (ox, oy, ow, oh) in others:
            ovx = min(x + w, ox + ow) - max(x, ox)
            ovy = min(y + h, oy + oh) - max(y, oy)
            if ovx > 1e-9:
                if abs(oy + oh - y) < 1e-9:
                    cov[0] += ovx
                if abs(oy - (y + h)) < 1e-9:
                    cov[1] += ovx
            if ovy > 1e-9:
                if abs(ox + ow - x) < 1e-9:
                    cov[2] += ovy
                if abs(ox - (x + w)) < 1e-9:
                    cov[3] += ovy
        return cov

    def _edge_covered_fully(r, others) -> list:
        """Per edge (bottom, top, left, right): fully covered by `others`?"""
        x, y, w, h = r
        lens = [w, w, h, h]
        cov = _edge_cover_lengths(r, others)
        return [cov[k] >= lens[k] - 1e-9 for k in range(4)]

    def interior_surrounded_ok(final=False) -> bool:
        """Inner-room rules, checked on every node (all sound: coverage of an
        edge or corner only grows).
        1. A designated interior rect must end up inner: all 4 edges fully
           covered and all 4 corners covered by rects. Once all rooms it must
           touch are placed an uncovered edge is a dead end; before that the
           uncovered edge length must still be coverable. A corner whose
           diagonal quadrant lies outside the plot or in a frozen hole is a
           dead end too; other uncovered corners are judged when `final`.
        2. A NON-designated rect that is inner (see `_is_inner`) is a rect
           the input did not declare interior: dead end."""
        for j, r in placements.items():
            others = [q for k, q in placements.items() if k != j]
            if j in interior:
                covlen = _edge_cover_lengths(r, others)
                x, y, w, h = r
                lens = [w, w, h, h]
                if not all(covlen[k] >= lens[k] - 1e-9 for k in range(4)):
                    un = [p for p in required_partners.get(j, []) if p not in placements]
                    if not un:
                        return False
                    missing = sum(max(0.0, lens[k] - covlen[k]) for k in range(4))
                    reach = sum(min(max(w, h), max(sizes[p])) for p in un)
                    if missing > reach + 1e-9:
                        return False
                cc = _corners_covered(r, others)
                if not all(cc):
                    corners = ((x, y, -1, -1), (x + w, y, 1, -1), (x, y + h, -1, 1), (x + w, y + h, 1, 1))
                    for k, (cx, cy, sx, sy) in enumerate(corners):
                        if cc[k]:
                            continue
                        pt = Point(cx + sx * _EPS, cy + sy * _EPS)
                        if final or not poly.contains(pt) or any(hh.distance(pt) < 1e-9 for hh in holes):
                            return False
            elif _is_inner(r, others):
                return False
        return True

    def _boundary_overlap_len(r) -> float:
        """Total length of the rect's edges lying on the plot boundary."""
        segs = _boundary_segments(boundary)
        if segs is None:
            return 0.0
        hs, vs = segs
        x, y, w, h = r
        tot = 0.0
        for yy in (y, y + h):
            for (c, lo, hi) in hs:
                if abs(c - yy) < 1e-9:
                    tot += max(0.0, min(x + w, hi) - max(x, lo))
        for xx in (x, x + w):
            for (c, lo, hi) in vs:
                if abs(c - xx) < 1e-9:
                    tot += max(0.0, min(y + h, hi) - max(y, lo))
        return tot

    def sealed_ok() -> bool:
        """A placed rect with u unplaced required neighbours needs u disjoint
        contacts of at least one grid unit each along edges that are still
        free (not covered by a rect, not on the plot wall). Less free edge
        than that can never recover (coverage only grows): dead end."""
        if grid_step <= 0:
            return True
        rem = remaining_ids()
        if not rem:
            return True
        remset = set(rem)
        for a, r in placements.items():
            u = sum(1 for p in required_partners.get(a, []) if p in remset)
            if not u:
                continue
            others = [q for k, q in placements.items() if k != a]
            cov = sum(_edge_cover_lengths(r, others))
            free_len = 2 * (r[2] + r[3]) - cov - _boundary_overlap_len(r)
            if free_len < u * grid_step - 1e-9:
                return False
        return True

    def pair_gap_ok() -> bool:
        """An unplaced rect j required to touch two placed rects a and b must
        be able to bridge them: along each axis its extent must cover the gap
        between a and b (in one of its two orientations)."""
        rem = remaining_ids()
        if not rem:
            return True
        for j in rem:
            pn = [p for p in required_partners.get(j, []) if p in placements]
            if len(pn) < 2:
                continue
            w0, h0 = sizes[j]
            for ia in range(len(pn)):
                ax, ay, aw, ah = placements[pn[ia]]
                for ib in range(ia + 1, len(pn)):
                    bx, by, bw, bh = placements[pn[ib]]
                    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
                    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
                    if not ((w0 >= dx - 1e-9 and h0 >= dy - 1e-9) or (h0 >= dx - 1e-9 and w0 >= dy - 1e-9)):
                        return False
        return True

    _EPS = 1e-6

    def _point_in_rects(px, py, rects_iter) -> bool:
        for (ox, oy, ow, oh) in rects_iter:
            if ox + 1e-9 < px < ox + ow - 1e-9 and oy + 1e-9 < py < oy + oh - 1e-9:
                return True
        return False

    def _corners_covered(r, others) -> list:
        """For each corner of r: is the quadrant diagonally outward from it
        covered by some other rect? (The two edge-adjacent quadrants are
        covered once the edges are.) A corner whose diagonal quadrant is
        free space, a hole or outside the plot is still touched by it."""
        x, y, w, h = r
        res = []
        for (cx, cy, sx, sy) in ((x, y, -1, -1), (x + w, y, 1, -1), (x, y + h, -1, 1), (x + w, y + h, 1, 1)):
            res.append(_point_in_rects(cx + sx * _EPS, cy + sy * _EPS, others))
        return res

    def _is_inner(r, others) -> bool:
        """New definition: a rect is inner only if no point of it touches the
        plot boundary or free space: it touches no wall even at a point, all 4
        edges are fully covered by rects, and so are all 4 corners."""
        if _touches_boundary_any(r[0], r[1], r[2], r[3], boundary):
            return False
        if not all(_edge_covered_fully(r, others)):
            return False
        return all(_corners_covered(r, others))

    def _enclosed(idx, r, extra=None) -> bool:
        """True if non-interior rect `idx` at `r` is (already, permanently)
        an inner rect: not touching the plot boundary even at a point, edges
        and corners fully covered by placed rects (plus `extra`). Coverage
        only grows, so this is a sound prune for an undeclared room."""
        if idx in interior:
            return False
        others = [p for j, p in placements.items() if j != idx]
        if extra is not None:
            others.append(extra)
        return _is_inner(r, others)

    def boundary_ok(idx, rx, ry, w, h) -> bool:
        if idx in interior:
            # an inner room touches the plot boundary nowhere, not even at a point
            return not _touches_boundary_any(rx, ry, w, h, boundary)
        if require_boundary_touch:
            return _shares_boundary_edge(rx, ry, w, h, boundary)
        # loosened rule: the new rect must not already be an undeclared inner
        # room, and placing it must not turn a placed boundary room into one
        cand = (rx, ry, w, h)
        if _enclosed(idx, cand, None):
            return False
        for j, p in placements.items():
            if j == idx or j in interior:
                continue
            if _rects_touch(cand, p) and _enclosed(j, p, cand):
                return False
        return True

    def placements_of(idx, point, free_shape, check_hole=True, slide_only=False):
        """All valid (rx, ry, w, h, new_free) placements of rect `idx`
        anchored at `point`, both orientations, all 4 quadrant offsets.
        `check_hole=False` skips single_hole_ok: `_tile_pocket` already
        recurses into whatever sub-pieces a placement leaves, so that
        filter would just redo the same search. Cheap plain-Python checks
        (`boundary_ok`, `required_adjacency_ok`) run before the shapely
        `.difference()` that only `single_hole_ok` needs."""
        px, py = point
        w0, h0 = sizes[idx]
        out = []
        do_slide = (slide_only and check_hole and require_single_hole and grid_step > 0
                    and max_holes - hole_count() >= 1)
        seen_pos = set()
        slides = []
        for uw, uh in {(w0, h0), (h0, w0)}:
            for ox, oy in _candidate_offsets(uw, uh):
                rx, ry = px + ox, py + oy
                if not _rect_fits_xywh(free_shape, rx, ry, uw, uh):
                    continue
                seen_pos.add((round(rx, 6), round(ry, 6), uw, uh))
                if slide_only:
                    # the anchored position itself was already tried; only
                    # collect the slid variants of it
                    pass
                if do_slide:
                    # same rect pushed along the free edge in whole grid steps,
                    # leaving a gap behind it (must end up a corner-touching hole)
                    for k in range(1, slide_max_steps + 1):
                        d = k * grid_step
                        for sx, sy in ((d, 0), (-d, 0), (0, d), (0, -d)):
                            slides.append((rx + sx, ry + sy, uw, uh))
                if slide_only:
                    continue
                if not boundary_ok(idx, rx, ry, uw, uh):
                    continue
                if not required_adjacency_ok(idx, rx, ry, uw, uh):
                    continue
                new_free = free_shape.difference(_shp.box(rx, ry, rx + uw, ry + uh))
                if check_hole and not single_hole_ok(new_free):
                    continue
                out.append((rx, ry, uw, uh, new_free))
        # slid placements come after every aligned one
        for (rx, ry, uw, uh) in slides:
            key = (round(rx, 6), round(ry, 6), uw, uh)
            if key in seen_pos:
                continue
            seen_pos.add(key)
            if not required_adjacency_ok(idx, rx, ry, uw, uh):
                continue
            if not _rect_fits_xywh(free_shape, rx, ry, uw, uh):
                continue
            if not boundary_ok(idx, rx, ry, uw, uh):
                continue
            new_free = free_shape.difference(_shp.box(rx, ry, rx + uw, ry + uh))
            if not single_hole_ok(new_free):
                continue
            out.append((rx, ry, uw, uh, new_free))
        if not require_boundary_touch:
            # try placements that touch the plot boundary first
            out.sort(key=lambda o: 0 if _shares_boundary_edge(o[0], o[1], o[2], o[3], boundary) else 1)
        return out

    def _tile_pocket(pocket_shape, depth_limit=None):
        """Exactly close `pocket_shape` with remaining rects, or return
        None (everything rolled back). Returns the forced ids on success,
        left applied to `placements`.

        Corner-cover branching: the lowest-then-leftmost vertex of the
        pocket's exterior is always a convex corner with free space only
        in its upper-right quadrant, so in ANY exact tiling exactly one
        rect has its lower-left corner there. Branching on just that
        corner (rect x orientation) is complete, replacing the old scan
        over every vertex x rect x quadrant (which re-found the same
        tilings from every vertex). Each placement is a plain rectangle
        test; sub-pieces recurse with no depth cap, since each level
        places a rect so depth <= number of remaining rects."""
        if pocket_shape.is_empty or pocket_shape.area < 1e-9:
            return []
        remaining = remaining_ids()
        if not remaining:
            return None
        pminx, pminy, pmaxx, pmaxy = pocket_shape.bounds
        pw, ph = pmaxx - pminx, pmaxy - pminy
        rem_area = sum(sizes[i][0] * sizes[i][1] for i in remaining)
        if rem_area < pocket_shape.area - 1e-6:
            return None
        fitting_ids = [
            i for i in remaining
            if (sizes[i][0] <= pw + 1e-6 and sizes[i][1] <= ph + 1e-6)
            or (sizes[i][1] <= pw + 1e-6 and sizes[i][0] <= ph + 1e-6)
        ]
        if not fitting_ids:
            return None
        # exact area cover needs a subset of fitting rects summing to the pocket area
        if not _subset_sum_ok([sizes[i][0] * sizes[i][1] for i in fitting_ids], pocket_shape.area):
            return None
        poly0 = max(pocket_shape.geoms, key=lambda g: g.area) if pocket_shape.geom_type == "MultiPolygon" else pocket_shape
        vx, vy = min(poly0.exterior.coords[:-1], key=lambda c: (c[1], c[0]))
        for idx in sorted(fitting_ids, key=lambda i: -(sizes[i][0] * sizes[i][1])):
            w0, h0 = sizes[idx]
            for uw, uh in {(w0, h0), (h0, w0)}:
                _check_time()
                if not _rect_fits(pocket_shape, box(vx, vy, vx + uw, vy + uh)):
                    continue
                if not boundary_ok(idx, vx, vy, uw, uh):
                    continue
                if not required_adjacency_ok(idx, vx, vy, uw, uh):
                    continue
                new_pocket = pocket_shape.difference(box(vx, vy, vx + uw, vy + uh))
                placements[idx] = (vx, vy, uw, uh)
                forced_here = [idx]
                ok = True
                for comp in _free_space_components(new_pocket):
                    sub_forced = _tile_pocket(comp)
                    if sub_forced is None:
                        ok = False
                        break
                    forced_here.extend(sub_forced)
                if ok:
                    return forced_here
                for f_idx in forced_here:
                    placements.pop(f_idx, None)
        return None

    _partial_budget = [0]

    def _tile_partial(piece, skipped, budget=1):
        """Tile a free `piece` (one connected component) with remaining
        rects while leaving at most `budget` connected leftovers (holes),
        each of which must touch a plot corner and which together contain
        every point in `skipped`. Returns (forced_ids, hole_shape) with the
        rects left applied to `placements`, or None with everything rolled
        back. `hole_shape` is None only if the piece ended up fully tiled;
        otherwise it is a Polygon/MultiPolygon with <= budget parts.

        Decision order is the lowest-leftmost convex corner `u` not yet in
        a hole: either a rect has its corner at `u` (every layout covering
        `u` does, since `u` is a convex corner of the free piece), or `u`
        lies in a hole ("skip"). That is complete for every packing of the
        piece with <= budget holes. After a rect goes down the piece may
        split: components holding hole points stay partial (their count
        must fit the budget); every other component is tiled exactly
        (`_tile_pocket`) or, budget permitting, becomes a new hole. A work
        cap (`_partial_budget`) hands back None if exhausted; the caller
        then falls back to freezing the whole pocket as the hole."""
        _check_time()
        _partial_budget[0] += 1
        if _partial_budget[0] > 3000:
            return None
        if budget < 1:
            return None
        if skipped and not _touches_corner(piece):
            return None
        cvs = [v for v in _convex_vertices(piece) if v not in skipped]
        if not cvs:
            return ([], piece) if skipped else None
        ux, uy = cvs[0]
        remaining = remaining_ids()
        for idx in sorted(remaining, key=lambda i: -(sizes[i][0] * sizes[i][1])):
            w0, h0 = sizes[idx]
            for uw, uh in {(w0, h0), (h0, w0)}:
                for ox, oy in _candidate_offsets(uw, uh):
                    rx, ry = ux + ox, uy + oy
                    cand = box(rx, ry, rx + uw, ry + uh)
                    if not _rect_fits(piece, cand):
                        continue
                    if any(cand.covers(Point(sv)) for sv in skipped):
                        continue
                    if not boundary_ok(idx, rx, ry, uw, uh):
                        continue
                    if not required_adjacency_ok(idx, rx, ry, uw, uh):
                        continue
                    placements[idx] = (rx, ry, uw, uh)
                    comps = [c for c in _free_space_components(piece.difference(cand)) if c.area > 1e-9]
                    with_hole = [k for k, c in enumerate(comps) if any(c.distance(Point(sv)) < 1e-7 for sv in skipped)]
                    if len(with_hole) > budget:
                        del placements[idx]
                        continue
                    others = [k for k in range(len(comps)) if k not in with_hole]
                    max_extra = budget - len(with_hole)
                    for n_extra in range(0, min(max_extra, len(others)) + 1):
                        for extra in _combinations(others, n_extra):
                            H = with_hole + list(extra)
                            if skipped and not H:
                                continue
                            forced = [idx]
                            ok = True
                            for k, c in enumerate(comps):
                                if k in H:
                                    continue
                                sub = _tile_pocket(c)
                                if sub is None:
                                    ok = False
                                    break
                                forced.extend(sub)
                            hole_parts = []
                            if ok:
                                used = 0
                                need_after = len(with_hole)
                                for k in H:
                                    c = comps[k]
                                    is_req = k in with_hole
                                    if is_req:
                                        need_after -= 1
                                    b = budget - used - need_after
                                    if b < 1:
                                        ok = False
                                        break
                                    sk = [sv for sv in skipped if c.distance(Point(sv)) < 1e-7]
                                    part = _tile_partial(c, sk, b)
                                    if part is None:
                                        ok = False
                                        break
                                    forced.extend(part[0])
                                    if part[1] is not None:
                                        hole_parts.append(part[1])
                                        used += len(_parts(part[1]))
                            if ok:
                                hole = None
                                if hole_parts:
                                    from shapely.ops import unary_union as _uu
                                    hole = _uu(hole_parts)
                                return forced, hole
                            for f in forced:
                                placements.pop(f, None)
                            placements[idx] = (rx, ry, uw, uh)
                    del placements[idx]
        # `u` is in a hole
        return _tile_partial(piece, list(skipped) + [(ux, uy)], budget)

    import os as _os_dbg
    _TILE_DEBUG = bool(_os_dbg.environ.get("TILE_DEBUG"))
    _fp_calls = [0]

    def _tile_one_pocket_logged(pocket):
        _fp_calls[0] += 1
        t0 = time.perf_counter()
        sub_forced = _tile_pocket(pocket)
        if _TILE_DEBUG:
            print(f"[fill_pockets] call#{_fp_calls[0]} pocket_area={pocket.area:.2f} "
                  f"remaining={len(remaining_ids())} time_used={time.perf_counter() - t0:.3f}s "
                  f"result={'FAIL' if sub_forced is None else sub_forced}",
                  file=__import__('sys').stderr)
        return sub_forced

    def _fill_pockets_shortcut(free_shape):
        """A placement can split free space into a main piece plus small
        pockets. Yields every viable way to resolve them, best first, as
        (forced_ids, active_free_shape) pairs:

          1. Tile every pocket exactly via `_tile_pocket`.
          2. Otherwise, while hole slots remain (`max_holes`), pick a subset
             S of pockets (smallest subsets first) that each touch a plot
             corner and become holes: each pocket in S is either filled as
             far as possible leaving connected holes (`_tile_partial`) or
             frozen whole; every other pocket is tiled exactly.

        Mutates `placements`/`holes` to match whichever alternative is
        currently yielded; a caller that keeps looping past a failed
        alternative triggers this generator's own cleanup (after the
        `yield`) before producing the next one."""
        comps = _free_space_components(free_shape)
        if len(comps) <= 1:
            yield [], free_shape
            return
        main = comps[0]
        pockets = comps[1:]

        def tile_all(skip_set=()):
            forced = []
            for i, pocket in enumerate(pockets):
                if i in skip_set:
                    continue
                sub_forced = _tile_one_pocket_logged(pocket)
                if sub_forced is None:
                    for f_idx in forced:
                        del placements[f_idx]
                    return None
                forced.extend(sub_forced)
            return forced, main

        result = tile_all()
        if result is not None:
            forced, active = result
            yield forced, active
            for f_idx in forced:
                del placements[f_idx]

        slots = max_holes - hole_count()
        if slots <= 0:
            return

        def resolve(plist, pos, slots_left):
            """Yield forced-id lists; holes appended to `holes` while yielded."""
            if pos == len(plist):
                yield []
                return
            pocket = plist[pos]
            need_after = len(plist) - pos - 1
            # (a) fill the pocket as far as possible, leaving connected holes
            _partial_budget[0] = 0
            part = _tile_partial(pocket, [], slots_left - need_after)
            if part is not None and part[1] is not None and part[0]:
                holes.append(part[1])
                k = len(_parts(part[1]))
                for rest in resolve(plist, pos + 1, slots_left - k):
                    yield part[0] + rest
                holes.pop()
                for f_idx in part[0]:
                    placements.pop(f_idx, None)
            elif part is not None:
                for f_idx in part[0]:
                    placements.pop(f_idx, None)
            # (b) freeze the whole pocket as a hole
            if slots_left - 1 >= need_after:
                holes.append(pocket)
                for rest in resolve(plist, pos + 1, slots_left - 1):
                    yield rest
                holes.pop()

        for size in range(1, min(slots, len(pockets)) + 1):
            for S in _combinations(range(len(pockets)), size):
                if any(not _touches_corner(pockets[i]) for i in S):
                    continue
                result = tile_all(skip_set=set(S))
                if result is None:
                    continue
                forced, active = result
                for hole_forced in resolve([pockets[i] for i in S], 0, slots):
                    yield forced + hole_forced, active
                for f_idx in forced:
                    del placements[f_idx]

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
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids()
                      and (not defer_interior_on[0] or interior_ready(i))]
        if not candidates:
            return float("inf")
        return min(local_score(i, point, free_shape) for i in candidates)

    def order_points(points_with_originator, free_shape):
        """Sort newly-exposed (point, originator) pairs by MRV. Anchors that
        can never do anything are dropped first (sound: free space and the
        remaining set only shrink along a branch): duplicates, points no
        longer on the free space's boundary, and originators with no
        remaining required partner."""
        if prune_anchors:
            rem = remaining_ids()
            kept, seen_po = [], set()
            fb = free_shape.boundary
            for p, o in points_with_originator:
                key = (round(p[0], 6), round(p[1], 6), o)
                if key in seen_po:
                    continue
                seen_po.add(key)
                if not any(t in rem for t in required_partners.get(o, [])):
                    continue
                if fb.distance(Point(p)) > 1e-7:
                    continue
                kept.append((p, o))
            points_with_originator = kept
        return sorted(points_with_originator, key=lambda po: point_score(po[0], po[1], free_shape))

    def seed_step(free_shape, deferred_only=False):
        """Point-first seed selection: fix one anchor point, rank every
        remaining id placeable there, and exhaust every tied candidate at
        that point before trying a different point.

        `triangle_corner_ids` is the PRIMARY rank, above degree: a known
        corner rect has nowhere else to go, so every point/candidate
        combo involving one is tried, everywhere, before any combo
        involving a non-corner rect. Degree in required_adjacency is the
        next rank (corner rooms tend to be lowest-degree); geometric
        local_score is the tie-break after that, then largest area
        first. Among degree-1 candidates, a point that's also one of the
        plot's own inner-L corners (`inner_l_points`) is preferred, since
        a degree-1 room tends to sit in a plot's inward notch on real
        floorplans."""
        verts = _polygon_vertices(free_shape)
        anchor_verts = [p for p in verts if boundary.distance(Point(p)) < 1e-9]
        if deferred_only:
            remaining = [i for i in remaining_ids() if i in interior and not interior_ready(i)]
        else:
            remaining = [i for i in remaining_ids() if not defer_interior_on[0] or interior_ready(i)]
        per_point = []
        for p in anchor_verts:
            _check_time()
            scored = []
            for idx in remaining:
                opts = placements_of(idx, p, free_shape)
                if not opts:
                    continue
                is_corner = 0 if idx in triangle_corner_ids else 1
                deg = len(required_partners.get(idx, []))
                w, h = sizes[idx]
                scored.append((is_corner, deg, _slot_count(opts, p), -(w * h), idx, opts))
            if not scored:
                continue
            scored.sort(key=lambda t: t[:5])
            # one entry per (corner, degree) level, so a failed
            # lowest-degree or non-corner seed falls back to the next
            # level rather than giving up
            for is_corner in sorted({c for c, _, _, _, _, _ in scored}):
                at_corner = [row for row in scored if row[0] == is_corner]
                for level in sorted({d for _, d, _, _, _, _ in at_corner}, key=_lvlkey):
                    level_score = min(s for _, d, s, _, _, _ in at_corner if d == level)
                    tied = sorted(
                        ((idx, opts) for _, d, s, _, idx, opts in at_corner if (d, s) == (level, level_score)),
                        key=lambda t: -(sizes[t[0]][0] * sizes[t[0]][1]),
                    )
                    is_inner_l = 0 if (level == 1 and p in inner_l_points) else 1
                    per_point.append((is_corner, level, is_inner_l, level_score, p, tied))
        if not per_point:
            return None
        per_point.sort(key=lambda t: (t[0], _lvlkey(t[1]), t[2], t[3]))
        return [(deg, score, p, tied) for (_ic, deg, _il, score, p, tied) in per_point]

    def remaining_ids():
        return [i for i in sizes if i not in placements]

    # Hole budget: every rect ends up placed, so the area left uncovered
    # in the final layout is exactly plot area - total rect area. Any free
    # region that no remaining rects can fill (see `waste_ok`) eats into it.
    hole_budget = max(0.0, poly.area - sum(w * h for w, h in sizes.values()))

    def waste_ok(free_shape) -> bool:
        """False once the area that MUST stay empty exceeds the budget.
        Counted: the pocket/hole already frozen as the one hole
        (`holes`), plus for every other free region its area minus the
        largest total of remaining rects that fit inside its bounding box
        (a lower bound on what stays empty there). Regions are disjoint
        and a region can only shrink, so the sum is a valid lower bound
        and exceeding the budget is a sure dead end."""
        used = sum(h.area for h in holes)
        if used > hole_budget + 1e-6:
            return False
        remaining = remaining_ids()
        for comp in _free_space_components(free_shape):
            if comp.area < 1e-9:
                continue
            bx0, by0, bx1, by1 = comp.bounds
            cw, ch = bx1 - bx0, by1 - by0
            fit = [sizes[i][0] * sizes[i][1] for i in remaining
                   if (sizes[i][0] <= cw + 1e-6 and sizes[i][1] <= ch + 1e-6)
                   or (sizes[i][1] <= cw + 1e-6 and sizes[i][0] <= ch + 1e-6)]
            used += comp.area - _best_subset_sum(fit, comp.area)
            if used > hole_budget + 1e-6:
                return False
        return True

    # convex plot corners (lowest-level necessary condition for coverage)
    _pc = list(poly.exterior.coords)[:-1]
    _sgn = 1 if sum(_pc[i][0] * _pc[(i + 1) % len(_pc)][1] - _pc[(i + 1) % len(_pc)][0] * _pc[i][1]
                    for i in range(len(_pc))) > 0 else -1
    convex_corners = []
    for _i, _v in enumerate(_pc):
        _a, _b = _pc[_i - 1], _pc[(_i + 1) % len(_pc)]
        if ((_v[0] - _a[0]) * (_b[1] - _v[1]) - (_v[1] - _a[1]) * (_b[0] - _v[0])) * _sgn > 0:
            convex_corners.append(_v)

    def _corner_coverable(v, free_shape, remaining) -> bool:
        """Could some remaining rect be placed with a corner exactly at
        convex plot corner `v` (all 4 quadrants x 2 orientations)? Only the
        right quadrant can fit, so this is a plain placeability test."""
        for i in remaining:
            w0, h0 = sizes[i]
            for uw, uh in {(w0, h0), (h0, w0)}:
                for rx in (v[0], v[0] - uw):
                    for ry in (v[1], v[1] - uh):
                        if not _rect_fits(free_shape, box(rx, ry, rx + uw, ry + uh)):
                            continue
                        if boundary_ok(i, rx, ry, uw, uh) and required_adjacency_ok(i, rx, ry, uw, uh):
                            return True
        return False

    def dead_corner_ok(free_shape, remaining) -> bool:
        """An uncovered convex plot corner nobody left can cover must end up
        inside a hole. Free components only shrink and stay distinct, so
        (frozen holes) + (components containing such a dead corner) is a
        lower bound on the final hole count."""
        dead_comps = set()
        comps = None
        for v in convex_corners:
            pt = Point(v)
            if free_shape.distance(pt) > 1e-9:
                continue                      # already covered
            if _corner_coverable(v, free_shape, remaining):
                continue
            if comps is None:
                comps = [c for c in _free_space_components(free_shape) if c.area > 1e-9]
            for k, c in enumerate(comps):
                if c.distance(pt) < 1e-9:
                    dead_comps.add(k)
                    break
        return hole_count() + len(dead_comps) <= max_holes

    # Border rooms first. An interior room touches no boundary, so placed
    # early it just floats beside one neighbour and the search re-arranges
    # it endlessly. In phase 1 an interior room is not a candidate anywhere
    # (frontier, seeds, triangle shortcut) until EVERY non-interior room is
    # placed; the remaining interior rooms then close the leftover free
    # space with the corner-cover tilers (`finish_interior`). Free space
    # that gets enclosed earlier is still tiled by `_fill_pockets_shortcut`.
    # If phase 1 fails outright, phase 2 reruns with no deferral, so the
    # ordering can cost time but never a solution.
    defer_interior_on = [bool(interior) and collect_solutions is None and _DEFER_INTERIOR]
    prune_anchors = _PRUNE_ANCHORS
    # root-restart state: every first placement (seed rect x position) gets a
    # node budget; budget-outs are retried in later rounds with a doubled
    # budget, fully refuted ones are never revisited
    root_limit = [None]
    root_budget = [_ROOT_BUDGET]
    done_roots = set()
    truncated = [False]

    def interior_ready(idx) -> bool:
        if idx not in interior:
            return True
        if all(q in placements for q in sizes if q not in interior):
            return True
        if _LOCAL_INTERIOR_READY:
            # earlier: as soon as every non-interior room it must touch is
            # placed, its spot is fully determined, so a wrong earlier choice
            # shows up right away (and the surround check can bite)
            border_nb = [q for q in required_partners.get(idx, []) if q not in interior]
            return bool(border_nb) and all(q in placements for q in border_nb)
        return False

    def finish_interior(free_shape) -> bool:
        """Only interior rooms remain: close the free space left by the
        border rooms with them -- every free piece tiled exactly, or (if no
        hole is used yet) one piece tiled with a single leftover hole.
        Leaves `placements` applied on success, rolled back on failure."""
        comps = [c for c in _free_space_components(free_shape) if c.area > 1e-9]

        def exact(cs):
            forced = []
            for c in cs:
                sub = _tile_pocket(c)
                if sub is None:
                    for f in forced:
                        placements.pop(f, None)
                    return None
                forced.extend(sub)
            return forced

        forced = exact(comps)
        if forced is not None:
            if not remaining_ids():
                return True
            for f in forced:
                placements.pop(f, None)
        slots = max_holes - hole_count() if require_single_hole else 0
        for size in range(1, min(slots, len(comps)) + 1):
            for S in _combinations(range(len(comps)), size):
                if any(not _touches_corner(comps[i]) for i in S):
                    continue
                forced = exact([comps[i] for i in range(len(comps)) if i not in S])
                if forced is None:
                    continue
                got, used, ok = [], 0, True
                for pos, i in enumerate(S):
                    bgt = slots - used - (len(S) - pos - 1)
                    if bgt < 1:
                        ok = False
                        break
                    _partial_budget[0] = 0
                    part = _tile_partial(comps[i], [], bgt)
                    if part is None:
                        ok = False
                        break
                    got.extend(part[0])
                    if part[1] is not None:
                        used += len(_parts(part[1]))
                if ok and not remaining_ids():
                    return True
                for f in got:
                    placements.pop(f, None)
                for f in forced:
                    placements.pop(f, None)
        return False

    def _solve_impl(free_shape, frontier):
        nodes[0] += 1
        if root_limit[0] is not None and nodes[0] > root_limit[0]:
            raise _BudgetOut()
        if _NODE_HOOK is not None:
            _NODE_HOOK(nodes[0], placements)
        if _SURROUND_CHECK and interior and not interior_surrounded_ok():
            return False
        if _PAIR_GAP_CHECK and not pair_gap_ok():
            return False
        if _SEALED_CHECK and not sealed_ok():
            return False
        _check_time()
        remaining = remaining_ids()
        if remaining and not waste_ok(free_shape):
            return False
        if _DEAD_CORNER_CHECK and require_single_hole and remaining and not dead_corner_ok(free_shape, remaining):
            return False
        if defer_interior_on[0] and remaining and all(i in interior for i in remaining):
            return bool(finish_interior(free_shape))
        if not remaining:
            if _SURROUND_CHECK and interior and not interior_surrounded_ok(final=True):
                return False
            if require_single_hole:
                total_pieces = _num_components(free_shape) + hole_count()
                if total_pieces > max_holes:
                    return False
                for _c in _free_space_components(free_shape):
                    if _c.area > 1e-6 and not _touches_corner(_c):
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

        if frontier:
            # plain FIFO over the frontier
            (point, originator), *rest = frontier
            partners = [i for i in required_partners.get(originator, []) if i in remaining]
            ready = [i for i in partners if not defer_interior_on[0] or interior_ready(i)]
            later = [i for i in partners if i not in ready]      # interior rooms still deferred
            if not ready and not later:
                return solve(free_shape, rest)
            cand_opts = {i: placements_of(i, point, free_shape) for i in ready + later}
            if slide_on[0]:
                # last-resort pass: also the slid placements, after every aligned one
                for i in ready + later:
                    cand_opts[i] = cand_opts[i] + placements_of(i, point, free_shape, slide_only=True)
            _ck = lambda i: (_slot_count(cand_opts[i], point), -(sizes[i][0] * sizes[i][1]))
            ready.sort(key=_ck)
            later.sort(key=_ck)
            # border rooms first; deferred interior rooms are tried at this
            # same point only after every border candidate has failed
            candidates = ready + later

            for cand in candidates:
                for (rx, ry, w, h, new_free) in cand_opts[cand]:
                    _check_time()
                    placements[cand] = (rx, ry, w, h)
                    new_points = order_points([(p, cand) for p in _new_frontier_points(rx, ry, w, h, point, new_free)], new_free)

                    # triangle shortcut: force in a required partner
                    # shared by two already-placed rects, at their L-union
                    # reflex corner or the shared perpendicular line
                    def resolve_chain(pending_pairs, extra_points, cur_free):
                        if not pending_pairs:
                            for forced_ids, filled_free in _fill_pockets_shortcut(cur_free):
                                result = solve(filled_free, extra_points + rest)
                                if result is True or result is None:
                                    return result
                            return False

                        (a, b), *rest_pairs = pending_pairs
                        common = [
                            t for t in required_partners.get(a, [])
                            if t in required_partners.get(b, []) and t in remaining_ids()
                        ]
                        if defer_interior_on[0]:
                            common.sort(key=lambda t: 0 if interior_ready(t) else 1)
                        if not common:
                            return resolve_chain(rest_pairs, extra_points, cur_free)

                        points_to_try = []
                        reflex = _reflex_corner(placements[a], placements[b])
                        if reflex is not None:
                            points_to_try.append(reflex)
                        for shared in _shared_corners(placements[a], placements[b]):
                            line = _perpendicular_search_line(placements[a], placements[b], shared)
                            if line is not None:
                                points_to_try.extend(_anchor_points_on_line(cur_free, *line))

                        any_option = False
                        # pass 1: partner aligned to the anchor (as before);
                        # pass 2 (secondary): partner slid along the free edge
                        # in whole grid steps, for a room floating between holes
                        for slide_pass in ((False, True) if slide_on[0] else (False,)):
                            for t_idx in common:
                                for pt in points_to_try:
                                    for (tx, ty, tw, th, t_free) in placements_of(t_idx, pt, cur_free, slide_only=slide_pass):
                                        _check_time()
                                        any_option = True
                                        placements[t_idx] = (tx, ty, tw, th)
                                        t_points = order_points([(p, t_idx) for p in _new_frontier_points(tx, ty, tw, th, pt, t_free)], t_free)
                                        result = resolve_chain(rest_pairs, extra_points + t_points, t_free)
                                        if result is True or result is None:
                                            return result
                                        del placements[t_idx]
                        if not any_option:
                            return False
                        return False

                    result = resolve_chain([(originator, cand)], new_points, new_free)
                    if result is True or result is None:
                        return result
                    del placements[cand]

            # every candidate failed here; if `originator` has no other
            # pending point left, this is a verified dead end
            still_pending_for_originator = any(o == originator for (_, o) in rest)
            if not still_pending_for_originator:
                return False
            return solve(free_shape, rest)

        # frontier empty: seed a fresh rectangle, point-first
        seed_lists = [seed_step(free_shape) or []]
        if defer_interior_on[0]:
            # border seeds first; deferred interior rooms as seeds only after
            # every border seed has failed, instead of backtracking
            seed_lists.append(seed_step(free_shape, deferred_only=True) or [])
        for per_point in seed_lists:
            for _deg, _score, point, tied in per_point:
                for seed, opts in tied:
                    if seed not in remaining_ids():
                        continue
                    for (rx, ry, uw, uh, new_free) in opts:
                        is_root = not placements and root_budget[0] is not None
                        rkey = (seed, round(rx, 6), round(ry, 6), round(uw, 6), round(uh, 6))
                        if is_root and rkey in done_roots:
                            continue
                        placements[seed] = (rx, ry, uw, uh)
                        new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                        if is_root:
                            root_limit[0] = nodes[0] + root_budget[0]
                        try:
                            for forced_ids, filled_free in _fill_pockets_shortcut(new_free):
                                result = solve(filled_free, new_points)
                                if result is True or result is None:
                                    root_limit[0] = None
                                    return result
                        except _BudgetOut:
                            if not is_root:
                                raise
                            truncated[0] = True
                            placements.clear()
                            holes.clear()
                            root_limit[0] = None
                            continue
                        if is_root:
                            done_roots.add(rkey)
                            root_limit[0] = None
                        del placements[seed]
        return False

    slide_on = [False]     # slide pass is a last resort: only after the plain search is exhausted
    memo = set()
    memo_hits = [0]

    def solve(free_shape, frontier):
        if not (_MEMO and collect_solutions is None):
            return _solve_impl(free_shape, frontier)
        key = (frozenset((i, round(p[0], 6), round(p[1], 6), round(p[2], 6), round(p[3], 6))
                         for i, p in placements.items()),
               tuple((round(pt[0], 6), round(pt[1], 6), o) for pt, o in frontier),
               round(sum(h.area for h in holes), 6), len(holes), defer_interior_on[0])
        if key in memo:
            memo_hits[0] += 1
            return False
        res = _solve_impl(free_shape, frontier)
        if res is False:
            memo.add(key)
        return res

    def run_rounds():
        """Restart loop: solve with a per-root node budget, doubling it each
        round until a round finishes with nothing cut off (then a False is a
        real proof) or a solution turns up."""
        root_budget[0] = _ROOT_BUDGET
        done_roots.clear()
        while True:
            truncated[0] = False
            res = solve(poly, [])
            if res is True or res is None:
                return res
            if not truncated[0] or root_budget[0] is None:
                return False
            placements.clear()
            holes.clear()
            root_budget[0] *= 2

    try:
        result = run_rounds()
        if result is False and defer_interior_on[0]:
            defer_interior_on[0] = False
            placements.clear()
            holes.clear()
            result = run_rounds()
        if result is False and _SLIDE and grid_step > 0 and max_holes - 0 >= 1 and require_single_hole:
            # the plain search proved nothing: anchored placements can miss a
            # room floating between holes, so retry with the slide pass on
            slide_on[0] = True
            defer_interior_on[0] = bool(interior) and collect_solutions is None and _DEFER_INTERIOR
            placements.clear()
            holes.clear()
            result = run_rounds()
            if result is False and defer_interior_on[0]:
                defer_interior_on[0] = False
                placements.clear()
                holes.clear()
                result = run_rounds()
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
    """Returns (overlap_count, outside_boundary_count, hole_count)."""
    boxes = [box(x, y, x + w, y + h) for x, y, w, h in placements.values()]
    overlaps = sum(
        1 for a in range(len(boxes)) for b in range(a + 1, len(boxes))
        if boxes[a].intersection(boxes[b]).area > 1e-9
    )
    poly_shape = Polygon(polygon)
    outside = sum(
        1 for (x, y, w, h) in placements.values()
        if poly_shape.intersection(box(x, y, x + w, y + h)).area < w * h - 1e-6
    )
    occupied = box(0, 0, 0, 0)
    for x, y, w, h in placements.values():
        occupied = occupied.union(box(x, y, x + w, y + h))
    leftover = poly_shape.difference(occupied)
    holes = len(leftover.geoms) if leftover.geom_type == "MultiPolygon" else (1 if not leftover.is_empty else 0)
    return overlaps, outside, holes


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_solution_grid(polygon, solutions, save_path, cols=5):
    """Saves a grid with one panel per solution (a dict of placements)."""
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
        for x, y, w, h in placements.values():
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=0.8, edgecolor="black",
                                            facecolor="tab:blue", alpha=0.65))
            ax.text(x + w / 2, y + h / 2, f"{w:g}x{h:g}", ha="center", va="center", fontsize=6)
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

_DEG1_AFTER3 = True

def _lvlkey(d):
    return 3.5 if (_DEG1_AFTER3 and d == 1) else d


EXAMPLES = {
    1: dict(  # A: 3BHK house, L-shaped garden wrapping the east end
        polygon=[(0, 0), (20, 0), (20, 8), (10, 8), (10, 18), (0, 18)],
        rects=dict(enumerate([(4, 4), (2, 4), (5, 4), (7, 5), (5, 4), (3, 4), (3, 4), (5, 5), (2, 3), (5, 2), (6, 4), (4, 4)])),
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (6, 8), (6, 9), (7, 8), (7, 9), (7, 10), (8, 9), (9, 10), (9, 11), (10, 11)],
        interior=[8],   # inner = no point touches the plot boundary or the hole
        names=['Bedroom 3', 'Bathroom', 'Foyer', 'Living room', 'Kitchen', 'Dining', 'Study', 'Bedroom 2', 'Common bath', 'Store', 'Master bedroom', 'Master bath'],
        hole='L-shaped garden / parking',
    ),
    2: dict(  # B: 4-bedroom house, stepped terrace on the tall wing
        polygon=[(0, 0), (18, 0), (18, 16), (9, 16), (9, 7), (0, 7)],
        rects=dict(enumerate([(5, 5), (7, 5), (6, 5), (5, 2), (2, 2), (2, 2), (2, 4), (2, 3), (4, 5), (3, 4), (4, 2)])),
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 4), (1, 5), (1, 6), (1, 8), (2, 8), (2, 9), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (7, 8), (7, 10), (8, 9), (8, 10)],
        names=['Bedroom 4', 'Living room', 'Dining', 'Study', 'Bathroom', 'Store', 'Kitchen', 'Utility', 'Master bedroom', 'Bedroom 3', 'Master bath'],
        hole='stepped open terrace',
    ),
    3: dict(  # C: family home, L-shaped courtyard at the foot of the tall wing
        polygon=[(0, 0), (8, 0), (8, 10), (18, 10), (18, 18), (0, 18)],
        rects=dict(enumerate([(4, 3), (4, 6), (4, 3), (6, 5), (5, 5), (3, 3), (5, 5), (2, 5), (3, 3), (4, 3)])),
        required_adjacency=[(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (4, 6), (4, 8), (5, 6), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9)],
        names=['Bathroom', 'Living room', 'Study', 'Bedroom 3', 'Kitchen', 'Dining', 'Master bedroom', 'Walk-in', 'Master bath', 'Bedroom 2'],
        hole='L-shaped courtyard',
    ),
    4: dict(  # D: 2-bedroom home, L-shaped yard at the tall end
        polygon=[(0, 0), (14, 0), (14, 6), (6, 6), (6, 15), (0, 15)],
        rects=dict(enumerate([(4, 4), (5, 4), (3, 4), (2, 4), (8, 2), (2, 4), (4, 5), (4, 2)])),
        required_adjacency=[(0, 1), (0, 5), (0, 6), (1, 2), (1, 4), (1, 6), (2, 3), (2, 4), (3, 4), (4, 6), (5, 6), (6, 7)],
        names=['Bedroom 1', 'Living room', 'Kitchen', 'Utility', 'Lobby', 'Bathroom', 'Dining', 'Study'],
        hole='L-shaped back yard',
    ),
    5: dict(  # E: large villa, stepped side garden with two inner rooms
        polygon=[(0, 0), (24, 0), (24, 10), (14, 10), (14, 20), (0, 20)],
        rects=dict(enumerate([(5, 5), (4, 5), (4, 5), (4, 5), (6, 2), (6, 6), (6, 5), (3, 5), (4, 5), (6, 5), (4, 5), (6, 2), (8, 5)])),
        required_adjacency=[(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 4), (2, 6), (3, 4), (3, 5), (3, 8), (4, 5), (4, 6), (5, 6), (5, 8), (5, 10), (5, 11), (6, 7), (6, 10), (8, 9), (8, 11), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12)],
        interior=[4, 5, 11],   # inner = no point touches the plot boundary or the hole
        names=['Bedroom 1', 'Bedroom 2', 'Bath & laundry', 'Master bedroom', 'Hall', 'Family room', 'Kitchen', 'Dining', 'Study', 'Bedroom 3', 'Gym', 'Store', 'Master suite'],
        hole='stepped side garden',
    ),
    6: dict(  # F: corner plot, L-shaped yard around the reflex corner
        polygon=[(0, 0), (15, 0), (15, 7), (8, 7), (8, 16), (0, 16)],
        rects=dict(enumerate([(5, 5), (6, 3), (3, 2), (4, 5), (4, 5), (4, 6), (4, 3), (4, 3)])),
        required_adjacency=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6), (5, 7), (6, 7)],
        names=['Bedroom 1', 'Garage', 'Bathroom', 'Living room', 'Kitchen', 'Bedroom 2', 'Study', 'Bedroom 3'],
        hole='L-shaped yard at the notch',
    ),
    7: dict(  # G: house with a stair-step (terraced) garden
        polygon=[(0, 0), (17, 0), (17, 8), (10, 8), (10, 17), (0, 17)],
        rects=dict(enumerate([(4, 3), (2, 3), (2, 3), (7, 3), (5, 5), (3, 2), (2, 5), (5, 5), (3, 3), (5, 6), (5, 4), (5, 2)])),
        required_adjacency=[(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 6), (3, 7), (4, 5), (4, 7), (4, 8), (6, 7), (6, 9), (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (9, 11), (10, 11)],
        interior=[7],   # inner = no point touches the plot boundary or the hole
        names=['Bedroom 1', 'Bathroom', 'Store', 'Living room', 'Kitchen', 'Porch', 'Bath 2', 'Dining', 'Study', 'Bedroom 2', 'Master bedroom', 'Master bath'],
        hole='stair-step terraced garden',
    ),
    8: dict(  # H: wrap-around L plot, central void with a corner-tucked 5x3
        polygon=[(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)],
        rects=dict(enumerate([(4, 11), (8, 3), (7, 3), (6, 6), (6, 2), (5, 3), (6, 4), (6, 4), (6, 8)])),
        required_adjacency=[(0, 1), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5), (4, 7), (6, 7), (6, 8), (7, 8)],
        names=['4x11', '8x3', '7x3', '6x6', '6x2', '5x3', '6x4 upper', '6x4 lower', '6x8'],
        hole='irregular central void',
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
                      interior_ids=example.get("interior", ()))


def run_custom_example(max_solutions: int = 5, time_limit: Optional[float] = 60.0):
    """Prompt the user for their own polygon / rects / required_adjacency
    and run the same solve-and-report flow as `run_example`. Each value is
    typed as a plain Python literal and parsed with `ast.literal_eval`."""
    print("\nEnter your own floorplan. Each value is a Python literal, e.g.:")
    print("  polygon: [(0,0), (10,0), (10,10), (0,10)]")
    print("  rects:   {0: (4,4), 1: (6,4), 2: (10,6)}")
    print("  required_adjacency: [(0,1), (1,2)]   (or [] for none)")

    try:
        polygon = ast.literal_eval(input("\npolygon: ").strip())
        rects_raw = ast.literal_eval(input("rects: ").strip())
        rects = {int(k): tuple(v) for k, v in rects_raw.items()}
        adj_raw = input("required_adjacency (blank for none): ").strip()
        required_adjacency = ast.literal_eval(adj_raw) if adj_raw else []
        int_raw = input("interior rects, ids that must not touch the boundary (blank for none): ").strip()
        interior_ids = list(ast.literal_eval(int_raw)) if int_raw else []
    except (ValueError, SyntaxError) as exc:
        print(f"Could not parse that: {exc}")
        return

    _solve_and_report("custom", polygon, rects, required_adjacency, max_solutions, time_limit,
                      interior_ids=interior_ids)


def _solve_and_report(label, polygon, rects, required_adjacency, max_solutions, time_limit, interior_ids=()):
    n_adj = len(required_adjacency or [])
    print(f"\nRunning example {label}: {len(rects)} rects, {n_adj} adjacencies")
    print(f"  polygon: {polygon}")
    print(f"  rects:   {rects}")
    if required_adjacency:
        print(f"  required_adjacency (PTPG): {sorted(required_adjacency)}")
    if interior_ids:
        print(f"  interior rects: {sorted(interior_ids)}")

    t0 = time.perf_counter()
    found0, placements0 = None, None
    try:
        found0, placements0, nodes0 = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            require_single_hole=True,
            require_boundary_touch=False,
            interior_ids=interior_ids,
            time_limit=time_limit,
        )
        t1 = time.perf_counter()
        status = "FEASIBLE" if found0 else "INFEASIBLE" if found0 is False else "INCONCLUSIVE"
        print(f"\n  initial solve: {status}   nodes explored: {nodes0}   time: {t1 - t0:.3f}s")
        if found0 and placements0:
            overlaps0, outside0, holes0 = verify_placements(polygon, placements0)
            poly_area0 = Polygon(polygon).area
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
            require_single_hole=True,
            require_boundary_touch=False,
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

    poly_area = Polygon(polygon).area
    for i, sol in enumerate(solutions, start=1):
        overlaps, outside, holes = verify_placements(polygon, sol)
        coverage = sum(w * h for _, _, w, h in sol.values()) / poly_area * 100
        print(f"    solution {i}: coverage={coverage:.1f}%  overlaps={overlaps}  outside={outside}  holes={holes}")

    save_path = f"corner_hole_sols_ex_{label}.png"
    render_solution_grid(polygon, solutions, save_path)
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