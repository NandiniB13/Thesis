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
part of one edge (`require_boundary_touch`). A placement is rejected if
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
import time
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry


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


def _rect_fits(free_shape: BaseGeometry, candidate: BaseGeometry) -> bool:
    if free_shape.is_empty:
        return False
    # cheap bbox reject before the expensive shapely intersection
    fminx, fminy, fmaxx, fmaxy = free_shape.bounds
    cminx, cminy, cmaxx, cmaxy = candidate.bounds
    if cminx < fminx - 1e-9 or cminy < fminy - 1e-9 or cmaxx > fmaxx + 1e-9 or cmaxy > fmaxy + 1e-9:
        return False
    return abs(free_shape.intersection(candidate).area - candidate.area) < 1e-6


def _shares_boundary_edge(rx: float, ry: float, w: float, h: float, boundary, tol: float = 1e-9) -> bool:
    """True if any edge of the rectangle overlaps the plot boundary by more than a point."""
    edges = [
        LineString([(rx, ry), (rx + w, ry)]),
        LineString([(rx, ry + h), (rx + w, ry + h)]),
        LineString([(rx, ry), (rx, ry + h)]),
        LineString([(rx + w, ry), (rx + w, ry + h)]),
    ]
    return any(edge.length >= tol and edge.intersection(boundary).length > tol for edge in edges)


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


def pack_rectangles_frontier(
    polygon,
    rects: Dict[int, Size],
    required_adjacency=None,
    *,
    require_single_hole: bool = True,
    require_boundary_touch: bool = True,
    time_limit: Optional[float] = 60.0,
    collect_solutions: Optional[list] = None,
    max_solutions: Optional[int] = None,
):
    """PTPG-driven frontier search. Returns (found, placements_or_None,
    nodes_explored). found is True / False / None (feasible / proven
    infeasible / cut off by time_limit).

    `time_limit` (seconds, wall-clock; None = unbounded) is checked via
    `_check_time` in every hot loop, not just `solve()`, since real work
    (a vertex scan, `_tile_pocket`'s own search) can happen without
    calling `solve()` again in between. Exceeding it raises
    `_SearchTimeout`, caught once at the bottom of this function.

    `required_adjacency` (an iterable of (id_a, id_b) pairs over the same
    ids as `rects`) is optional -- `None`/`[]` means no adjacency
    constraint (every rect is its own region, re-seeded independently;
    `require_single_hole`/`require_boundary_touch` still apply).

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

    placements: Dict[int, Placement] = {}
    nodes = [0]
    deadline = time.perf_counter() + time_limit if time_limit is not None else None

    def _check_time():
        if deadline is not None and time.perf_counter() > deadline:
            raise _SearchTimeout()

    # The one pocket (if any) this branch has set aside as the final
    # leftover hole `require_single_hole` allows -- see
    # `_fill_pockets_shortcut`. Reset to None whenever that choice is undone.
    hole_geom = [None]

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
        allowed hole yet (`hole_geom[0] is None`), one failing pocket is
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
        freebies = 1 if hole_geom[0] is None else 0
        for pocket in comps[1:]:
            ok = bool(remaining) and pocket.area <= total_remaining_area + 1e-6
            if ok:
                minx, miny, maxx, maxy = pocket.bounds
                if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
                    ok = False
            if not ok:
                if freebies > 0:
                    freebies -= 1
                    continue
                return False
        return True

    def boundary_ok(rx, ry, w, h) -> bool:
        return (not require_boundary_touch) or _shares_boundary_edge(rx, ry, w, h, boundary)

    def placements_of(idx, point, free_shape, check_hole=True):
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
        for uw, uh in {(w0, h0), (h0, w0)}:
            for ox, oy in _candidate_offsets(uw, uh):
                rx, ry = px + ox, py + oy
                candidate = box(rx, ry, rx + uw, ry + uh)
                if not _rect_fits(free_shape, candidate):
                    continue
                if not boundary_ok(rx, ry, uw, uh):
                    continue
                if not required_adjacency_ok(idx, rx, ry, uw, uh):
                    continue
                new_free = free_shape.difference(candidate)
                if check_hole and not single_hole_ok(new_free):
                    continue
                out.append((rx, ry, uw, uh, new_free))
        return out

    def _rect_still_placeable(q_idx, free_shape) -> bool:
        """Whether the not-yet-placed `q_idx` could be placed anywhere in
        the current free space, checked across every live vertex.
        `required_adjacency_ok` checks exactly against every placed rect,
        so this existence check is sound: free space only shrinks, so a
        rect unplaceable now stays unplaceable later in this branch (a
        boundary stretch can still split into a new vertex later, but
        that's an accepted limitation of the anchor-at-a-vertex convention)."""
        for v in _polygon_vertices(free_shape):
            _check_time()
            if placements_of(q_idx, v, free_shape):
                return True
        return False

    def _tile_pocket(pocket_shape, depth_limit=3):
        """Try to exactly close `pocket_shape` using one or more remaining
        rects: anchor a rect at a live vertex, and if it leaves a
        sub-piece, recurse into that too, until the pocket is fully
        covered or every option is exhausted. Returns the list of forced
        ids on success (left applied to `placements`), or None on
        failure with everything rolled back. `depth_limit` bounds
        recursion depth (structural, not a work budget); wall-clock cost
        is bounded by `_check_time`."""
        if pocket_shape.is_empty or pocket_shape.area < 1e-9:
            return []
        if depth_limit <= 0:
            return None
        remaining = remaining_ids()
        if not remaining:
            return None
        # cheap prunes before any shapely candidate generation
        pminx, pminy, pmaxx, pmaxy = pocket_shape.bounds
        pw, ph = pmaxx - pminx, pmaxy - pminy
        if sum(sizes[i][0] * sizes[i][1] for i in remaining) < pocket_shape.area - 1e-6:
            return None
        fitting_ids = [
            i for i in remaining
            if (sizes[i][0] <= pw + 1e-6 and sizes[i][1] <= ph + 1e-6)
            or (sizes[i][1] <= pw + 1e-6 and sizes[i][0] <= ph + 1e-6)
        ]
        if not fitting_ids:
            return None
        verts = _polygon_vertices(pocket_shape)
        for point in verts:
            for idx in fitting_ids:
                for (rx, ry, uw, uh, new_pocket) in placements_of(idx, point, pocket_shape, check_hole=False):
                    _check_time()
                    placements[idx] = (rx, ry, uw, uh)
                    forced_here = [idx]
                    ok = True
                    for comp in _free_space_components(new_pocket):
                        sub_forced = _tile_pocket(comp, depth_limit - 1)
                        if sub_forced is None:
                            ok = False
                            break
                        forced_here.extend(sub_forced)
                    if ok:
                        return forced_here
                    for f_idx in forced_here:
                        if f_idx in placements:
                            del placements[f_idx]
        return None

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
          2. If that fails and `hole_geom[0]` is still None, try freezing
             each pocket in turn (leave it untouched forever, excluded
             from the active free_shape) while tiling the rest -- this is
             the one leftover region `require_single_hole` allows.

        Mutates `placements`/`hole_geom[0]` to match whichever
        alternative is currently yielded; a caller that keeps looping
        past a failed alternative triggers this generator's own cleanup
        (after the `yield`) before producing the next one."""
        comps = _free_space_components(free_shape)
        if len(comps) <= 1:
            yield [], free_shape
            return
        main = comps[0]
        pockets = comps[1:]

        def tile_all(skip_index=None):
            forced = []
            for i, pocket in enumerate(pockets):
                if i == skip_index:
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

        if hole_geom[0] is None:
            for i, pocket in enumerate(pockets):
                result = tile_all(skip_index=i)
                if result is None:
                    continue
                forced, active = result
                hole_geom[0] = pocket
                yield forced, active
                hole_geom[0] = None
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
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids()]
        if not candidates:
            return float("inf")
        return min(local_score(i, point, free_shape) for i in candidates)

    def order_points(points_with_originator, free_shape):
        """Sort newly-exposed (point, originator) pairs by MRV."""
        return sorted(points_with_originator, key=lambda po: point_score(po[0], po[1], free_shape))

    def seed_step(free_shape):
        """Point-first seed selection: fix one anchor point, rank every
        remaining id placeable there, and exhaust every degree-tied
        candidate at that point before trying a different point.

        Degree in required_adjacency is the primary rank (corner rooms
        tend to be lowest-degree); geometric local_score is the
        tie-break, then largest area first. Among degree-1 candidates, a
        point that's also one of the plot's own inner-L corners
        (`inner_l_points`) is preferred, since a degree-1 room tends to
        sit in a plot's inward notch on real floorplans."""
        verts = _polygon_vertices(free_shape)
        anchor_verts = [p for p in verts if boundary.distance(Point(p)) < 1e-9]
        remaining = remaining_ids()
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
            scored.sort(key=lambda t: t[:4])
            # one entry per degree level, so a failed lowest-degree seed
            # falls back to the next level rather than giving up
            for level in sorted({d for d, _, _, _, _ in scored}):
                level_score = min(s for d, s, _, _, _ in scored if d == level)
                tied = sorted(
                    ((idx, opts) for d, s, _, idx, opts in scored if (d, s) == (level, level_score)),
                    key=lambda t: -(sizes[t[0]][0] * sizes[t[0]][1]),
                )
                is_inner_l = 0 if (level == 1 and p in inner_l_points) else 1
                per_point.append((level, is_inner_l, level_score, p, tied))
        if not per_point:
            return None
        per_point.sort(key=lambda t: (t[0], t[1], t[2]))
        return [(deg, score, p, tied) for (deg, _il, score, p, tied) in per_point]

    def remaining_ids():
        return [i for i in sizes if i not in placements]

    def solve(free_shape, frontier):
        nodes[0] += 1
        _check_time()
        remaining = remaining_ids()
        if not remaining:
            if require_single_hole:
                total_pieces = _num_components(free_shape) + (1 if hole_geom[0] is not None else 0)
                if total_pieces > 1:
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
            candidates = [i for i in required_partners.get(originator, []) if i in remaining]
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
                        for t_idx in common:
                            for pt in points_to_try:
                                for (tx, ty, tw, th, t_free) in placements_of(t_idx, pt, cur_free):
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
        per_point = seed_step(free_shape)
        if not per_point:
            return False
        for _deg, _score, point, tied in per_point:
            for seed, opts in tied:
                if seed not in remaining_ids():
                    continue
                for (rx, ry, uw, uh, new_free) in opts:
                    placements[seed] = (rx, ry, uw, uh)
                    new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                    for forced_ids, filled_free in _fill_pockets_shortcut(new_free):
                        result = solve(filled_free, new_points)
                        if result is True or result is None:
                            return result
                    del placements[seed]
        return False

    try:
        result = solve(poly, [])
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

EXAMPLES = {
    1: dict(
        polygon=[(0, 0), (25, 0), (25, 15), (15, 15), (15, 30), (0, 30)],
        rects=dict(enumerate([(15, 6), (10, 6), (15, 4), (15, 4), (8, 7), (7, 7), (5, 9), (16, 9), (4, 9)])),
        required_adjacency=[(0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 7), (2, 3),
                             (3, 6), (3, 7), (3, 8), (4, 5), (6, 7), (6, 8)],
    ),
    2: dict(
        polygon=[(0, 0), (24, 0), (24, 16), (16, 16), (16, 24), (0, 24)],
        rects=dict(enumerate([(24, 4), (8, 7), (8, 7), (13, 4), (7, 4), (4, 4), (16, 4), (16, 5)])),
        required_adjacency=[(0, 6), (0, 7), (1, 4), (1, 7), (2, 3), (2, 4), (3, 5), (3, 6), (5, 6)],
    ),
    3: dict(
        polygon=[(0, 0), (35, 0), (35, 20), (22, 20), (22, 40), (0, 40)],
        rects=dict(enumerate([(8, 8), (13, 10), (14, 8), (17, 12), (12, 10), (5, 5), (5, 5),
                               (12, 10), (6, 5), (6, 5), (10, 10), (5, 7), (5, 5)])),
        required_adjacency=[(0, 2), (0, 12), (0, 3), (2, 3), (3, 12), (3, 11),
                            (12, 11), (3, 10), (11, 10), (10, 5), (10, 4), (5, 4), (5, 6), (6, 4), (4, 7), (7, 8),
                            (7, 9), (7, 1), (8, 9), (9, 1)],
    ),
    4: dict(
        polygon=[(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)],
        rects=dict(enumerate([(6, 8), (6, 6), (3, 7), (6, 4), (6, 2), (4, 6), (11, 4), (9, 2), (3, 8)])),
        required_adjacency=[(0, 3), (0, 5), (1, 2), (1, 4), (2, 6), (3, 4), (3, 5), (6, 8), (7, 8)],
    ),
    5: dict(
        polygon=[(0, 0), (20, 0), (20, 15), (10, 15), (10, 30), (0, 30)],
        rects=dict(enumerate([(6, 6), (11, 4), (1, 5), (2, 4), (9, 6), (6, 10), (5, 10), (9, 5), (10, 2)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)],
    ),
    6: dict(
        polygon=[(0, 0), (16, 0), (16, 6), (9, 6), (9, 14), (0, 14)],
        rects=dict(enumerate([(6, 4), (3, 3), (3, 4), (5, 3), (3, 8), (3, 6), (5, 2), (3, 4), (2, 3), (6, 2)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    ),
    7: dict(
        polygon=[(0, 0), (16, 0), (16, 8), (6, 8), (6, 23), (0, 23)],
        rects=dict(enumerate([(3, 2), (3, 2), (6, 2), (3, 2), (3, 2), (4, 6),
                               (5, 6), (8, 5), (8, 3), (3, 2), (3, 2), (3, 4), (5, 4), (3, 4)])),
        required_adjacency=[
            (0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4),
            (1, 5), (2, 5), (4, 5), (5, 6), (6, 7), (6, 8),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8),
        ],
    ),
    8: dict(
        polygon=[(0, 7), (0, 13), (20, 13), (20, 0), (13, 0), (13, 7)],
        rects=dict(enumerate([(7, 4), (7, 3), (2, 3), (4, 4), (2, 3), (2, 2), (2, 5), (2, 2), (2, 4), (4, 5), (3, 4), (3, 4)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9), (9, 10), (10, 11)],
    ),
    9: dict(
        polygon=[(0, 0),(21,0),(21,6),(9,6),(9,18),(0,18)],
        rects=dict(enumerate([(6,8),(6,4),(2,2),(3,2),(4,5),(2,5),(3,5),(2,5),(3,3),(4,9)])),
        required_adjacency=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)],
    ),
    10: dict(
        polygon=[(0, 0), (18, 0), (18, 13), (10, 13), (10, 24), (0, 24)],
        rects=dict(enumerate([(9, 6),(4, 4),(4, 4),(6, 4),(4, 4),(5, 6),(3, 4),(2, 4),(4, 4),(3, 4),(5, 5),(7, 7),(5, 3)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
    ),
}


# --------------------------------------------------------------------------
# Interactive runner
# --------------------------------------------------------------------------

def run_example(
    choice: int,
    max_solutions: int = 5,
    time_limit: Optional[float] = 300.0,
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

    n_adj = len(example.get("required_adjacency") or [])
    print(f"\nRunning example {choice}: {len(rects)} rects, {n_adj} adjacencies")
    print(f"  polygon: {polygon}")
    print(f"  rects:   {rects}")
    if required_adjacency:
        print(f"  required_adjacency (PTPG): {sorted(required_adjacency)}")

    t0 = time.perf_counter()
    found0, placements0 = None, None
    try:
        found0, placements0, nodes0 = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            require_single_hole=True,
            require_boundary_touch=True,
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
            require_boundary_touch=True,
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

    save_path = f"packing_multi_solutions_ex_{choice}.png"
    render_solution_grid(polygon, solutions, save_path)
    print(f"  saved to {save_path}")


def main():
    print("Available examples:")
    for key, ex in EXAMPLES.items():
        n_adj = len(ex.get("required_adjacency") or [])
        print(f"  {key}. {len(ex['rects'])} rects, {n_adj} adjacencies")

    raw = input(f"\nChoose an example [{min(EXAMPLES)}-{max(EXAMPLES)}]: ").strip()
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