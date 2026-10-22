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

The total rectangle area must equal the plot area (`check_area_match`).
Every candidate is anchored at a live vertex of the current free
polygon, and every placed rectangle must touch the plot boundary along
part of one edge (`require_boundary_touch`), except `interior_ids`
rects, which must not share any edge segment with it. When free space splits into
several regions, the search works only on the smallest one (its area
must equal some subset of the remaining rect areas); a region that
cannot be filled is a dead end and the search backtracks. The final touch-graph must match
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


def _vertex_in_closed_box(rx, ry, w, h, vertices, tol: float = 1e-9) -> bool:
    """True if any plot vertex lies in the closed rectangle (corner, on an
    edge, or inside)."""
    return any(rx - tol <= vx <= rx + w + tol and ry - tol <= vy <= ry + h + tol
               for (vx, vy) in vertices)


def check_area_match(polygon, rects: Dict[int, Size], tol: float = 1e-6) -> Tuple[bool, str]:
    """(True, "") if the rects' total area is <= the plot area, else
    (False, message). Total area < plot area leaves a hole (see
    `allow_hole` / `single_hole` in `pack_rectangles_frontier`)."""
    plot_area = Polygon(polygon).area
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


def _touches_boundary_edge(shape: BaseGeometry, boundary, tol: float = 1e-9) -> bool:
    """True if `shape` meets the plot boundary AT ALL -- a shared edge
    segment or even a single point counts. (Name kept; used for the hole
    and for free-space pieces.)"""
    return shape.intersects(boundary)


def _touches_boundary(rx: float, ry: float, w: float, h: float, boundary) -> bool:
    """True if the closed rectangle meets the plot boundary at all (an edge
    segment or a single point). A rect that does NOT is an interior rect."""
    return box(rx, ry, rx + w, ry + h).intersects(boundary)


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
    require_boundary_touch: bool = True,
    interior_ids: Iterable[int] = (),
    time_limit: Optional[float] = 60.0,
    collect_solutions: Optional[list] = None,
    max_solutions: Optional[int] = None,
    single_hole: bool = True,
    hole_corner_rules: bool = True,
    triangle_shortcut: bool = True,
    rect_union_try: bool = False,
    border_first: bool = False,
    interior_corner_rule: bool = True,
    interior_grid: bool = True,
    weak_forward_check: bool = True,
):
    """PTPG-driven frontier search. Returns (found, placements_or_None,
    nodes_explored). found is True / False / None (feasible / proven
    infeasible / cut off by time_limit). Returns False immediately if the
    total rect area exceeds the plot area (`check_area_match`) or if any
    rect has no required neighbour, the required adjacency is not one connected
    graph, or every rect is interior (`check_no_isolated`).

    If the total rect area is < the plot area the leftover is a HOLE. The
    hole must be interior: no leftover piece may touch the plot boundary at
    all (not even at a single point), and with `single_hole`
    (default) the leftover must be one connected piece. The leftover area
    is fixed (plot area - rect area) so it is also a pruning budget. When
    the areas are equal the search is the exact-tiling search.

    `triangle_shortcut` (default True): right after placing a rect the
    search forces in a required partner shared by it and its originator,
    anchored at their reflex corner or on the shared perpendicular line --
    but ONLY when the two rects do not join into a single rectangle
    (`_union_is_rectangle`). If they do (e.g. two rooms side by side), the
    shared partner is left to the normal chain order: seed, a partner of
    the seed, a partner of THAT rect, and so on. False switches the
    shortcut off entirely.

    `rect_union_try` (default False): when the two placed rects join into a
    single rectangle the shortcut is still ATTEMPTED (forced placement of
    the shared partner), but unlike the non-rectangular case a failed
    attempt is not a dead end: the search falls back to the normal
    unforced continuation. False skips the shortcut for rectangular unions
    altogether.

    `interior_grid` (default True): everything runs as before until every
    boundary room is placed. From then on the interior rooms that are still
    unplaced are placed on a GRID instead of at corners of the free space:
    every lattice position (step = gcd of all plot coordinates and room
    sizes), both orientations, where the room fits in the free space, is not
    on the plot boundary and satisfies the required adjacency against every
    room already placed (touch iff required). The room with the fewest such
    positions goes next. Free space that still touches the plot boundary at
    that moment can never be filled by interior rooms, so it ends the
    branch. With whole-number data this is complete for the remaining
    interior rooms (a floating room can sit anywhere in the pocket).

    `interior_corner_rule` (default True): interior rooms are placed
    whenever the normal search reaches them (no ordering), but the corner
    points an interior room exposes may only be used to place OTHER
    INTERIOR rooms. A boundary room is never anchored at a frontier point
    that came from an interior room (the same holds for the triangle
    shortcut: a boundary room is not forced in next to an interior pair).
    Boundary rooms still get their anchors from the plot boundary and from
    other boundary rooms.

    `border_first` (default False): interior rooms (`interior_ids`) are not
    placed until EVERY boundary room has been placed. While boundary rooms
    remain, interior rooms are not candidates anywhere (frontier, seeds,
    triangle shortcut, region seeding) and free-space pieces that do not
    touch the plot boundary are left alone. Once the last boundary room is
    down, seeding may anchor at any free-space vertex, not only at ones on
    the plot boundary, so the interior rooms can be placed inside the
    pockets the boundary rooms enclosed. If that ordering finds nothing,
    the search is rerun once with no ordering restriction, so a `False`
    never rests on the ordering alone.

    `hole_corner_rules` (default True) keeps the triangle-corner placement
    filter and the exhaustive corner clause on in hole mode too, applied to
    every boundary rect (see the comment at `triangle_corner_ids`). This is
    NOT proven sound for rects that meet the boundary only at a point or
    whose side ends at a plot vertex; pass False for the guarded filter
    (edge-contact rects, no plot vertex in the box) and no corner clause.

    `interior_ids` are rects that must not touch the plot boundary at all
    (no shared edge segment and no single-point contact). Every other rect
    is a boundary rect: with `require_boundary_touch` it must meet the
    boundary, and a single-point contact is enough.

    `time_limit` (seconds, wall-clock; None = unbounded) is checked via
    `_check_time` in every hot loop, not just `solve()`, since real work
    (a vertex scan) can happen without calling `solve()` again in
    between. Exceeding it raises
    `_SearchTimeout`, caught once at the bottom of this function.

    `required_adjacency` (an iterable of (id_a, id_b) pairs over the same
    ids as `rects`) is optional -- `None`/`[]` means no adjacency
    constraint (every rect is its own region, re-seeded independently;
    `require_boundary_touch` still applies).

    Pass `collect_solutions` (a list) to keep searching past the first
    valid placement: each is appended and the search backtracks for a
    genuinely different one, up to `max_solutions` (None = exhaustive).
    In this mode the second return value is always `collect_solutions`
    itself."""
    sizes = dict(rects)
    if not check_area_match(polygon, sizes)[0]:
        return False, collect_solutions, 0
    if not check_no_isolated(sizes, required_adjacency, interior_ids)[0]:
        return False, collect_solutions, 0
    hole_mode = Polygon(polygon).area - sum(w * h for w, h in sizes.values()) > 1e-6

    boundary = Polygon(polygon).boundary
    poly = Polygon(polygon)
    inner_l_points = _polygon_reflex_vertices(polygon)
    convex_vertices = _polygon_convex_vertices(polygon)
    plot_vertices = list(polygon)
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

    # Triangle-corner rule: a rect with exactly 2 partners that touch each
    # other must sit on a convex plot corner.
    #  * Exact tiling (area == plot area): every side is covered by a
    #    neighbour or the boundary and a neighbour covers only one side, so
    #    the rect has two perpendicular sides on the boundary. Applies to
    #    every rect.
    #  * Hole allowed: valid only for BOUNDARY rects (they must share an edge
    #    with the plot boundary). Of the sides perpendicular to its boundary
    #    side, none can face the (interior) hole -- the hole would reach the
    #    boundary at that side's end point -- and the 2 partners cover at
    #    most 2 sides (they touch each other, so they are not opposite), so
    #    the remaining perpendicular side is boundary too: a corner. An
    #    interior rect may have sides facing the hole, so it is excluded.
    #    The "hole would reach the boundary" step needs the boundary to run
    #    straight past the end of that side, which fails at a plot vertex
    #    (e.g. a reflex corner right beside the rect: the hole may sit there
    #    and touch the boundary only at that point). So in hole mode the
    #    placement filter is skipped for any rect whose box touches a plot
    #    vertex, and for a rect that meets the boundary only at a point (the
    #    argument needs a shared edge segment), and the exhaustive corner clause in `seed_step` (which
    #    assumes the plain rule) is off.
    # Sufficient, not necessary. `placements_of` rejects these rects away
    # from convex plot corners and `seed_step` tries them first.
    triangle_corner_ids = {
        i for i, ps in required_partners.items()
        if len(ps) == 2 and ps[1] in required_partners.get(ps[0], [])
        and (not hole_mode or (i not in interior and require_boundary_touch))
    }

    placements: Dict[int, Placement] = {}
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
        touches = _touches_boundary(rx, ry, w, h, boundary)
        if idx in interior:
            return not touches
        return touches or not require_boundary_touch

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
                candidate = box(rx, ry, rx + uw, ry + uh)
                if not _rect_fits(free_shape, candidate):
                    continue
                if (idx in triangle_corner_ids and not _at_plot_corner(rx, ry, uw, uh, convex_vertices)
                        and not (hole_mode and not hole_corner_rules
                                 and (not _shares_boundary_edge(rx, ry, uw, uh, boundary)
                                      or _vertex_in_closed_box(rx, ry, uw, uh, plot_vertices)))):
                    continue
                if not boundary_ok(idx, rx, ry, uw, uh):
                    continue
                if not required_adjacency_ok(idx, rx, ry, uw, uh):
                    continue
                new_free = free_shape.difference(candidate)
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
        if not weak_forward_check:
            return False
        # weakened check: a room may slide along a wall and have no corner
        # on a free-space corner yet. Look for ANY lattice position where
        # it fits and matches the required adjacency exactly (cheap
        # adjacency / boundary tests first, shapely fit last).
        w0, h0 = sizes[q_idx]
        fminx, fminy, fmaxx, fmaxy = free_shape.bounds
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
                    if _rect_fits(free_shape, box(rx, ry, rx + uw, ry + uh)):
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
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids() and interior_ready(i) and eligible(originator, i)]
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
        anchor_verts = [p for p in verts if boundary.distance(Point(p)) < 1e-9]
        if interior_phase():
            anchor_verts = list(verts)   # interior rooms sit inside enclosed pockets
        remaining = ready_ids()
        # exhaustive corner clause: every layout has each unplaced
        # triangle-corner rect on a free convex plot corner. Seed ONE of
        # them (the one with the fewest corner placements) at every such
        # corner; if all of those seeds fail, no layout exists: stop.
        corner_ids = [i for i in remaining if i in triangle_corner_ids] if (not hole_mode or hole_corner_rules) else []
        if corner_ids:
            corner_anchors = [
                p for p in anchor_verts
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
                    ((idx, opts) for d, sc, _, idx, opts in scored if (d, sc) == (level, level_score)),
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
    defer_on = [bool(interior) and border_first]

    def interior_phase() -> bool:
        """True once ordering is on and every boundary room is placed."""
        return defer_on[0] and all(b in placements for b in border_ids)

    def interior_ready(idx) -> bool:
        return idx not in interior or not defer_on[0] or interior_phase()

    def eligible(originator, idx) -> bool:
        """May `idx` be anchored at a frontier point exposed by `originator`?"""
        return not interior_corner_rule or originator not in interior or idx in interior

    def remaining_ids():
        return [i for i in sizes if i not in placements]

    def ready_ids():
        return [i for i in sizes if i not in placements and interior_ready(i)]

    def _on_region(point, region):
        return region.distance(Point(point)) < 1e-9

    def _region_fillable(region, remaining):
        """Necessary: some subset of the remaining rects must have total
        area == region area, and the region must be at least as large as
        the smallest remaining rect in both directions."""
        if not remaining:
            return False
        area = region.area
        areas = [sizes[i][0] * sizes[i][1] for i in remaining]
        if area > sum(areas) + 1e-6:
            return False
        minx, miny, maxx, maxy = region.bounds
        min_side = min(min(sizes[i]) for i in remaining)
        if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
            return False
        if all(abs(a - round(a)) < 1e-9 for a in areas) and abs(area - round(area)) < 1e-6:
            bits = 1
            for a in areas:
                bits |= bits << int(round(a))
            return bool((bits >> int(round(area))) & 1)
        return True

    def region_seed(free_shape, region):
        """Seed inside `region` only. Any exact tiling of the region has a
        tile with a corner at EVERY vertex of the region, so one vertex is
        enough: pick the one with the fewest placements (0 = dead end) and
        branch over every rect placeable there."""
        remaining = ready_ids()
        best = None
        for v in _polygon_vertices(region):
            _check_time()
            rows, total = [], 0
            for idx in remaining:
                opts = placements_of(idx, v, free_shape)
                if opts:
                    w, h = sizes[idx]
                    rows.append((len(required_partners.get(idx, [])), _slot_count(opts, v), -(w * h), idx, opts))
                    total += len(opts)
            if total == 0:
                return None
            if best is None or (total, v) < best[0]:
                best = ((total, v), v, rows)
        _, v, rows = best
        rows.sort(key=lambda r: r[:4])
        return [(r[0], r[1], v, [(r[3], r[4])]) for r in rows]

    def leftover_ok(free_shape) -> bool:
        """All rects placed: the leftover must be an interior hole -- no
        piece shares an edge segment with the plot boundary, and (if
        `single_hole`) it is one connected piece."""
        if free_shape.is_empty or free_shape.area < 1e-9:
            return True
        comps = [c for c in _free_space_components(free_shape) if c.area > 1e-9]
        if single_hole and len(comps) > 1:
            return False
        return not any(_touches_boundary_edge(c, boundary) for c in comps)

    def hole_prune_ok(free_shape, remaining) -> bool:
        """Hole-mode pruning (sound). (1) Free area is plot area - placed
        area, so the leftover is always the fixed budget. (2) Once the free
        space has split into several pieces, a piece that shares an edge
        with the plot boundary can never end up as the hole, so remaining
        rects must fill it except for at most the hole budget H (subset-sum
        window on area, and it must be big enough for the smallest
        remaining rect). (3) More than one
        interior piece that cannot be filled at all violates `single_hole`
        only if their combined area exceeds the budget -- already implied
        by (1), so not re-checked."""
        if free_shape.geom_type == "Polygon":
            return True
        comps = [c for c in _free_space_components(free_shape) if c.area > 1e-9]
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
        H = max(0.0, free_shape.area - sum(areas))
        need = 0.0
        for c in comps:
            if not _touches_boundary_edge(c, boundary):
                continue
            need += c.area
            if not remaining:
                return False
            minx, miny, maxx, maxy = c.bounds
            if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
                return False
            if bits is not None and abs(c.area - round(c.area)) < 1e-6 and abs(H - round(H)) < 1e-6:
                hi = int(round(c.area)); lo = max(0, hi - int(round(H)))
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
        fminx, fminy, fmaxx, fmaxy = free_shape.bounds
        out = []
        # general triangle shortcut where it applies: two placed required
        # partners of idx that require each other and whose union is not a
        # rectangle -> idx nestles at their reflex corner / shared line
        if triangle_shortcut:
            ps = [q for q in required_partners.get(idx, []) if q in placements]
            pts = None
            for i1 in range(len(ps)):
                for i2 in range(i1 + 1, len(ps)):
                    a, b = ps[i1], ps[i2]
                    if b not in required_partners.get(a, []):
                        continue
                    if _union_is_rectangle(placements[a], placements[b]):
                        continue
                    pts = (pts or [])
                    pts.extend(_reflex_corners(placements[a], placements[b]))
                    for shared in _shared_corners(placements[a], placements[b]):
                        line = _perpendicular_search_line(placements[a], placements[b], shared)
                        if line is not None:
                            pts.extend(_anchor_points_on_line(free_shape, *line))
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
                    if not _rect_fits(free_shape, box(rx, ry, rx + uw, ry + uh)):
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
            result = grid_fill(free_shape.difference(box(rx, ry, rx + uw, ry + uh)),
                               [i for i in remaining if i != idx])
            if result is True or result is None:
                return result
            del placements[idx]
        return False

    def solve(free_shape, frontier):
        nodes[0] += 1
        _check_time()
        remaining = remaining_ids()
        if (interior_grid and remaining and all(i in interior for i in remaining)
                and all(b in placements for b in border_ids)):
            # every boundary rect is down: free space still touching the plot
            # boundary can never be filled by interior rects
            if any(c.area > 1e-9 and c.intersects(boundary) for c in _free_space_components(free_shape)):
                return False
            return grid_fill(free_shape, remaining)
        if not remaining:
            if hole_mode and not leftover_ok(free_shape):
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

        active = None
        if hole_mode and not hole_prune_ok(free_shape, remaining):
            return False
        if free_shape.geom_type != "Polygon" and not hole_mode:
            comps = _free_space_components(free_shape)
            if len(comps) > 1:
                if defer_on[0] and not interior_phase():
                    # pockets away from the plot boundary wait for the interior phase
                    comps = [c for c in comps if c.intersects(boundary)]
                if not comps:
                    active = None
                else:
                    active = min(comps, key=lambda c: c.area)
            if active is not None:
                if not _region_fillable(active, remaining):
                    return False
                on = [f for f in frontier if _on_region(f[0], active)]
                off = [f for f in frontier if not _on_region(f[0], active)]
                frontier = on + off

        if frontier and (active is None or _on_region(frontier[0][0], active)):
            # plain FIFO over the frontier
            (point, originator), *rest = frontier
            candidates = [i for i in required_partners.get(originator, []) if i in remaining and interior_ready(i) and eligible(originator, i)]
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
                        if rect_union and not rect_union_try:
                            # a and b join into one rectangle: no reflex
                            # corner for a third room to nestle into, so
                            # the shortcut does not apply -- the shared
                            # partner is placed later by the normal search
                            return resolve_chain(rest_pairs, extra_points, cur_free)
                        common = [
                            t for t in required_partners.get(a, [])
                            if t in required_partners.get(b, []) and t in remaining_ids() and interior_ready(t)
                            and eligible(a, t) and eligible(b, t)
                        ]
                        if not common:
                            return resolve_chain(rest_pairs, extra_points, cur_free)

                        points_to_try = []
                        points_to_try.extend(_reflex_corners(placements[a], placements[b]))
                        for shared in _shared_corners(placements[a], placements[b]):
                            line = _perpendicular_search_line(placements[a], placements[b], shared)
                            if line is not None:
                                points_to_try.extend(_anchor_points_on_line(cur_free, *line))

                        any_option = False
                        # rectangular union: the forced placement is only an
                        # attempt; when it fails we fall through to the
                        # normal unforced continuation instead of declaring
                        # a dead end
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
                        if rect_union:
                            return resolve_chain(rest_pairs, extra_points, cur_free)
                        if not any_option:
                            return False
                        return False

                    result = resolve_chain([(originator, cand)] if triangle_shortcut else [], new_points, new_free)
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
        per_point = region_seed(free_shape, active) if active is not None else seed_step(free_shape)
        if not per_point:
            return False
        for _deg, _score, point, tied in per_point:
            for seed, opts in tied:
                if seed not in remaining_ids() or not interior_ready(seed):
                    continue
                for (rx, ry, uw, uh, new_free) in opts:
                    placements[seed] = (rx, ry, uw, uh)
                    new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                    result = solve(new_free, new_points + (frontier if active is not None else []))
                    if result is True or result is None:
                        return result
                    del placements[seed]
        return False

    found_before = len(collect_solutions) if collect_solutions is not None else 0
    try:
        result = solve(poly, [])
        if defer_on[0]:
            nothing = (len(collect_solutions) == found_before) if collect_solutions is not None else (result is False)
            if nothing and result is not True:
                # phase 2: no border-first ordering
                defer_on[0] = False
                placements.clear()
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
            require_boundary_touch=True,
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
            require_boundary_touch=True,
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