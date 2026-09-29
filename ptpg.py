"""
Exact backtracking packer: places a set of rectangles inside a
rectilinear polygon plot with no overlaps and no rectangle crossing
the plot boundary.

The one solver is `pack_rectangles_frontier`, a PTPG-driven frontier
search built around `required_adjacency` (a set of (id_a, id_b) pairs
over the same keys as the `rects` dict -- e.g. from a PTPG, a planar
adjacency graph specifying which rooms must be next to each other).
Rather than re-scanning the whole free polygon's boundary at every
step, it grows outward from rectangles already placed, using the PTPG
graph itself to decide what to try next; see the "Frontier / PTPG-
driven search" section below for the full algorithm notes.

Every candidate placement is anchored at a live vertex of the current
free polygon, and every placed rectangle must touch the plot's
original boundary along at least part of one edge
(`require_boundary_touch`). A placement is rejected outright if it
would fragment the remaining free space into more pieces than the
still-remaining rects can close -- either by exact tiling, or, once
one has already been claimed, by being set aside as the ONE leftover
hole `require_single_hole` still allows (see `_fill_pockets_shortcut`).
The final layout's touch-graph must match `required_adjacency`
EXACTLY: every listed pair ends up sharing a positive-length edge, and
no unlisted pair does -- checked incrementally as each rectangle is
placed, against every rect already down, so a violating branch is
pruned immediately rather than only caught at the final leaf.

Rectangles are identified by explicit ids everywhere (`rects` is a dict
{id: (w, h)}, not a plain list), so a rect's id is never ambiguous or
tied to its position in a list -- the same id is used in `rects`,
`placements`, and `required_adjacency`.

Multi-solution mode
--------------------
Passing `collect_solutions` (a list) makes the search keep going after
each valid full packing instead of stopping at the first one, so it
can return several genuinely different layouts (deduplicated by
geometry, not by which rectangle index landed where).
"""
import time
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

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
    """The concave (reflex, interior angle > 180 deg) vertices of the PLOT's
    own boundary, as given -- the inner corner(s) of an L-shaped or
    staircase plot, where the boundary turns inward rather than outward.
    Handles either vertex winding (clockwise or counterclockwise) by first
    reading the polygon's own orientation off its signed area (shoelace
    formula): positive means counterclockwise, negative clockwise, and
    which turn direction counts as reflex flips accordingly."""
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
    # Cheap bbox reject before paying for the real (much more expensive)
    # shapely intersection -- a candidate whose own bounding box sticks out
    # past free_shape's can never fit, and this case is extremely common
    # when trying every orientation x offset off a point (especially the
    # pocket-tiling search, which tries many candidates per point against a
    # small pocket). `.bounds` is O(1) on shapely geometries.
    fminx, fminy, fmaxx, fmaxy = free_shape.bounds
    cminx, cminy, cmaxx, cmaxy = candidate.bounds
    if cminx < fminx - 1e-9 or cminy < fminy - 1e-9 or cmaxx > fmaxx + 1e-9 or cmaxy > fmaxy + 1e-9:
        return False
    return abs(free_shape.intersection(candidate).area - candidate.area) < 1e-6


def _shares_boundary_edge(rx: float, ry: float, w: float, h: float, boundary, tol: float = 1e-9) -> bool:
    """True if any of the rectangle's four edges overlaps the plot boundary by more than a point."""
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
    # vertical adjacency: one's right edge meets the other's left edge
    if abs((ax + aw) - bx) < tol or abs((bx + bw) - ax) < tol:
        overlap = min(ay + ah, by + bh) - max(ay, by)
        if overlap > tol:
            return True
    # horizontal adjacency: one's top edge meets the other's bottom edge
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
    """Raw connected pieces of free space (no erosion -- exact boundaries),
    largest first."""
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
# Instead of re-scanning the WHOLE free shape's boundary for the next
# most-constrained rectangle at every step, this grows outward from
# rectangles already placed, using the PTPG graph itself to decide what
# to try next:
#
#  1. Seed: pick a rectangle via whole-boundary MRV (most geometrically
#     constrained size, tie-broken by area), and place it at one of its
#     live boundary anchor points.
#  2. That placement exposes at most 2 NEW anchor points (its two corners
#     adjacent to the one it was anchored at, excluding the diagonal
#     corner). At each such point, only rects required to be adjacent to
#     the rect that exposed it are considered -- not the whole remaining
#     set -- ranked by the same MRV idea but scored locally (how many
#     orientations/positions work at THIS point).
#  3. Triangle shortcut: whenever the rect just placed (rect2) and the rect
#     that exposed its anchor point (rect1) share a common required
#     partner (rect3) -- a triangle in the PTPG graph -- rect3 is placed
#     (mandatorily, before anything else) at the reflex corner of rect1
#     and rect2's L-shaped union (the point where their far edges meet at
#     a right angle; this can be an interior point, not necessarily on the
#     plot boundary). If no orientation of rect3 fits there, or every
#     orientation that fits fails the boundary-touch requirement, this
#     whole rect2 placement is invalid and backtracked -- not just rect3.
#  4. Placement continues growing the frontier this way; whenever the
#     frontier is fully exhausted (every open point has no remaining
#     PTPG-adjacent candidate, or none fit), a fresh rectangle is seeded
#     via step 1 again, so disconnected parts of the PTPG graph (or slack
#     area with no adjacency requirement at all) still get filled.
# --------------------------------------------------------------------------

def _point_on_rect_edge(vx: float, vy: float, rx: float, ry: float, w: float, h: float, tol: float = 1e-9) -> bool:
    """True if (vx, vy) lies on one of the 4 edges of the rect (rx, ry, w,
    h) -- anywhere along the edge segment, not just at its endpoints."""
    on_vert_edge = (abs(vx - rx) < tol or abs(vx - rx - w) < tol) and ry - tol <= vy <= ry + h + tol
    on_horiz_edge = (abs(vy - ry) < tol or abs(vy - ry - h) < tol) and rx - tol <= vx <= rx + w + tol
    return on_vert_edge or on_horiz_edge


def _new_frontier_points(rx, ry, w, h, used_point, free_shape=None):
    """The rectangle's exposed corners for further growth: always the 2
    corners adjacent (by one shared edge) to the corner it was anchored at,
    plus the diagonal corner -- but ONLY if `free_shape` is given and that
    diagonal corner is still a genuine live vertex of the current free
    polygon. For ordinary corner-anchored growth off a convex free-polygon
    corner, the diagonal corner is normally swallowed by the placement and
    isn't live; but the triangle shortcut anchors at a REFLEX (concave)
    corner, where the diagonal corner can easily stay exposed -- so it must
    be checked, not assumed dead, whenever we can verify it.

    Also (when `free_shape` is given): any OTHER live vertex of the free
    polygon that lies somewhere along this rect's own edges, even though
    it isn't one of the rect's 4 corners. This covers a real, previously
    missed case -- a PLOT boundary corner (the polygon's own reflex
    corner, say) can sit partway along a just-placed rect's edge without
    being a corner of that rect at all; a required partner anchored there
    would touch the boundary AND this rect simultaneously, satisfying
    both a boundary-touch requirement and the adjacency in one placement,
    and that anchor point is otherwise never generated or tried."""
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
            continue  # already handled above (adjacent/diagonal/used_point)
        if _point_on_rect_edge(vx, vy, rx, ry, w, h):
            result.append((vx, vy))
    return result


def _reflex_corner(box_a: Placement, box_b: Placement):
    """The inner (concave) corner of the L-shaped union of two touching,
    perpendicularly-arranged rectangles -- the point where one's far edge
    meets the other's far edge at a right angle. Returns None if the two
    boxes don't form a proper L (e.g. they're collinear/aligned into one
    straight strip, which has no reflex corner)."""
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
    """Every point that is a corner of BOTH box_a and box_b, if the two
    boxes meet exactly corner-to-corner (flush along their touching edge)
    at one or both ends -- [] if no such point exists (the ordinary case,
    where they meet at a corner of only one of them, per _reflex_corner).
    When box_a and box_b are the SAME size along their touching edge (e.g.
    stacked directly on top of each other, equal width), they're flush at
    BOTH ends at once, giving two distinct shared corners, not one --
    returning only one (as an earlier version of this function did) would
    silently drop the other, and with it every placement anchored there:
    see the regression this was written to fix, where the dropped corner
    was the only one with any live free-space vertex left to anchor on."""
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
    """When two rects meet exactly at a shared corner, they touch along one
    axis (say a shared x, with y ranges overlapping); the OTHER axis, at
    the shared corner's coordinate on it, is a line that an edge of BOTH
    rects lies flush on (e.g. both bottom edges at the same y). A third
    rect that must touch both of them can be anchored anywhere along that
    line, not just at the shared corner itself -- returns (axis, value)
    where axis is 'x' or 'y', or None if the two boxes don't actually
    touch along a positive-length edge (shouldn't happen when they're
    required partners)."""
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
    """Live vertices of the current free polygon lying on the line x=value
    (axis='x') or y=value (axis='y')."""
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
    node_limit: int = 300_000,
    verbose: bool = False,
    snapshots=None,
    snapshot_limit=None,
    collect_solutions: Optional[list] = None,
    max_solutions: Optional[int] = None,
):
    """PTPG-driven frontier search -- the sole solver in this module, see
    the module docstring and the "Frontier / PTPG-driven search" notes
    above. Returns (found, placements_or_None, nodes_explored). found is
    True / False / None (True=feasible, False=proven infeasible, None=hit
    node_limit / snapshot_limit before finishing).

    `required_adjacency` (an iterable of (id_a, id_b) pairs over the same
    ids as `rects`) is optional -- `None` or `[]` means no adjacency
    constraint at all (every rect is then its own disconnected "region",
    each re-seeded independently via `seed_step`; every id still needs
    a placement, and `require_single_hole`/`require_boundary_touch` still
    apply, they just aren't guided by any PTPG graph).

    Pass `collect_solutions` (a list) to keep searching past the first
    complete, valid placement instead of stopping there: each full
    placement found is appended (as its own dict, order-of-discovery) and
    the search backtracks to look for a genuinely different one, up to
    `max_solutions` (None = exhaust the whole search space). `found` and
    the returned placements still describe the FIRST solution only, same
    as when `collect_solutions` isn't given -- the rest live in the list."""
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
    # The ONE pocket (if any) this branch has set aside, untouched, as the
    # final leftover hole `require_single_hole` still permits -- see
    # `_fill_pockets_shortcut` below. None until some branch claims it;
    # reset back to None whenever that branch's choice is undone, exactly
    # like `placements` itself, so it's always consistent with whichever
    # placements are currently live.
    hole_geom = [None]

    def log(msg):
        if verbose:
            print("  " * len(placements) + msg)
        if snapshots is not None and (snapshot_limit is None or len(snapshots) < snapshot_limit):
            snapshots.append((msg, dict(placements)))

    def required_adjacency_ok(idx, rx, ry, w, h) -> bool:
        cand = (rx, ry, w, h)
        partners = required_partners[idx]
        for j, placed in placements.items():
            if _rects_touch(cand, placed) != (j in partners):
                return False
        return True

    def single_hole_ok(new_free) -> bool:
        """Not just 'exactly one component': a placement that splits off one
        or more small pockets is still viable if EACH pocket can be tiled --
        exactly, respecting boundary-touch and required_adjacency_ok -- by
        some combination of the still-remaining rects (see `_tile_pocket`).
        A pocket no longer has to match a single remaining rect's size: two
        or more rects placed together, honoring their own adjacencies, can
        close it too. Running the real tiling search here, though, would
        mean paying for it on every candidate placement examined anywhere
        in the outer search (MRV scoring alone calls this many times over
        for every point x every remaining rect) -- far too often to afford.
        So this is only a cheap NECESSARY pre-filter (each pocket's area
        can't exceed the total area still available to fill it, and a
        pocket that isn't even a candidate for a real tiling -- e.g. one
        with a dimension smaller than every remaining rect's short side --
        is rejected outright); the real, sufficient check is deferred
        entirely to `_fill_pockets_shortcut`, run once only when a
        placement actually gets committed.

        `require_single_hole` only requires the FINAL layout to have at
        most one leftover chunk of free space -- it does NOT require every
        pocket that splits off mid-search to get tiled. So a pocket that
        fails the tileability checks below isn't automatically a dead end:
        if this branch hasn't claimed its one allowed permanent hole yet
        (`hole_geom[0] is None`), exactly one such pocket is still a live
        option (`_fill_pockets_shortcut` will actually try freezing it).
        Only a SECOND pocket that simultaneously fails is a sure reject --
        two untileable pockets can't both become the one permitted hole."""
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
        anchored at `point`, both orientations, all 4 quadrant offsets --
        filtered by fit, boundary-touch, adjacency and single-hole.
        `check_hole=False` skips the single_hole_ok filter: `_tile_pocket`
        passes this, since it already explicitly recurses into whatever
        sub-pieces a placement leaves behind -- running single_hole_ok's own
        (now full-blown tiling-search) filter on top of that would just
        redo the exact same recursive search a second time at every level,
        for no benefit.

        Checks run cheapest-first: `boundary_ok` and `required_adjacency_ok`
        are plain-Python tests against the candidate's own coordinates and
        the (already-placed) `placements` dict, so they're checked right
        after the geometric fit test -- BEFORE paying for the shapely
        `.difference()` that builds `new_free`, which only `single_hole_ok`
        actually needs. This is called very often (every candidate id x
        every anchor point, from `seed_step`, the frontier loop, and
        `_tile_pocket`), so skipping the difference() call on the common
        case of a boundary/adjacency reject is a real, cheap win."""
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
        """Could the not-yet-placed `q_idx` be placed ANYWHERE at all right
        now, given the current free space? Checked across every live
        vertex of the free polygon, since this search only ever anchors
        placements at those. `required_adjacency_ok` checks EXACTLY
        against every already-placed rect (touching it iff it's a required
        partner), so any placement `placements_of` returns already
        satisfies every adjacency `q_idx` currently has to already-placed
        rects -- this plain existence check IS the full reachability
        check, not merely a check against one specific rect (which is what
        went wrong last time: restricting the search to vertices on one
        particular rect's own boundary missed that a valid anchor for
        `q_idx` can belong to a completely different, already-placed rect
        and still end up touching the one we care about).

        SOUND, not a heuristic: free space only shrinks as placements
        accumulate, so a rect that can't be placed anywhere right now
        can't be placed anywhere later in this branch either -- with the
        one caveat that already applies throughout this file: a currently
        uninterrupted stretch of free boundary can still split into a new
        live vertex later (once some other rect lands nearby), exposing an
        anchor point that doesn't exist yet. That's a pre-existing,
        accepted limitation of this whole search's anchor-at-a-live-vertex
        convention, not something new introduced here."""
        for v in _polygon_vertices(free_shape):
            if placements_of(q_idx, v, free_shape):
                return True
        return False

    # Total work budget for `_tile_pocket`, shared across every probe for
    # the whole `pack_rectangles_frontier` call (feasibility probes from
    # `single_hole_ok` and real commits from `_fill_pockets_shortcut`
    # alike) -- NOT reset per call. `single_hole_ok` runs this search as a
    # provisional feasibility probe on every candidate placement that
    # splits off a pocket, which can happen very often; a per-call budget
    # bounds only one probe, but a hostile/complex board can still trigger
    # thousands of probes, so the budget has to bound the whole run to
    # avoid hanging. Once exhausted, every further probe is treated as
    # infeasible (conservative -- may over-reject) rather than searching
    # forever.
    tile_budget = [150000]

    def _tile_pocket(pocket_shape, depth_limit=3):
        """Try to exactly close `pocket_shape` -- a connected piece of free
        space split off by some placement -- using one or more of the still-
        remaining rects. A pocket no longer has to be a single rect's exact
        size: this anchors a remaining rect at a live vertex of the pocket
        (via the same `placements_of`, scoped to the pocket as the free
        universe), and if placing it leaves a sub-piece of the pocket still
        open, recurses into that sub-piece and tries to tile it too --
        continuing until the whole pocket is covered or every option is
        exhausted. Each rect is checked with the same boundary_ok and
        required_adjacency_ok rules as everywhere else, against everything
        already placed, INCLUDING whatever earlier rects this same tiling
        attempt has already tentatively placed (they live in `placements`
        like any real placement while this search is in progress). Returns
        the list of forced ids on success (left applied to `placements`),
        or None on failure, having rolled back everything this call placed.
        `tile_budget` bounds the total work of one top-level tiling attempt
        (reset by the caller before the first call) -- without it, this
        search is run as a *feasibility probe* inside `single_hole_ok` for
        every candidate placement anywhere in the outer search, and an
        unbounded one can blow up badly; running out of budget is treated
        as failure (conservative: may reject a pocket that a deeper search
        could have tiled)."""
        if pocket_shape.is_empty or pocket_shape.area < 1e-9:
            return []
        if depth_limit <= 0:
            return None
        remaining = remaining_ids()
        if not remaining:
            return None
        # Cheap prunes before generating any shapely candidates at all: if
        # what's left can't even cover the pocket's area, or a given rect
        # can't fit the pocket's bounding box in EITHER orientation, there's
        # no point paying for `placements_of`'s box/intersection calls.
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
                    tile_budget[0] -= 1
                    if tile_budget[0] <= 0:
                        return None
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
        budget_before = tile_budget[0]
        sub_forced = _tile_pocket(pocket)
        if _TILE_DEBUG:
            import sys as _sys_dbg
            print(f"[fill_pockets] call#{_fp_calls[0]} pocket_area={pocket.area:.2f} "
                  f"remaining={len(remaining_ids())} budget_used={budget_before - tile_budget[0]} "
                  f"result={'FAIL' if sub_forced is None else sub_forced} budget_left={tile_budget[0]}",
                  file=__import__('sys').stderr)
        return sub_forced

    def _fill_pockets_shortcut(free_shape):
        """A placement can split the free space into a main piece (the
        largest, by area) plus one or more small pockets. Yields every
        viable way to resolve them, best first, as (forced_ids, active_
        free_shape) pairs -- the caller `for`-loops over this and recurses
        into `solve` on each in turn:

          1. Tile EVERY pocket exactly, via `_tile_pocket` -- the ordinary
             case, and the only one tried once this branch has already
             claimed its one allowed hole (see below).
          2. If that fails and `hole_geom[0]` is still None (no hole
             claimed anywhere yet in this branch), try, for each pocket in
             turn: freeze THAT ONE pocket -- leave it untouched forever,
             excluded from the active free_shape handed to the rest of the
             search -- and tile every other pocket as usual. This is what
             lets a genuinely valid layout stand, where `require_single_
             hole` only demands the FINAL layout have at most one leftover
             region: that region doesn't have to be whatever's left when
             every rect runs out, it can just as well be a pocket that
             split off early and was always meant to stay empty.

        Mutates `placements` (and `hole_geom[0]`) to match whichever
        alternative is currently yielded; a caller that keeps looping past
        a failed alternative triggers this generator's own cleanup (the
        code after the `yield`) before it produces the next one, so
        failed attempts are always rolled back automatically -- a caller
        that `return`s while an alternative is live correctly keeps it."""
        comps = _free_space_components(free_shape)
        if len(comps) <= 1:
            yield [], free_shape
            return
        main = comps[0]
        pockets = comps[1:]

        def tile_all(skip_index=None):
            """Tile every pocket except `pockets[skip_index]`. Returns
            (forced_ids, active_free_shape) with `placements` already
            mutated to match, or None having rolled back anything this
            attempt placed."""
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
            # Every tiled pocket is fully consumed (that's what "tiled"
            # means); a skipped pocket is deliberately left OUT of the
            # active free_shape too (it's the frozen hole, not part of the
            # region the rest of the search keeps growing into) -- either
            # way, `main` alone is the right active free_shape to hand on.
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
        """MRV constrainedness given an already-computed `placements_of`
        result: number of distinct quadrant slots (up/down x left/right
        off the anchor point) among its options. A rect that fits one way
        in a given quadrant AND fits (via the other orientation) in that
        SAME quadrant isn't more free than one that only fits one way --
        rotating in place isn't an extra degree of freedom worth scoring,
        it's just a fallback to try if the first orientation doesn't
        work -- so this counts slots, not raw orientation x quadrant
        combinations."""
        px, py = point
        slots = set()
        for (rx, ry, w, h, _new_free) in opts:
            qx = 0 if abs(rx - px) < 1e-9 else 1
            qy = 0 if abs(ry - py) < 1e-9 else 1
            slots.add((qx, qy))
        return len(slots)

    def local_score(idx, point, free_shape):
        """`_slot_count` for `idx` at `point`, computing its placements_of
        result fresh -- use this only when that result isn't already
        available from elsewhere (e.g. `point_score` below, scoring a
        freshly-exposed point no one has evaluated yet); when it IS
        already in hand (see the frontier loop and `seed_step`), call
        `_slot_count` directly on it instead of recomputing here."""
        return _slot_count(placements_of(idx, point, free_shape), point)

    def point_score(point, originator, free_shape):
        """MRV score for a frontier point itself: the score of whichever
        PTPG-eligible candidate is most constrained there. A point with no
        eligible candidate at all scores as +inf (deprioritized -- it'll
        just get dropped when its turn comes)."""
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids()]
        if not candidates:
            return float("inf")
        return min(local_score(i, point, free_shape) for i in candidates)

    def order_points(points_with_originator, free_shape):
        """Sort newly-exposed (point, originator) pairs by MRV -- most
        constrained point first -- instead of arbitrary corner order, so
        the frontier actually grows toward the tightest spot next, not
        whichever corner happened to come first geometrically."""
        return sorted(points_with_originator, key=lambda po: point_score(po[0], po[1], free_shape))

    def seed_step(free_shape):
        """Point-first seed selection, mirroring how frontier growth already
        works: fix ONE anchor point first, then rank every remaining id that
        can actually be placed there, and only after exhausting every
        degree-tied candidate at that point do we ever consider a different
        point. This replaces the old candidate-first version (which tried
        one id across all its points before ever trying the next id) --
        per explicit user correction: tied candidates at the SAME point
        must be tried (largest area first) before backtracking to a
        different point.

        Ground truth on real floorplans shows actual corner rooms are
        almost always the LOWEST-degree rooms in the PTPG graph, so degree
        in required_adjacency is still the primary rank; among same-degree
        ids at a point, geometric local_score (fewest placements = most
        constrained) is the tie-break, then largest area first.

        One more wrinkle, ahead of that feasibility tie-break: whenever a
        point's best candidate is degree-1 (touches only one other rect --
        the strongest signal of an actual corner room), a point that is
        also one of the PLOT's own inner-L corners (its reflex/concave
        vertices, `inner_l_points`) is preferred over an equally-scored
        point that isn't. A degree-1 room is exactly the kind that tends to
        sit in a plot's own inward notch on real floorplans, so this
        breaks a (deg, feasibility) tie toward that corner first rather
        than leaving it to whatever order the free polygon's vertices
        happen to come out in. It only ever matters among degree-1 points
        (an inner-L point among degree-2+ points sorts exactly as before)."""
        verts = _polygon_vertices(free_shape)
        anchor_verts = [p for p in verts if boundary.distance(Point(p)) < 1e-9]
        remaining = remaining_ids()
        per_point = []
        for p in anchor_verts:
            scored = []
            for idx in remaining:
                opts = placements_of(idx, p, free_shape)
                if not opts:
                    continue
                deg = len(required_partners.get(idx, []))
                w, h = sizes[idx]
                # Keep `opts` alongside the score -- the caller needs the
                # exact same (idx, p, free_shape) result to actually try
                # placing this id, and placements_of() isn't cheap, so
                # computing it once here and handing it back avoids a
                # second identical call per tied candidate down below.
                scored.append((deg, _slot_count(opts, p), -(w * h), idx, opts))
            if not scored:
                continue
            scored.sort(key=lambda t: t[:4])
            # One entry per DEGREE LEVEL present at this point, not just the
            # lowest one: if every lowest-degree seed fails at every point,
            # the search falls back to the next-lowest degree (and so on)
            # instead of giving up. Since per_point is sorted by degree
            # first, all points for degree d are exhausted before any
            # degree-(d+1) seed is tried.
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
        if nodes[0] > node_limit:
            return None
        if snapshot_limit is not None and snapshots is not None and len(snapshots) >= snapshot_limit:
            return None
        remaining = remaining_ids()
        if not remaining:
            if require_single_hole:
                # Total leftover = whatever's still unfilled in the active
                # free_shape, PLUS the one frozen hole this branch may have
                # claimed (already excluded from free_shape itself, so the
                # two never double-count the same area) -- both together
                # must still add up to at most one connected piece.
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
                # Force backtracking to look for a genuinely different
                # complete placement (different box geometry, not just a
                # different id-assignment or discovery order onto the same
                # boxes), instead of stopping at the first one.
                return False
            return True

        if frontier:
            # Plain FIFO over the frontier: always branch on the point
            # queued first. (An MRV reorder here -- branch on whichever
            # pending point has the fewest valid options -- was tried and
            # found to regress example 2 (found=True -> False), so it was
            # reverted; per the user, that idea belongs only in `seed_step`
            # above, which already does exactly this "least-degree rect at
            # the least-feasible point" ordering for the first rectangle of
            # each newly-seeded region, tie-broken by descending area.)
            (point, originator), *rest = frontier
            candidates = [i for i in required_partners.get(originator, []) if i in remaining]
            if not candidates:
                return solve(free_shape, rest)
            # Compute each candidate's live placement options ONCE here --
            # both the MRV sort (via `_slot_count`) and the branching loop
            # right after it need the exact same (idx, point, free_shape)
            # result, and placements_of() isn't cheap (a shapely fit/diff
            # call per orientation x offset), so compute it once and reuse
            # rather than recomputing it a second time per candidate.
            cand_opts = {i: placements_of(i, point, free_shape) for i in candidates}
            # Most-constrained-first: fewest valid placements at this point wins.
            candidates.sort(key=lambda i: (_slot_count(cand_opts[i], point), -(sizes[i][0] * sizes[i][1])))

            for cand in candidates:
                for (rx, ry, w, h, new_free) in cand_opts[cand]:
                    placements[cand] = (rx, ry, w, h)
                    log(f"rect{cand} {sizes[cand]} @ ({rx:g},{ry:g}) [frontier from {originator}]")
                    new_points = order_points([(p, cand) for p in _new_frontier_points(rx, ry, w, h, point, new_free)], new_free)

                    # Forward-check: this placement just consumed part of
                    # the free space, possibly leaving one of `cand`'s own
                    # required partners, or one of `originator`'s OTHER
                    # required partners, with nowhere left to go at all
                    # (checked via `_rect_still_placeable`, which is a
                    # sound existence check, not restricted to any one
                    # rect's boundary -- see its docstring). Catch that now
                    # rather than relying on DFS to drill down to the dead
                    # point on its own, possibly many nodes later.
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
                        log(f"backtrack rect{cand} [forward-check: unplaceable]")
                        continue

                    # Triangle shortcut: whenever two already-placed rects
                    # share a required partner not yet placed, that partner
                    # must go in now, mandatorily -- at the reflex corner of
                    # the pair's L-union when they meet at a corner of only
                    # ONE of them (the ordinary case), or, when they meet
                    # exactly corner-to-corner (a corner of BOTH), by
                    # scanning the live anchor points along the line that an
                    # edge of both rects lies flush on.
                    #
                    # This used to also chain: after placing that partner,
                    # re-check IT against each original rect for its own
                    # unresolved shared partner, and so on. That chaining
                    # was tried and reverted -- confirmed by direct testing
                    # (re-running example 1 with it on vs. off) to be a net
                    # regression: forcing each newly-placed partner into an
                    # immediate, all-or-nothing placement compounds the
                    # shortcut's existing risk (killing a branch whose true
                    # solution defers that adjacency to a different,
                    # board-dependent point discovered later) at every
                    # additional link in the chain, which pruned away
                    # example 1's actual solution entirely (found=False)
                    # where the single-level version still finds it.
                    # `pending_pairs` is left as a list (rather than a
                    # single pair) only so the base case below still reads
                    # naturally as "no pair left to resolve"; it never grows
                    # past its initial one entry now.
                    def resolve_chain(pending_pairs, extra_points, cur_free):
                        if not pending_pairs:
                            for forced_ids, filled_free in _fill_pockets_shortcut(cur_free):
                                for f_idx in forced_ids:
                                    log(f"rect{f_idx} {sizes[f_idx]} @ {placements[f_idx][:2]} [pocket fill]")
                                result = solve(filled_free, extra_points + rest)
                                if result is True or result is None:
                                    return result
                                # loop continues -> the generator undoes
                                # forced_ids (and any frozen hole) itself
                                # before handing back its next alternative
                            return False

                        (a, b), *rest_pairs = pending_pairs
                        common = [
                            t for t in required_partners.get(a, [])
                            if t in required_partners.get(b, []) and t in remaining_ids()
                        ]
                        if not common:
                            return resolve_chain(rest_pairs, extra_points, cur_free)

                        # The touching segment between `a` and `b` has two
                        # ends: the genuine reflex (concave) corner of their
                        # L-union -- a corner of only ONE of them -- and,
                        # separately, whichever OTHER end(s) happen to be a
                        # corner of BOTH (only when they're also flush along
                        # a perpendicular line there, e.g. both bottom edges
                        # at the same y). When `a` and `b` are equal-sized
                        # along their touching edge, BOTH ends are
                        # corner-of-both at once -- two distinct shared
                        # corners, not one -- so every one of them gets its
                        # own perpendicular search line; using only the
                        # first can silently miss the only line that still
                        # has any live anchor point on it.
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
                                    any_option = True
                                    placements[t_idx] = (tx, ty, tw, th)
                                    log(f"rect{t_idx} {sizes[t_idx]} @ ({tx:g},{ty:g}) [triangle {a}-{b}]")
                                    t_points = order_points([(p, t_idx) for p in _new_frontier_points(tx, ty, tw, th, pt, t_free)], t_free)
                                    result = resolve_chain(rest_pairs, extra_points + t_points, t_free)
                                    if result is True or result is None:
                                        return result
                                    del placements[t_idx]
                                    log(f"backtrack rect{t_idx}")
                        if not any_option:
                            return False
                        return False

                    result = resolve_chain([(originator, cand)], new_points, new_free)
                    if result is True or result is None:
                        return result
                    del placements[cand]
                    log(f"backtrack rect{cand}")

            # Every candidate/placement at this point failed. If `originator`
            # has no OTHER pending point left anywhere in the frontier, this
            # was its last chance to get a required partner placed via its
            # own exposed corners -- that's a verified dead end for whoever
            # placed `originator`, so fail now instead of quietly falling
            # through to seeding fresh rects while `originator` sits there
            # unsatisfied. (If `candidates` was empty to begin with, nothing
            # was ever required here -- that's not a failure, just an unused
            # point, so we still fall through to `rest` normally.)
            still_pending_for_originator = any(o == originator for (_, o) in rest)
            if not still_pending_for_originator:
                return False
            return solve(free_shape, rest)

        # Frontier empty: seed a fresh rectangle, point-first -- fix the
        # single best anchor point, exhaust every degree/score-tied
        # candidate there (largest area first), and only fall back to the
        # next-best point if the whole tied group fails at this one.
        per_point = seed_step(free_shape)
        if not per_point:
            return False
        for _deg, _score, point, tied in per_point:
            for seed, opts in tied:
                if seed not in remaining_ids():
                    continue  # placed by an earlier point/candidate already tried this call
                for (rx, ry, uw, uh, new_free) in opts:
                    placements[seed] = (rx, ry, uw, uh)
                    log(f"rect{seed} {sizes[seed]} @ ({rx:g},{ry:g}) [seed @ {point}]")
                    new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                    for forced_ids, filled_free in _fill_pockets_shortcut(new_free):
                        for f_idx in forced_ids:
                            log(f"rect{f_idx} {sizes[f_idx]} @ {placements[f_idx][:2]} [pocket fill]")
                        result = solve(filled_free, new_points)
                        if result is True or result is None:
                            return result
                        # loop continues -> generator undoes forced_ids
                        # (and any frozen hole) before its next alternative
                    del placements[seed]
                    log(f"backtrack seed rect{seed}")
        return False

    result = solve(poly, [])
    if collect_solutions is not None:
        # `placements` at this point is whatever the search last had active
        # (mid-backtrack scratch state) -- the real answer is always the
        # accumulated list, regardless of why `solve` returned: `True`
        # means max_solutions was reached, `False` means the whole tree was
        # exhausted (every solution already collected), `None` means
        # node_limit cut it off early with whatever was found so far.
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
    """Canonical geometry-only key for deduplicating layouts (ignores which
    rectangle index landed where, only the final set of boxes matters)."""
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
        name="25x30 L-plot, notch 10x15 (9 rects)",
        polygon=[(0, 0), (25, 0), (25, 15), (15, 15), (15, 30), (0, 30)],
        # rects is a dict {id: (w, h)} -- dict(enumerate([...])) just turns
        # the readable list literal below into {0: (15,6), 1: (10,6), ...}
        # so every rect has an explicit id, used consistently in
        # `placements` and in `required_adjacency` pairs (no implicit
        # list-position matching to keep track of).
        rects=dict(enumerate([(15, 6), (10, 6), (15, 4), (15, 4), (8, 7), (7, 7), (5, 9), (16, 9), (4, 9)])),
        # required_adjacency is this example's PTPG: the exact touch-graph
        # of one verified solution (found under this example's own
        # settings), so it's guaranteed realizable.
        required_adjacency=[(0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 7), (2, 3),
                             (3, 6), (3, 7), (3, 8), (4, 5), (6, 7), (6, 8)],
    ),
    2: dict(
        name="24x24 L-plot, wide notch (8 rects)",
        polygon=[(0, 0), (24, 0), (24, 16), (16, 16), (16, 24), (0, 24)],
        rects=dict(enumerate([(24, 4), (8, 7), (8, 7), (13, 4), (7, 4), (4, 4), (16, 4), (16, 5)])),
        required_adjacency=[(0, 6), (0, 7), (1, 4), (1, 7), (2, 3), (2, 4), (3, 5), (3, 6), (5, 6)],
    ),
    3: dict(
        name="35x40 L-plot, 13 rects, 120-unit slack",
        polygon=[(0, 0), (35, 0), (35, 20), (22, 20), (22, 40), (0, 40)],
        rects=dict(enumerate([(8, 8), (13, 10), (14, 8), (17, 12), (12, 10), (5, 5), (5, 5),
                               (12, 10), (6, 5), (6, 5), (10, 10), (5, 7), (5, 5)])),
        required_adjacency=[(0, 2), (0, 12), (0, 3), (2, 3), (3, 12), (3, 11),
                            (12, 11), (3, 10), (11, 10), (10, 5), (10, 4), (5, 4), (5, 6), (6, 4), (4, 7), (7, 8),
                            (7, 9), (7, 1), (8, 9), (9, 1)],
    ),
    4: dict(
        name="24x20 L-plot, 9-rect floorplan, exact tiling",
        polygon=[(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)],
        rects=dict(enumerate([(6, 8), (6, 6), (3, 7), (6, 4), (6, 2), (4, 6), (11, 4), (9, 2), (3, 8)])),
        required_adjacency=[(0, 3), (0, 5), (1, 2), (1, 4), (2, 6), (3, 4), (3, 5), (6, 8), (7, 8)],
    ),
    5: dict(
        name="20x30 L-plot, 10-rect floorplan",
        polygon=[(0, 0), (20, 0), (20, 15), (10, 15), (10, 30), (0, 30)],
        rects=dict(enumerate([(6, 6), (11, 4), (1, 5), (2, 4), (9, 6), (6, 10), (5, 10), (9, 5), (10, 2)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)],
    ),
    6: dict(
        name="16x14 notched L-plot, 10-rect floorplan",
        polygon=[(0, 0), (16, 0), (16, 6), (9, 6), (9, 14), (0, 14)],
        rects=dict(enumerate([(6, 4), (3, 3), (3, 4), (5, 3), (3, 8), (3, 6), (5, 2), (3, 4), (2, 3), (6, 2)])),
        required_adjacency=[(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    ),
    7: dict(
        name="actual floorplan",
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
        name="actual floorplan",
        polygon=[(0, 7), (0, 13), (20, 13), (20, 0), (13, 0), (13, 7)],
        rects=dict(enumerate([(7, 4), (7, 3), (2, 3), (4, 4), (2, 3), (2, 2), (2, 5), (2, 2), (2, 4), (4, 5), (3, 4), (3, 4)])),
        required_adjacency=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9), (9, 10), (10, 11)],
    ),
    9: dict(
        name="actual floorplan",
        polygon=[(0, 0),(21,0),(21,6),(9,6),(9,18),(0,18)],
        rects=dict(enumerate([(6,8),(6,4),(2,2),(3,2),(4,5),(2,5),(3,5),(2,5),(3,3),(4,9)])),
        required_adjacency=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)],
    ),
    10: dict(
        name="actual floorplan",
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
    node_limit: int = 200_000,
    required_adjacency: Optional[Iterable[Tuple[int, int]]] = None,
):
    """
    `required_adjacency` defaults to None here, meaning "use this
    example's own PTPG" (each example in EXAMPLES carries its own
    `required_adjacency`). Pass an explicit iterable (including `[]`,
    meaning "no two rects may touch at all" under exact-match semantics)
    to use exactly that PTPG instead of the example's own. Pass
    `required_adjacency=False` to explicitly turn the constraint off
    (no adjacency requirement at all) -- `False` is a distinct sentinel
    from `None` (use example default) and from `[]` (require an empty
    touch-graph).

    Both the initial single-solution answer and the multi-solution
    search below run entirely through `pack_rectangles_frontier` -- the
    only solver in this module (see the module docstring).
    """
    example = EXAMPLES[choice]
    polygon, rects = example["polygon"], example["rects"]
    if required_adjacency is False:
        required_adjacency = None
    elif required_adjacency is None:
        required_adjacency = example.get("required_adjacency")

    print(f"\nRunning example {choice}: {example['name']}")
    print(f"  polygon: {polygon}")
    print(f"  rects:   {rects}")
    if required_adjacency:
        print(f"  required_adjacency (PTPG): {sorted(required_adjacency)}")

    # Initial answer: solve once and report it immediately, before the
    # (usually slower) multi-solution search below runs. Ctrl+C during this
    # phase just skips straight to the multi-solution search below.
    t0 = time.perf_counter()
    found0, placements0 = None, None
    try:
        found0, placements0, nodes0 = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            require_single_hole=True,
            require_boundary_touch=True,
            node_limit=node_limit,
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
    # Seed with the initial solve's result (if any) so it's never lost --
    # the multi-solution search recognizes it and won't re-record or lose
    # it if interrupted before finding anything new.
    solutions: list = [placements0] if (found0 is True and placements0) else []
    interrupted = False
    try:
        found, solutions, nodes = pack_rectangles_frontier(
            polygon, rects,
            required_adjacency,
            require_single_hole=True,
            require_boundary_touch=True,
            node_limit=node_limit,
            collect_solutions=solutions,
            max_solutions=max_solutions,
        )
    except KeyboardInterrupt:
        # `solutions` was mutated in place as the search went, so whatever
        # was found before the interrupt is still here to report and save.
        interrupted = True
        found = None
        nodes = "?"

    if not solutions:
        if interrupted:
            print(f"  interrupted before finding any solution (nodes explored: {nodes})")
        elif found is False:
            print(f"  proven infeasible: no solution exists under these constraints (nodes explored: {nodes})")
        else:
            print(f"  hit node_limit ({node_limit}) before finding any solution -- inconclusive (nodes explored: {nodes})")
        return

    if found is True:
        print(f"  found {len(solutions)} distinct solution(s), nodes explored: {nodes}")
    elif interrupted:
        print(f"  stopped by user -- showing {len(solutions)} of {max_solutions} distinct solution(s) found so far "
              f"(nodes explored: {nodes})")
    else:
        print(f"  hit node_limit ({node_limit}) before finishing -- showing {len(solutions)} of {max_solutions} "
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
        print(f"  {key}. {ex['name']}")

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