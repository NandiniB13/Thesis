"""
Exact backtracking packer for L-shaped, T-shaped (2 reflex angles), and general
rectilinear polygons (k >= 2 reflex angles) with interior rooms.

`pack_rectangles_frontier` is a PTPG-driven frontier search built around
`required_adjacency` (a set of (id_a, id_b) pairs over the same keys as `rects`,
representing which rooms must share a positive-length wall).

Key capabilities for T-shapes and general rectilinear polygons:
1. Reflex-Bridge Double-Overhang Anchoring (`_overhang_candidate_coords`):
   Handles rooms at a T-junction (or multi-reflex junction) that overhang an
   already-placed stem/arm neighbor on BOTH sides so none of the new room's
   4 corners lies on the neighbor.
2. Multi-Reflex Union Discovery (`_reflex_corners`):
   When two touching placed rooms form a T-union (one overhanging the other on
   both ends), extracts BOTH concave reflex corners for the triangle shortcut.
3. True 90-Degree Convex-Only Region Seeding (`_region_convex_and_reflex_vertices`):
   Filters out 180-degree collinear vertices left behind by Shapely `.difference()`
   and 270-degree reflex pass-through vertices when seeding disconnected regions.
4. Multi-Component Free-Space & PTPG Eligibility Pruning (`_all_regions_viable`):
   When a placement at a T-junction splits free space into 2, 3, or more regions,
   verifies that EVERY region is fillable by remaining rooms whose already-placed
   PTPG neighbors touch that region.
5. Active Interior-Room Viability Lookahead (`_interior_rect_still_viable`):
   Verifies that unplaced interior rooms (`interior_ids`) can still geometrically
   bridge across all of their already-placed PTPG neighbors inside `free_shape`.
"""
import ast
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

Size = Tuple[float, float]
Placement = Tuple[float, float, float, float]


class _SearchTimeout(Exception):
    """Raised by `_check_time` once `time_limit` is exceeded."""


def _ring_classified_vertices(coords: List[Tuple[float, float]], is_hole: bool = False, tol: float = 1e-9):
    """Classify ring vertices into (convex_set, reflex_set, non_collinear_list).
    Works for any winding order. For an interior hole ring (`is_hole=True`),
    interior angles of the polygon are inverted relative to the hole's interior."""
    pts = list(coords[:-1]) if (len(coords) > 1 and coords[0] == coords[-1]) else list(coords)
    # Remove consecutive duplicate points
    cleaned = []
    for p in pts:
        if not cleaned or abs(p[0] - cleaned[-1][0]) > tol or abs(p[1] - cleaned[-1][1]) > tol:
            cleaned.append((float(p[0]), float(p[1])))
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= tol and abs(cleaned[0][1] - cleaned[-1][1]) <= tol:
        cleaned.pop()
    n = len(cleaned)
    if n < 4:
        return set(), set(), cleaned
    area2 = sum(
        cleaned[i][0] * cleaned[(i + 1) % n][1] - cleaned[(i + 1) % n][0] * cleaned[i][1]
        for i in range(n)
    )
    ccw = area2 > 0
    convex, reflex, non_collinear = set(), set(), []
    for i in range(n):
        ax, ay = cleaned[i - 1]
        bx, by = cleaned[i]
        cx, cy = cleaned[(i + 1) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) <= tol:
            continue  # collinear (180-degree) vertex
        non_collinear.append((bx, by))
        is_left_turn = cross > tol
        is_convex_in_ring = is_left_turn if ccw else (not is_left_turn)
        if is_hole:
            is_convex_in_ring = not is_convex_in_ring
        if is_convex_in_ring:
            convex.add((bx, by))
        else:
            reflex.add((bx, by))
    return convex, reflex, non_collinear


def _polygon_vertices(shape: BaseGeometry) -> List[Tuple[float, float]]:
    polys = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    pts = set()
    for p in polys:
        if p.is_empty:
            continue
        pts.update((float(x), float(y)) for x, y in p.exterior.coords[:-1])
        for interior in p.interiors:
            pts.update((float(x), float(y)) for x, y in interior.coords[:-1])
    return list(pts)


def _region_convex_and_reflex_vertices(shape: BaseGeometry, tol: float = 1e-9) -> Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]:
    """True 90-degree convex and 270-degree reflex vertices of `shape`
    (collinear 180-degree vertices left by Shapely `.difference()` are excluded)."""
    polys = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    convex_all, reflex_all = set(), set()
    for p in polys:
        if p.is_empty:
            continue
        c_ext, r_ext, _ = _ring_classified_vertices(list(p.exterior.coords), is_hole=False, tol=tol)
        convex_all.update(c_ext)
        reflex_all.update(r_ext)
        for hole in p.interiors:
            c_int, r_int, _ = _ring_classified_vertices(list(hole.coords), is_hole=True, tol=tol)
            convex_all.update(c_int)
            reflex_all.update(r_int)
    return convex_all, reflex_all


def _polygon_reflex_vertices(polygon: List[Tuple[float, float]], tol: float = 1e-9) -> set:
    _, reflex, _ = _ring_classified_vertices(polygon, is_hole=False, tol=tol)
    return reflex


def _polygon_convex_vertices(polygon: List[Tuple[float, float]], tol: float = 1e-9) -> set:
    convex, _, _ = _ring_classified_vertices(polygon, is_hole=False, tol=tol)
    return convex


def _boundary_segments(polygon: List[Tuple[float, float]], tol: float = 1e-9):
    """Precompute axis-aligned boundary segments (horiz, vert) for O(1) pure-Python
    boundary-touch checks."""
    n = len(polygon)
    horiz = []  # (y, x_min, x_max)
    vert = []   # (x, y_min, y_max)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if abs(y1 - y2) < tol:
            horiz.append((float(y1), float(min(x1, x2)), float(max(x1, x2))))
        elif abs(x1 - x2) < tol:
            vert.append((float(x1), float(min(y1, y2)), float(max(y1, y2))))
    return horiz, vert


def _shares_boundary_edge_fast(rx: float, ry: float, w: float, h: float, horiz_segs, vert_segs, tol: float = 1e-9) -> bool:
    """Fast pure-Python check: True if any of the 4 edges of (rx, ry, w, h)
    overlaps the plot boundary by more than a point (> tol)."""
    rx2 = rx + w
    ry2 = ry + h
    for y, x1, x2 in horiz_segs:
        if abs(y - ry) < tol or abs(y - ry2) < tol:
            if min(rx2, x2) - max(rx, x1) > tol:
                return True
    for x, y1, y2 in vert_segs:
        if abs(x - rx) < tol or abs(x - rx2) < tol:
            if min(ry2, y2) - max(ry, y1) > tol:
                return True
    return False


def _at_plot_corner(rx: float, ry: float, w: float, h: float, convex_vertices, tol: float = 1e-9) -> bool:
    corners = ((rx, ry), (rx + w, ry), (rx, ry + h), (rx + w, ry + h))
    return any(
        abs(cx - vx) < tol and abs(cy - vy) < tol
        for (cx, cy) in corners for (vx, vy) in convex_vertices
    )


def check_area_match(polygon, rects: Dict[int, Size], tol: float = 1e-6) -> Tuple[bool, str]:
    plot_area = Polygon(polygon).area
    rect_area = sum(w * h for w, h in rects.values())
    if abs(rect_area - plot_area) > tol:
        return False, (f"total rect area {rect_area:g} != plot area {plot_area:g} "
                       f"(difference {rect_area - plot_area:+g})")
    return True, ""


def _rect_fits(free_shape: BaseGeometry, rx: float, ry: float, w: float, h: float, candidate: BaseGeometry) -> bool:
    if free_shape.is_empty:
        return False
    fminx, fminy, fmaxx, fmaxy = free_shape.bounds
    if rx < fminx - 1e-9 or ry < fminy - 1e-9 or rx + w > fmaxx + 1e-9 or ry + h > fmaxy + 1e-9:
        return False
    return abs(free_shape.intersection(candidate).area - (w * h)) < 1e-6


def _rects_touch(a: Placement, b: Placement, tol: float = 1e-9) -> bool:
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


def _rects_touch_or_corner(a: Placement, b: Placement, tol: float = 1e-9) -> bool:
    """True if two axis-aligned rectangles share a 1D wall segment OR a 0D corner point."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_gap = max(ax, bx) - min(ax + aw, bx + bw)
    y_gap = max(ay, by) - min(ay + ah, by + bh)
    return x_gap <= tol and y_gap <= tol


def _free_space_components(shape: BaseGeometry):
    if shape.is_empty:
        return []
    comps = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    return sorted(comps, key=lambda c: -c.area)


_ORIENTATION_OFFSETS = [(0, 0), (-1, 0), (0, -1), (-1, -1)]


def _candidate_offsets(w, h):
    return [(ox * w, oy * h) for ox, oy in _ORIENTATION_OFFSETS]


def _point_on_rect_edge(vx: float, vy: float, rx: float, ry: float, w: float, h: float, tol: float = 1e-9) -> bool:
    on_vert_edge = (abs(vx - rx) < tol or abs(vx - rx - w) < tol) and ry - tol <= vy <= ry + h + tol
    on_horiz_edge = (abs(vy - ry) < tol or abs(vy - ry - h) < tol) and rx - tol <= vx <= rx + w + tol
    return on_vert_edge or on_horiz_edge


def _new_frontier_points(rx, ry, w, h, used_point, free_shape=None):
    """Exposed corners and on-edge live vertices of a newly placed rectangle.
    Handles both standard corner-anchored placements and T-junction overhang
    placements (where `used_point` lies strictly inside an edge of the rectangle)."""
    corners = [(rx, ry), (rx + w, ry), (rx, ry + h), (rx + w, ry + h)]
    px, py = used_point
    if free_shape is None:
        return [
            c for c in corners
            if not (abs(c[0] - px) < 1e-9 and abs(c[1] - py) < 1e-9)
            and (abs(c[0] - px) < 1e-9 or abs(c[1] - py) < 1e-9)
        ]
    live_verts = _polygon_vertices(free_shape)
    result = []
    for (cx, cy) in corners:
        if any(abs(cx - vx) < 1e-9 and abs(cy - vy) < 1e-9 for (vx, vy) in live_verts):
            result.append((cx, cy))
    for (vx, vy) in live_verts:
        if abs(vx - px) < 1e-9 and abs(vy - py) < 1e-9:
            continue
        if any(abs(vx - cx) < 1e-9 and abs(vy - cy) < 1e-9 for (cx, cy) in corners):
            continue
        if _point_on_rect_edge(vx, vy, rx, ry, w, h):
            result.append((vx, vy))
    return result


def _reflex_corners(box_a: Placement, box_b: Placement, tol: float = 1e-9) -> List[Tuple[float, float]]:
    """All inner (concave) corners of the union of two touching rectangles.
    - L-union of two touching rectangles: 1 reflex corner.
    - T-union (one rectangle overhanging the other on both sides): 2 reflex corners.
    - Flush / collinear union: 0 reflex corners."""
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    xs = [x1, x1 + w1, x2, x2 + w2]
    ys = [y1, y1 + h1, y2, y2 + h2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    corners = sorted({
        (x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1),
        (x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2),
    })
    out = []
    for (cx, cy) in corners:
        if xmin + tol < cx < xmax - tol and ymin + tol < cy < ymax - tol:
            out.append((cx, cy))
    return out



def _reflex_corner(box_a: Placement, box_b: Placement):
    """Backwards-compatible wrapper returning the first reflex corner if any."""
    corners = _reflex_corners(box_a, box_b)
    return corners[0] if corners else None


def _shared_corners(box_a: Placement, box_b: Placement, tol: float = 1e-9):
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    corners_a = {(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)}
    corners_b = {(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)}
    out = []
    for ca in sorted(corners_a):
        for cb in sorted(corners_b):
            if abs(ca[0] - cb[0]) < tol and abs(ca[1] - cb[1]) < tol:
                out.append(ca)
    return out


def _perpendicular_search_line(box_a: Placement, box_b: Placement, shared_corner, tol: float = 1e-6):
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
    verts = _polygon_vertices(free_shape)
    idx = 0 if axis == "x" else 1
    return [p for p in verts if abs(p[idx] - value) < tol]


def _layout_key(placements: Dict[int, Placement]) -> tuple:
    return tuple(sorted(
        (round(x, 6), round(y, 6), round(w, 6), round(h, 6))
        for x, y, w, h in placements.values()
    ))


def _point_in_poly_fast(px: float, py: float, poly_pts: List[Tuple[float, float]]) -> bool:
    """Fast pure-Python ray-casting point-in-polygon check for grid cell centers."""
    inside = False
    n = len(poly_pts)
    j = n - 1
    for i in range(n):
        xi, yi = poly_pts[i]
        xj, yj = poly_pts[j]
        if (yi > py) != (yj > py) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _normalize_adjacency_pairs(required_adjacency) -> List[Tuple[int, int]]:
    """Normalize user-supplied adjacency input so a single pair like `(0, 1)` or
    a list of pairs `[(0, 1), (2, 3)]` or set `{(0, 1)}` all work seamlessly."""
    if not required_adjacency:
        return []
    if (
        isinstance(required_adjacency, (tuple, list))
        and len(required_adjacency) == 2
        and all(isinstance(x, (int, float)) for x in required_adjacency)
    ):
        return [(int(required_adjacency[0]), int(required_adjacency[1]))]
    out = []
    for item in required_adjacency:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
        else:
            raise ValueError(f"Invalid adjacency pair: {item!r}")
    return out


def _pack_mrv_corner_ptpg_single(
    polygon,
    rects,
    required_adjacency=None,
    *,
    require_boundary_touch=True,
    interior_ids=(),
    time_limit=60.0,
    collect_solutions=None,
    max_solutions=None,
    allow_corner_adj: bool = False,
    strict_exact_graph: bool = False,
    pure_mrv: bool = False,
    max_nodes: Optional[int] = None,
    seed_quadrant: Optional[Tuple[int, int]] = None,
    use_wall_len: bool = False,
):
    sizes = dict(rects)
    if not check_area_match(polygon, sizes)[0]:
        return False, collect_solutions, 0

    horiz_segs, vert_segs = _boundary_segments(polygon)
    convex_vertices = _polygon_convex_vertices(polygon)
    interior = set(interior_ids or ())
    unknown = interior - set(sizes)
    if unknown:
        raise ValueError(f"interior_ids references unknown rect ids: {sorted(unknown)}")

    adj_pairs = _normalize_adjacency_pairs(required_adjacency)
    enforce_adj = bool(adj_pairs)
    required_partners = {i: set() for i in sizes}
    for a, b in adj_pairs:
        if a == b:
            raise ValueError(f"required_adjacency has a self-pair: ({a!r}, {a!r})")
        if a not in sizes or b not in sizes:
            raise ValueError(f"required_adjacency references an unknown rect id ({a!r}, {b!r})")
        required_partners[a].add(b)
        required_partners[b].add(a)

    # Topological invariant: ONLY when solving a full exact maximal PTPG (`strict_exact_graph=True`),
    # a non-interior room with degree 2 whose two partners touch each other must sit in a 90-degree plot corner.
    triangle_corner_ids = set()
    if enforce_adj and strict_exact_graph and require_boundary_touch and not allow_corner_adj:
        for i, ps in required_partners.items():
            if i not in interior and len(ps) == 2:
                p0, p1 = tuple(ps)
                if p1 in required_partners[p0]:
                    triangle_corner_ids.add(i)

    xs_p = [p[0] for p in polygon]
    ys_p = [p[1] for p in polygon]
    minx, miny, maxx, maxy = min(xs_p), min(ys_p), max(xs_p), max(ys_p)
    ox, oy = int(round(minx)), int(round(miny))
    W, H = int(round(maxx - minx)), int(round(maxy - miny))

    grid = [[0] * W for _ in range(H)]
    is_bnd_cell = [[False] * W for _ in range(H)]
    total_bnd_cells = 0
    for y in range(H):
        cy = oy + y + 0.5
        for x in range(W):
            cx = ox + x + 0.5
            if not _point_in_poly_fast(cx, cy, polygon):
                grid[y][x] = 2
            else:
                on_b = (
                    x == 0 or x == W - 1 or y == 0 or y == H - 1
                    or not _point_in_poly_fast(cx - 1.0, cy, polygon)
                    or not _point_in_poly_fast(cx + 1.0, cy, polygon)
                    or not _point_in_poly_fast(cx, cy - 1.0, polygon)
                    or not _point_in_poly_fast(cx, cy + 1.0, polygon)
                )
                if on_b:
                    is_bnd_cell[y][x] = True
                    total_bnd_cells += 1

    bnd_psum = [[0] * (W + 1) for _ in range(H + 1)]
    for y in range(H):
         b_row = is_bnd_cell[y]
         p_row = bnd_psum[y + 1]
         prev_row = bnd_psum[y]
         r_sum = 0
         for x in range(W):
             if b_row[x]:
                 r_sum += 1
             p_row[x + 1] = prev_row[x + 1] + r_sum

    int_sizes = {rid: (int(round(w0)), int(round(h0))) for rid, (w0, h0) in sizes.items()}
    max_bnd_for_rid = {}
    for rid, (w0, h0) in sizes.items():
        if rid in interior:
            max_bnd_for_rid[rid] = 0
        else:
            iw, ih = int(round(w0)), int(round(h0))
            max_bnd_for_rid[rid] = iw * ih if min(iw, ih) == 1 else min(iw * ih, 2 * (iw + ih) - 4)

    # Group rooms by geometric equivalence class `(min(w,h), max(w,h), is_interior)`
    # so duplicate-size rooms (e.g. two 4x7 rooms #4 and #5) can be cleanly relabeled
    # to satisfy partial adjacency constraints without symmetry-breaking collisions!
    iso_classes = {}
    rid_to_iso = {}
    for rid, (w0, h0) in sizes.items():
        ikey = (int(round(min(w0, h0))), int(round(max(w0, h0))), rid in interior)
        iso_classes.setdefault(ikey, []).append(rid)
        rid_to_iso[rid] = ikey

    has_isomorphic_dups = any(len(v) > 1 for v in iso_classes.values())
    use_iso_class_search = enforce_adj and (not strict_exact_graph) and has_isomorphic_dups and len(adj_pairs) <= 8

    def try_relabel_solution(raw_placements):
        if not enforce_adj:
            return dict(raw_placements)
        touch_fn = _rects_touch_or_corner if allow_corner_adj else _rects_touch

        def check_map(cand_p):
            for a, b in adj_pairs:
                if not touch_fn(cand_p[a], cand_p[b]):
                    return False
            if strict_exact_graph:
                rids = list(cand_p.keys())
                for idx_a in range(len(rids)):
                    ra = rids[idx_a]
                    for idx_b in range(idx_a + 1, len(rids)):
                        rb = rids[idx_b]
                        if rb not in required_partners[ra] and _rects_touch(cand_p[ra], cand_p[rb]):
                            return False
            return True

        if check_map(raw_placements):
            return dict(raw_placements)
        if not has_isomorphic_dups:
            return None

        constrained_rids = [r for r, ps in required_partners.items() if ps]
        involved_classes = sorted({rid_to_iso[r] for r in constrained_rids})
        work = dict(raw_placements)

        def assign_class(c_idx):
            if c_idx == len(involved_classes):
                return dict(work) if check_map(work) else None
            ikey = involved_classes[c_idx]
            members = iso_classes[ikey]
            boxes = [raw_placements[m] for m in members]
            c_members = [m for m in members if required_partners[m]]
            u_members = [m for m in members if not required_partners[m]]
            used_b = [False] * len(boxes)

            def assign_member(m_idx):
                if m_idx == len(c_members):
                    rem_boxes = [boxes[bi] for bi in range(len(boxes)) if not used_b[bi]]
                    for um, bx in zip(u_members, rem_boxes):
                        work[um] = bx
                    return assign_class(c_idx + 1)
                m_id = c_members[m_idx]
                for bi, bx in enumerate(boxes):
                    if used_b[bi]:
                        continue
                    ok = True
                    for p_id in required_partners[m_id]:
                        p_ikey = rid_to_iso[p_id]
                        p_idx = involved_classes.index(p_ikey)
                        if p_idx < c_idx or (p_idx == c_idx and p_id in c_members[:m_idx]):
                            if not touch_fn(bx, work[p_id]):
                                ok = False
                                break
                    if not ok:
                        continue
                    used_b[bi] = True
                    work[m_id] = bx
                    res = assign_member(m_idx + 1)
                    if res is not None:
                        return res
                    used_b[bi] = False
                return None

            return assign_member(0)

        return assign_class(0)

    placements = {}
    used = set()
    nodes = [0]
    deadline = time.perf_counter() + time_limit if time_limit is not None else None
    seen_layouts = {_layout_key(s) for s in collect_solutions} if collect_solutions else set()

    def touches_exterior_wall(gx, gy, uw, uh):
        if gx == 0 or gy == 0 or gx + uw == W or gy + uh == H:
            return True
        for dx in range(uw):
            if grid[gy - 1][gx + dx] == 2 or grid[gy + uh][gx + dx] == 2:
                return True
        for dy in range(uh):
            if grid[gy + dy][gx - 1] == 2 or grid[gy + dy][gx + uw] == 2:
                return True
        return False

    def can_place(rid, gx, gy, uw, uh, is_int):
        if gx < 0 or gy < 0 or gx + uw > W or gy + uh > H:
            return False
        for dy in range(uh):
            row = grid[gy + dy]
            for dx in range(uw):
                if row[gx + dx] != 0:
                    return False
        tb = (bnd_psum[gy + uh][gx + uw] - bnd_psum[gy][gx + uw] - bnd_psum[gy + uh][gx] + bnd_psum[gy][gx]) > 0
        if is_int:
            if tb:
                return False
        elif require_boundary_touch and not tb:
            return False
        rx, ry = float(ox + gx), float(oy + gy)
        if rid in triangle_corner_ids and not _at_plot_corner(rx, ry, uw, uh, convex_vertices):
            return False
        if enforce_adj:
            cand = (rx, ry, float(uw), float(uh))
            partners = required_partners[rid]
            if use_iso_class_search:
                rid_class_members = iso_classes[rid_to_iso[rid]]
                unplaced_same_class = [m for m in rid_class_members if m not in used and m != rid]
                if not unplaced_same_class:
                    for j, placed_box in placements.items():
                        if j in partners:
                            j_members = iso_classes[rid_to_iso[j]]
                            if not any(m not in used for m in j_members):
                                ok_touch = _rects_touch_or_corner(cand, placed_box) if allow_corner_adj else _rects_touch(cand, placed_box)
                                if not ok_touch:
                                    return False
            else:
                for j, placed_box in placements.items():
                    if j in partners:
                        ok_touch = _rects_touch_or_corner(cand, placed_box) if allow_corner_adj else _rects_touch(cand, placed_box)
                        if not ok_touch:
                            return False
                    elif strict_exact_graph:
                        if _rects_touch(cand, placed_box):
                            return False
        return True

    def can_unplaced_fit_adjacent_to(pgx, pgy, puw, puh, q_id):
        qw0, qh0 = sizes[q_id]
        qw_i, qh_i = int(round(qw0)), int(round(qh0))
        q_int = q_id in interior
        for uw, uh in ((qw_i, qh_i), (qh_i, qw_i)) if qw_i != qh_i else ((qw_i, qh_i),):
            for gx in (pgx - uw, pgx + puw):
                if gx < 0 or gx + uw > W:
                    continue
                y_start = max(0, pgy - uh + (0 if allow_corner_adj else 1))
                y_end = min(H - uh, pgy + puh - (0 if allow_corner_adj else 1))
                for gy in range(y_start, y_end + 1):
                    if can_place(q_id, gx, gy, uw, uh, q_int):
                        return True
            for gy in (pgy - uh, pgy + puh):
                if gy < 0 or gy + uh > H:
                    continue
                x_start = max(0, pgx - uw + (0 if allow_corner_adj else 1))
                x_end = min(W - uw, pgx + puw - (0 if allow_corner_adj else 1))
                for gx in range(x_start, x_end + 1):
                    if can_place(q_id, gx, gy, uw, uh, q_int):
                        return True
        return False

    def mark(gx, gy, uw, uh, val):
        for dy in range(uh):
            row = grid[gy + dy]
            for dx in range(uw):
                row[gx + dx] = val
        return bnd_psum[gy + uh][gx + uw] - bnd_psum[gy][gx + uw] - bnd_psum[gy + uh][gx] + bnd_psum[gy][gx]

    rx_grid = [[0] * W for _ in range(H)]
    ry_grid = [[0] * W for _ in range(H)]
    int_rx_grid = [[0] * W for _ in range(H)]

    def compute_runs_and_validate(rem_types, min_side, min_bnd_side, narrow_area_cap):
        for y in range(H):
            row = grid[y]
            b_row = is_bnd_cell[y]
            ir_row = int_rx_grid[y]
            x = 0
            while x < W:
                if row[x] == 0 and not b_row[x]:
                    x0 = x
                    x += 1
                    while x < W and row[x] == 0 and not b_row[x]:
                        x += 1
                    irun = x - x0
                    for xx in range(x0, x):
                        ir_row[xx] = irun
                else:
                    ir_row[x] = 0
                    x += 1

        for y in range(H):
            row = grid[y]
            r_row = rx_grid[y]
            b_row = is_bnd_cell[y]
            x = 0
            while x < W:
                if row[x] == 0:
                    x0 = x
                    has_bnd = b_row[x]
                    x += 1
                    while x < W and row[x] == 0:
                        if b_row[x]:
                            has_bnd = True
                        x += 1
                    run = x - x0
                    if run < (min_bnd_side if has_bnd else min_side):
                        return False
                    for xx in range(x0, x):
                        r_row[xx] = run
                else:
                    r_row[x] = 0
                    x += 1

        narrow_cells = 0
        max_ry_for_rx = [0] * (W + 1)
        for x in range(W):
            y = 0
            while y < H:
                if grid[y][x] == 0:
                    y0 = y
                    has_bnd = is_bnd_cell[y][x]
                    y += 1
                    while y < H and grid[y][x] == 0:
                        if is_bnd_cell[y][x]:
                            has_bnd = True
                        y += 1
                    run = y - y0
                    if run < (min_bnd_side if has_bnd else min_side):
                        return False
                    for yy in range(y0, y):
                        ry_grid[yy][x] = run
                        rx_val = rx_grid[yy][x]
                        if run > max_ry_for_rx[rx_val]:
                            max_ry_for_rx[rx_val] = run
                        if rx_val == 1 or run == 1:
                            narrow_cells += 1
                            if narrow_cells > narrow_area_cap:
                                return False
                            cell_bnd = is_bnd_cell[yy][x]
                            ok = False
                            for (w_min, h_max, is_int, _rid) in rem_types:
                                if cell_bnd and is_int:
                                    continue
                                if (w_min <= rx_val and h_max <= run) or (h_max <= rx_val and w_min <= run):
                                    ok = True
                                    break
                            if not ok:
                                return False
                else:
                    ry_grid[y][x] = 0
                    y += 1

        for w_idx in range(W - 1, 0, -1):
            if max_ry_for_rx[w_idx + 1] > max_ry_for_rx[w_idx]:
                max_ry_for_rx[w_idx] = max_ry_for_rx[w_idx + 1]

        has_int_rem = any(is_int for (_wm, _hm, is_int, _r) in rem_types)
        if has_int_rem:
            max_iry_for_irx = [0] * (W + 1)
            for x in range(W):
                y = 0
                while y < H:
                    if grid[y][x] == 0 and not is_bnd_cell[y][x]:
                        y0 = y
                        max_irx = int_rx_grid[y][x]
                        y += 1
                        while y < H and grid[y][x] == 0 and not is_bnd_cell[y][x]:
                            if int_rx_grid[y][x] > max_irx:
                                max_irx = int_rx_grid[y][x]
                            y += 1
                        iry = y - y0
                        if iry > max_iry_for_irx[max_irx]:
                            max_iry_for_irx[max_irx] = iry
                    else:
                        y += 1
            for w_idx in range(W - 1, 0, -1):
                if max_iry_for_irx[w_idx + 1] > max_iry_for_irx[w_idx]:
                    max_iry_for_irx[w_idx] = max_iry_for_irx[w_idx + 1]

        for (w_min, h_max, is_int, _rid) in rem_types:
            if is_int:
                ok_fit = (w_min <= W and max_iry_for_irx[w_min] >= h_max) or (h_max <= W and max_iry_for_irx[h_max] >= w_min)
            else:
                ok_fit = (w_min <= W and max_ry_for_rx[w_min] >= h_max) or (h_max <= W and max_ry_for_rx[h_max] >= w_min)
            if not ok_fit:
                return False



        if enforce_adj:
            for p_id, (px, py, pw, ph) in placements.items():
                if use_iso_class_search and any(m not in used for m in iso_classes[rid_to_iso[p_id]]):
                    continue
                pgx, pgy = int(round(px - ox)), int(round(py - oy))
                puw, puh = int(round(pw)), int(round(ph))
                for q_id in required_partners[p_id]:
                    if q_id not in used:
                        if use_iso_class_search and any(m in used for m in iso_classes[rid_to_iso[q_id]]):
                            continue
                        if not can_unplaced_fit_adjacent_to(pgx, pgy, puw, puh, q_id):
                            return False

            if not use_iso_class_search:
                for q_id, (qw, qh) in sizes.items():
                    if q_id in used:
                        continue
                    placed_ps = [placements[p] for p in required_partners[q_id] if p in placements]
                    if len(placed_ps) >= 2:
                        ok_span = False
                        for uw, uh in ((qw, qh), (qh, qw)) if qw != qh else ((qw, qh),):
                            rx_min = max(px - uw for px, py, pw, ph in placed_ps)
                            rx_max = min(px + pw for px, py, pw, ph in placed_ps)
                            ry_min = max(py - uh for px, py, pw, ph in placed_ps)
                            ry_max = min(py + ph for px, py, pw, ph in placed_ps)
                            if rx_min <= rx_max + 1e-6 and ry_min <= ry_max + 1e-6:
                                ok_span = True
                                break
                        if not ok_span:
                            return False
        return True

    def find_convex_corners():
        corners = []
        for y in range(H):
            row = grid[y]
            for x in range(W):
                if row[x] != 0:
                    continue
                left_wall = (x == 0 or row[x - 1] != 0)
                right_wall = (x == W - 1 or row[x + 1] != 0)
                bot_wall = (y == 0 or grid[y - 1][x] != 0)
                top_wall = (y == H - 1 or grid[y + 1][x] != 0)
                if left_wall and bot_wall:
                    corners.append((x, y, +1, +1))
                if right_wall and bot_wall:
                    corners.append((x, y, -1, +1))
                if left_wall and top_wall:
                    corners.append((x, y, +1, -1))
                if right_wall and top_wall:
                    corners.append((x, y, -1, -1))
        return corners

    total_rem_bnd_area = sum(w * h for rid, (w, h) in sizes.items() if rid not in interior)
    total_rem_max_bnd_cap = sum(max_bnd_for_rid.values())

    def backtrack(uncov_bnd_cells, rem_bnd_area, rem_max_bnd_cap):
        nodes[0] += 1
        if max_nodes is not None and nodes[0] > max_nodes:
            raise _SearchTimeout()
        if deadline and time.perf_counter() > deadline:
            raise _SearchTimeout()
        if len(used) == len(sizes):
            relabeled = try_relabel_solution(placements)
            if relabeled is None:
                return False
            if collect_solutions is not None:
                k = _layout_key(relabeled)
                if k not in seen_layouts:
                    seen_layouts.add(k)
                    collect_solutions.append(relabeled)
                    if max_solutions and len(collect_solutions) >= max_solutions:
                        return True
                return False
            placements.clear()
            placements.update(relabeled)
            return True

        if require_boundary_touch and (rem_bnd_area < uncov_bnd_cells or rem_max_bnd_cap < uncov_bnd_cells):
            return False

        rem_types = []
        seen_type_keys = set()
        min_side = 10**9
        min_bnd_side = 10**9
        narrow_area_cap = 0

        for rid, (w0, h0) in int_sizes.items():
            if rid in used:
                continue
            s_min, s_max = (w0, h0) if w0 <= h0 else (h0, w0)
            if s_min < min_side:
                min_side = s_min
            if s_min == 1:
                narrow_area_cap += w0 * h0
            is_int = rid in interior
            if not is_int and s_min < min_bnd_side:
                min_bnd_side = s_min
            t_key = (
                (s_min, s_max, is_int)
                if (not enforce_adj or use_iso_class_search)
                else (s_min, s_max, is_int, frozenset(required_partners[rid]))
            )
            if t_key not in seen_type_keys:
                seen_type_keys.add(t_key)
                rem_types.append((s_min, s_max, is_int, rid))

        if not compute_runs_and_validate(rem_types, min_side, min_bnd_side, narrow_area_cap):
            return False

        corners = find_convex_corners()
        if not corners:
            return False

        active_front_rids = set()
        if enforce_adj and not strict_exact_graph and not use_iso_class_search:
            for p_id in placements:
                for q_id in required_partners[p_id]:
                    if q_id not in used:
                        active_front_rids.add(q_id)

        best_corner_opts = None
        best_key = (10**9, 10**9, 10**9, 10**9, 10**9, 10**9)

        for (cx, cy, dir_x, dir_y) in corners:
            rx_len = 0
            wall_x_len = 0
            xx = cx
            wy = cy - dir_y
            while 0 <= xx < W and grid[cy][xx] == 0:
                rx_len += 1
                if use_wall_len and (wy < 0 or wy >= H or grid[wy][xx] != 0) and wall_x_len == rx_len - 1:
                    wall_x_len += 1
                xx += dir_x
            ry_len = 0
            wall_y_len = 0
            yy = cy
            wx = cx - dir_x
            while 0 <= yy < H and grid[yy][cx] == 0:
                ry_len += 1
                if use_wall_len and (wx < 0 or wx >= W or grid[yy][wx] != 0) and wall_y_len == ry_len - 1:
                    wall_y_len += 1
                yy += dir_y

            opts = []
            has_pending_partner = False
            for (w_min, h_max, is_int, rid) in rem_types:
                is_front = rid in active_front_rids
                is_seeded_room = (not placements) and bool(required_partners[rid]) and (not use_iso_class_search)
                for uw, uh in ((w_min, h_max), (h_max, w_min)) if w_min != h_max else ((w_min, h_max),):
                    uw_i, uh_i = uw, uh
                    if uw_i > rx_len or uh_i > ry_len:
                        continue
                    rem_x = rx_len - uw_i
                    if 0 < rem_x < min_side:
                        continue
                    if require_boundary_touch and 0 < rem_x < min_bnd_side and is_bnd_cell[cy][cx + dir_x * uw_i]:
                        continue
                    rem_y = ry_len - uh_i
                    if 0 < rem_y < min_side:
                        continue
                    if require_boundary_touch and 0 < rem_y < min_bnd_side and is_bnd_cell[cy + dir_y * uh_i][cx]:
                        continue
                    gx = cx if dir_x == 1 else (cx + 1 - uw_i)
                    gy = cy if dir_y == 1 else (cy + 1 - uh_i)
                    if can_place(rid, gx, gy, uw_i, uh_i, is_int):
                        if is_front:
                            has_pending_partner = True
                        flush_score = (1 if (uw_i == rx_len or (use_wall_len and uw_i == wall_x_len)) else 0) + (1 if (uh_i == ry_len or (use_wall_len and uh_i == wall_y_len)) else 0)
                        seed_boost = 1 if (is_front or is_seeded_room) else 0
                        opts.append((-seed_boost, -flush_score, 1 if is_int else 0, -(uw_i * uh_i), rid, gx, gy, uw_i, uh_i, is_int))
            if not opts:
                return False
            flush_cnt = sum(1 for o in opts if o[1] < 0)
            eff_len = len(opts) if pure_mrv else (flush_cnt if (0 < flush_cnt <= 2 and min(rx_len, ry_len) <= 10) else len(opts))
            q_tie = 0
            if not placements and seed_quadrant is not None:
                q_tie = -(cx * seed_quadrant[0] + cy * seed_quadrant[1])
            ckey = (0 if has_pending_partner else 1, q_tie, eff_len, len(opts), min(wall_x_len, wall_y_len) if use_wall_len else min(rx_len, ry_len), (wall_x_len + wall_y_len) if use_wall_len else (rx_len + ry_len))
            if ckey < best_key:
                best_key = ckey
                best_corner_opts = opts

        best_corner_opts.sort()
        for _sb, _fl, _int_flag, _neg_area, rid, gx, gy, uw_i, uh_i, is_int in best_corner_opts:
            bnd_cov = mark(gx, gy, uw_i, uh_i, 1)
            used.add(rid)
            placements[rid] = (float(ox + gx), float(oy + gy), float(uw_i), float(uh_i))
            res = backtrack(
                uncov_bnd_cells - bnd_cov,
                rem_bnd_area - (0 if is_int else uw_i * uh_i),
                rem_max_bnd_cap - max_bnd_for_rid[rid],
            )
            if res is True or res is None:
                return res
            del placements[rid]
            used.remove(rid)
            mark(gx, gy, uw_i, uh_i, 0)
        return False

    try:
        res = backtrack(total_bnd_cells, total_rem_bnd_area, total_rem_max_bnd_cap)
    except _SearchTimeout:
        res = None

    if collect_solutions is not None:
        return res, collect_solutions, nodes[0]
    return (True, dict(placements), nodes[0]) if res is True else ((False, None, nodes[0]) if res is False else (None, None, nodes[0]))


def _pack_mrv_corner_ptpg(
    polygon,
    rects,
    required_adjacency=None,
    *,
    require_boundary_touch=True,
    interior_ids=(),
    time_limit=60.0,
    collect_solutions=None,
    max_solutions=None,
    allow_corner_adj: bool = False,
):
    adj_pairs = _normalize_adjacency_pairs(required_adjacency)
    enforce_adj = bool(adj_pairs)
    total_nodes = 0
    t0 = time.perf_counter()

    def rem_time():
        return None if time_limit is None else max(0.05, time_limit - (time.perf_counter() - t0))

    # Build a portfolio of search passes:
    # - If a full PTPG graph is provided (`len(adj_pairs) >= len(rects)`), try `strict_exact_graph=True`
    #   first (solves full PTPGs in ~15-150 nodes), then automatically fall back to mandatory-contact mode
    #   (`strict_exact_graph=False`) so extra incidental touches never cause false infeasibility.
    # - Try both arm-cap flush ordering (`pure_mrv=False`) and pure MRV ordering (`pure_mrv=True`) with
    #   directional seed tie-breakers so asymmetric single-pair constraints in any wing are found rapidly.
    passes = []
    if enforce_adj and len(adj_pairs) >= len(rects):
        passes.append((True, False, False, 15000, None, False))
        passes.append((True, True, False, 15000, None, False))

    passes.append((False, False, allow_corner_adj, 3000, None, False))
    for qd in ((1, -1), (-1, -1), (-1, 1), (1, 1)):
        passes.append((False, False, allow_corner_adj, 1800, qd, False))
    passes.append((False, True, allow_corner_adj, 2500, None, False))
    passes.append((False, False, allow_corner_adj, 150000, None, False))

    if enforce_adj and not allow_corner_adj:
        if len(adj_pairs) >= len(rects):
            passes.append((True, False, True, 15000, None, False))
        passes.append((False, False, True, 25000, None, False))
        passes.append((False, True, True, 25000, None, False))

    for strict_g, pmrv, corner_adj, n_Wait, q_dir, w_len in passes:
        rt = rem_time()
        if rt is not None and rt <= 0.05:
            break
        initial_cnt = len(collect_solutions) if collect_solutions is not None else 0
        res, out, nodes = _pack_mrv_corner_ptpg_single(
            polygon,
            rects,
            adj_pairs,
            require_boundary_touch=require_boundary_touch,
            interior_ids=interior_ids,
            time_limit=rt,
            collect_solutions=collect_solutions,
            max_solutions=max_solutions,
            allow_corner_adj=corner_adj,
            strict_exact_graph=strict_g,
            pure_mrv=pmrv,
            max_nodes=n_Wait,
            seed_quadrant=q_dir,
            use_wall_len=w_len,
        )
        total_nodes += nodes
        if collect_solutions is not None:
            if len(collect_solutions) > initial_cnt and (max_solutions is None or len(collect_solutions) >= max_solutions):
                return res, collect_solutions, total_nodes
            if res is False and n_Wait >= 100000 and not strict_g and len(collect_solutions) > 0:
                return False, collect_solutions, total_nodes
        else:
            if res is True:
                return True, out, total_nodes
            if res is False and not enforce_adj and pmrv:
                return False, None, total_nodes

    if collect_solutions is not None:
        return (True if collect_solutions else False), collect_solutions, total_nodes
    return False, None, total_nodes


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
):
    sizes = dict(rects)
    if not check_area_match(polygon, sizes)[0]:
        return False, collect_solutions, 0

    poly = Polygon(polygon)
    boundary = poly.boundary
    horiz_segs, vert_segs = _boundary_segments(polygon)
    reflex_vertices = _polygon_reflex_vertices(polygon)
    convex_vertices = _polygon_convex_vertices(polygon)
    interior = set(interior_ids)
    unknown = interior - set(sizes)
    if unknown:
        raise ValueError(f"interior_ids references unknown rect ids: {sorted(unknown)}")
    adj_pairs = _normalize_adjacency_pairs(required_adjacency)
    is_int_grid = (
        all(abs(x - round(x)) < 1e-6 and abs(y - round(y)) < 1e-6 for x, y in polygon)
        and all(abs(w - round(w)) < 1e-6 and abs(h - round(h)) < 1e-6 for w, h in sizes.values())
    )
    if is_int_grid:
        return _pack_mrv_corner_ptpg(
            polygon, sizes, adj_pairs,
            require_boundary_touch=require_boundary_touch,
            interior_ids=interior,
            time_limit=time_limit,
            collect_solutions=collect_solutions,
            max_solutions=max_solutions,
        )

    seen_layouts = {_layout_key(sol) for sol in collect_solutions} if collect_solutions else set()

    required_partners: Dict[int, List[int]] = {i: [] for i in sizes}
    for a, b in adj_pairs:
        if a == b:
            raise ValueError(f"required_adjacency has a self-pair: ({a!r}, {a!r})")
        if a not in sizes or b not in sizes:
            raise ValueError(f"required_adjacency references an unknown rect id ({a!r}, {b!r})")
        if b not in required_partners[a]:
            required_partners[a].append(b)
        if a not in required_partners[b]:
            required_partners[b].append(a)

    strict_exact = len(adj_pairs) >= len(sizes)
    triangle_corner_ids = (
        {
            i for i, ps in required_partners.items()
            if i not in interior and len(ps) == 2 and ps[1] in required_partners.get(ps[0], [])
        }
        if strict_exact
        else set()
    )

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
            if j in partners and not _rects_touch(cand, placed):
                return False
            if strict_exact and j not in partners and _rects_touch(cand, placed):
                return False
        return True

    def boundary_ok(idx, rx, ry, w, h) -> bool:
        touches = _shares_boundary_edge_fast(rx, ry, w, h, horiz_segs, vert_segs)
        if idx in interior:
            return not touches
        return touches or not require_boundary_touch

    reflex_xs = {vx for vx, _ in reflex_vertices}
    reflex_ys = {vy for _, vy in reflex_vertices}

    def _overhang_only_coords(idx, px, py, uw, uh, free_shape, seen_coords):
        """Generate non-corner straddling/overhang (rx, ry) coordinates for `idx` of size `(uw, uh)`
        at `point = (px, py)`:
        - Covers Double-Overhang (bridging across both ends of an already-placed partner `bx..bx+bw`), AND
        - Covers Single-Reflex / Staggered T-Junction Overhang (`rx < px < rx + uw` when `px` lies on
          a plot reflex line `reflex_xs` or `py` lies on `reflex_ys`)."""
        placed_partners = [
            placements[p_id] for p_id in required_partners.get(idx, [])
            if p_id in placements
        ]
        if not placed_partners:
            return []

        out = []
        fminx, fminy, fmaxx, fmaxy = free_shape.bounds
        live_verts = None
        rem_neighbor_dims = None
        on_reflex_x = any(abs(px - rx_v) < 1e-9 for rx_v in reflex_xs)
        on_reflex_y = any(abs(py - ry_v) < 1e-9 for ry_v in reflex_ys)

        for bx, by, bw, bh in placed_partners:
            # Horizontal contact along y = py (py == by or py == by + bh)
            if (abs(py - by) < 1e-9 or abs(py - (by + bh)) < 1e-9) and (abs(px - bx) < 1e-9 or abs(px - (bx + bw)) < 1e-9):
                can_double_overhang = (uw > bw + 1e-9 and fminx < bx - 1e-9 and fmaxx > bx + bw + 1e-9)
                can_single_straddle = (on_reflex_x and fminx < px - 1e-9 and fmaxx > px + 1e-9)
                if can_double_overhang or can_single_straddle:
                    if live_verts is None:
                        live_verts = _polygon_vertices(free_shape)
                        rem_neighbor_dims = {d for n_id in required_partners.get(idx, []) if n_id not in placements for d in sizes[n_id]}
                    x_cands = set()
                    for vx, _vy in live_verts:
                        x_cands.add(vx)
                        x_cands.add(vx - uw)
                        for d in rem_neighbor_dims:
                            x_cands.add(vx + d)
                            x_cands.add(vx - d - uw)
                    for rx in sorted(x_cands):
                        if not (fminx - 1e-9 <= rx and rx + uw <= fmaxx + 1e-9):
                            continue
                        is_double = (rx + 1e-9 < bx and bx + bw < rx + uw - 1e-9)
                        is_single = (can_single_straddle and rx + 1e-9 < px < rx + uw - 1e-9)
                        if is_double or is_single:
                            for ry in (py, py - uh):
                                key = (round(rx, 6), round(ry, 6))
                                if key not in seen_coords:
                                    seen_coords.add(key)
                                    out.append((rx, ry))

            # Vertical contact along x = px (px == bx or px == bx + bw)
            if (abs(px - bx) < 1e-9 or abs(px - (bx + bw)) < 1e-9) and (abs(py - by) < 1e-9 or abs(py - (by + bh)) < 1e-9):
                can_double_overhang = (uh > bh + 1e-9 and fminy < by - 1e-9 and fmaxy > by + bh + 1e-9)
                can_single_straddle = (on_reflex_y and fminy < py - 1e-9 and fmaxy > py + 1e-9)
                if can_double_overhang or can_single_straddle:
                    if live_verts is None:
                        live_verts = _polygon_vertices(free_shape)
                        rem_neighbor_dims = {d for n_id in required_partners.get(idx, []) if n_id not in placements for d in sizes[n_id]}
                    y_cands = set()
                    for _vx, vy in live_verts:
                        y_cands.add(vy)
                        y_cands.add(vy - uh)
                        for d in rem_neighbor_dims:
                            y_cands.add(vy + d)
                            y_cands.add(vy - d - uh)
                    for ry in sorted(y_cands):
                        if not (fminy - 1e-9 <= ry and ry + uh <= fmaxy + 1e-9):
                            continue
                        is_double = (ry + 1e-9 < by and by + bh < ry + uh - 1e-9)
                        is_single = (can_single_straddle and ry + 1e-9 < py < ry + uh - 1e-9)
                        if is_double or is_single:
                            for rx in (px, px - uw):
                                key = (round(rx, 6), round(ry, 6))
                                if key not in seen_coords:
                                    seen_coords.add(key)
                                    out.append((rx, ry))

        return out

    def placements_of(idx, point, free_shape, compute_diff: bool = True, allow_overhang: bool = False):
        if free_shape.is_empty:
            return []
        px, py = point
        w0, h0 = sizes[idx]
        fminx, fminy, fmaxx, fmaxy = free_shape.bounds
        orientations = list({(w0, h0), (h0, w0)})
        seen_coords = set()
        trial_list = []

        # Pass 1: Standard 4-quadrant corner placements across BOTH orientations first
        for uw, uh in orientations:
            for ox, oy in _candidate_offsets(uw, uh):
                rx, ry = px + ox, py + oy
                key = (round(rx, 6), round(ry, 6), uw, uh)
                if key not in seen_coords:
                    seen_coords.add(key)
                    trial_list.append((rx, ry, uw, uh))

        # Pass 2: T-junction double-overhang and single-reflex straddle placements
        if allow_overhang:
            for uw, uh in orientations:
                ov_seen = {(k[0], k[1]) for k in seen_coords if k[2] == uw and k[3] == uh}
                for rx, ry in _overhang_only_coords(idx, px, py, uw, uh, free_shape, ov_seen):
                    trial_list.append((rx, ry, uw, uh))

        out = []
        for rx, ry, uw, uh in trial_list:
            if rx < fminx - 1e-9 or ry < fminy - 1e-9 or rx + uw > fmaxx + 1e-9 or ry + uh > fmaxy + 1e-9:
                continue
            if idx in triangle_corner_ids and not _at_plot_corner(rx, ry, uw, uh, convex_vertices):
                continue
            if not boundary_ok(idx, rx, ry, uw, uh):
                continue
            if not required_adjacency_ok(idx, rx, ry, uw, uh):
                continue
            candidate = box(rx, ry, rx + uw, ry + uh)
            if abs(free_shape.intersection(candidate).area - (uw * uh)) >= 1e-6:
                continue
            new_free = free_shape.difference(candidate) if compute_diff else None
            out.append((rx, ry, uw, uh, new_free))
        return out

    def _interior_rect_still_viable(q_idx, free_shape) -> bool:
        w0, h0 = sizes[q_idx]
        fminx, fminy, fmaxx, fmaxy = free_shape.bounds
        fw, fh = fmaxx - fminx, fmaxy - fminy
        placed_partners = [placements[p] for p in required_partners.get(q_idx, []) if p in placements]
        for uw, uh in {(w0, h0), (h0, w0)}:
            if uw > fw + 1e-6 or uh > fh + 1e-6:
                continue
            if not placed_partners:
                return True
            rx_min = max(px - uw for px, py, pw, ph in placed_partners)
            rx_max = min(px + pw for px, py, pw, ph in placed_partners)
            ry_min = max(py - uh for px, py, pw, ph in placed_partners)
            ry_max = min(py + ph for px, py, pw, ph in placed_partners)
            if rx_min <= rx_max + 1e-6 and ry_min <= ry_max + 1e-6:
                return True
        return False

    def _rect_still_placeable(q_idx, free_shape) -> bool:
        if q_idx in interior:
            return _interior_rect_still_viable(q_idx, free_shape)
        verts = _polygon_vertices(free_shape)
        for v in verts:
            _check_time()
            if placements_of(q_idx, v, free_shape, compute_diff=False, allow_overhang=False):
                return True
        for v in verts:
            _check_time()
            if placements_of(q_idx, v, free_shape, compute_diff=False, allow_overhang=True):
                return True
        return False

    def _slot_count(opts, point):
        px, py = point
        slots = set()
        for (rx, ry, w, h, _new_free) in opts:
            qx = 0 if abs(rx - px) < 1e-9 else 1
            qy = 0 if abs(ry - py) < 1e-9 else 1
            slots.add((qx, qy))
        return len(slots)

    def local_score(idx, point, free_shape):
        return _slot_count(placements_of(idx, point, free_shape, compute_diff=False), point)

    def point_score(point, originator, free_shape):
        candidates = [i for i in required_partners.get(originator, []) if i in remaining_ids()]
        if not candidates:
            return float("inf")
        return min(local_score(i, point, free_shape) for i in candidates)

    def order_points(points_with_originator, free_shape):
        return sorted(points_with_originator, key=lambda po: point_score(po[0], po[1], free_shape))

    def seed_step(free_shape):
        verts = _polygon_vertices(free_shape)
        anchor_verts = [p for p in verts if boundary.distance(Point(p)) < 1e-9]
        remaining = remaining_ids()
        corner_ids = [i for i in remaining if i in triangle_corner_ids]
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
                best = min(_slot_count(table[i][p], p) for i in corner_ids if p in table[i])
                return (best, _slot_count(table[idx][p], p), p)

            return [(0, corner_key(p)[0], p, [(idx, table[idx][p])]) for p in sorted(table[idx], key=corner_key)]

        per_point = []
        for p in anchor_verts:
            _check_time()
            scored = []
            for idx in remaining:
                if idx in interior:
                    continue
                opts = placements_of(idx, p, free_shape)
                if not opts:
                    continue
                deg = len(required_partners.get(idx, []))
                w, h = sizes[idx]
                scored.append((deg, _slot_count(opts, p), -(w * h), idx, opts))
            if not scored:
                continue
            for level in sorted({d for d, _, _, _, _ in scored}):
                level_score = min(sc for d, sc, _, _, _ in scored if d == level)
                tied = sorted(
                    ((idx, opts) for d, sc, _, idx, opts in scored if (d, sc) == (level, level_score)),
                    key=lambda t: -(sizes[t[0]][0] * sizes[t[0]][1]),
                )
                is_reflex_notch = 0 if (level == 1 and p in reflex_vertices) else 1
                per_point.append((level, is_reflex_notch, level_score, p, tied))
        if not per_point:
            return None
        per_point.sort(key=lambda t: t[:3])
        return [(deg, score, p, tied) for (deg, _il, score, p, tied) in per_point]

    def remaining_ids():
        return [i for i in sizes if i not in placements]

    def _on_region(point, region):
        return region.distance(Point(point)) < 1e-9

    def _region_fillable(region, eligible_ids):
        if not eligible_ids:
            return False
        area = region.area
        areas = [sizes[i][0] * sizes[i][1] for i in eligible_ids]
        if area > sum(areas) + 1e-6:
            return False
        minx, miny, maxx, maxy = region.bounds
        min_side = min(min(sizes[i]) for i in eligible_ids)
        if (maxx - minx) < min_side - 1e-6 or (maxy - miny) < min_side - 1e-6:
            return False
        if all(abs(a - round(a)) < 1e-9 for a in areas) and abs(area - round(area)) < 1e-6:
            bits = 1
            for a in areas:
                bits |= bits << int(round(a))
            return bool((bits >> int(round(area))) & 1)
        return True

    def _box_touches_region_bounds(placed_box, rbounds, tol=1e-6):
        bx, by, bw, bh = placed_box
        rminx, rminy, rmaxx, rmaxy = rbounds
        x_ov = min(bx + bw, rmaxx) - max(bx, rminx)
        y_ov = min(by + bh, rmaxy) - max(by, rminy)
        return (x_ov > tol and y_ov >= -tol) or (y_ov > tol and x_ov >= -tol)

    def _all_regions_viable(comps, remaining):
        for comp in comps:
            rbounds = comp.bounds
            eligible = [
                i for i in remaining
                if all(
                    _box_touches_region_bounds(placements[p], rbounds)
                    for p in required_partners.get(i, [])
                    if p in placements
                )
            ]
            if not _region_fillable(comp, eligible):
                return False
        return True

    def region_seed(free_shape, region):
        remaining = remaining_ids()
        convex_verts, _ = _region_convex_and_reflex_vertices(region)
        target_verts = sorted(convex_verts) if convex_verts else _polygon_vertices(region)
        best = None
        for v in target_verts:
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
        if best is None:
            return None
        _, v, rows = best
        rows.sort(key=lambda r: r[:4])
        return [(r[0], r[1], v, [(r[3], r[4])]) for r in rows]

    def solve(free_shape, frontier):
        nodes[0] += 1
        _check_time()
        remaining = remaining_ids()
        if not remaining:
            if collect_solutions is not None:
                key = _layout_key(placements)
                if key not in seen_layouts:
                    seen_layouts.add(key)
                    collect_solutions.append(dict(placements))
                    if max_solutions is not None and len(collect_solutions) >= max_solutions:
                        return True
                return False
            return True

        active = None
        if free_shape.geom_type != "Polygon":
            comps = _free_space_components(free_shape)
            if len(comps) > 1:
                if not _all_regions_viable(comps, remaining):
                    return False
                active = min(comps, key=lambda c: c.area)
                on = [f for f in frontier if _on_region(f[0], active)]
                off = [f for f in frontier if not _on_region(f[0], active)]
                frontier = on + off

        if frontier and (active is None or _on_region(frontier[0][0], active)):
            (point, originator), *rest = frontier
            candidates = [i for i in required_partners.get(originator, []) if i in remaining]
            if not candidates:
                return solve(free_shape, rest)

            cand_opts = {}
            for i in candidates:
                opts = placements_of(i, point, free_shape, compute_diff=True, allow_overhang=True)
                cand_opts[i] = opts
            candidates.sort(key=lambda i: (_slot_count(cand_opts[i], point), -(sizes[i][0] * sizes[i][1])))

            for cand in candidates:
                for (rx, ry, w, h, new_free) in cand_opts[cand]:
                    _check_time()
                    placements[cand] = (rx, ry, w, h)
                    new_points = order_points([(p, cand) for p in _new_frontier_points(rx, ry, w, h, point, new_free)], new_free)

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

                    def resolve_chain(pending_pairs, extra_points, cur_free):
                        if not pending_pairs:
                            result = solve(cur_free, extra_points + rest)
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
                        for reflex in _reflex_corners(placements[a], placements[b]):
                            points_to_try.append(reflex)
                        for shared in _shared_corners(placements[a], placements[b]):
                            line = _perpendicular_search_line(placements[a], placements[b], shared)
                            if line is not None:
                                points_to_try.extend(_anchor_points_on_line(cur_free, *line))

                        unique_pts = []
                        for pt in points_to_try:
                            if not any(abs(pt[0] - u[0]) < 1e-9 and abs(pt[1] - u[1]) < 1e-9 for u in unique_pts):
                                unique_pts.append(pt)

                        any_option = False
                        for t_idx in common:
                            for pt in unique_pts:
                                for (tx, ty, tw, th, t_free) in placements_of(t_idx, pt, cur_free, compute_diff=True, allow_overhang=True):
                                    _check_time()
                                    any_option = True
                                    placements[t_idx] = (tx, ty, tw, th)
                                    t_points = order_points([(p, t_idx) for p in _new_frontier_points(tx, ty, tw, th, pt, t_free)], t_free)
                                    next_pairs = ([(a, b)] if len(common) > 1 else []) + rest_pairs
                                    result = resolve_chain(next_pairs, extra_points + t_points, t_free)
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

            still_pending_for_originator = any(o == originator for (_, o) in rest)
            if not still_pending_for_originator:
                return False
            return solve(free_shape, rest)

        per_point = region_seed(free_shape, active) if active is not None else seed_step(free_shape)
        if not per_point:
            return False
        for _deg, _score, point, tied in per_point:
            for seed, opts in tied:
                if seed not in remaining_ids():
                    continue
                for (rx, ry, uw, uh, new_free) in opts:
                    placements[seed] = (rx, ry, uw, uh)
                    new_points = order_points([(p, seed) for p in _new_frontier_points(rx, ry, uw, uh, point, new_free)], new_free)
                    result = solve(new_free, new_points + (frontier if active is not None else []))
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

def render_solution_grid(polygon, solutions, save_path, cols=5, interior_ids=(), title_prefix="solution"):
    """Saves a grid with one panel per solution (a dict of placements).
    Boundary rooms are rendered in blue (`tab:blue`), interior rooms in orange
    (`tab:orange`), and concave reflex plot vertices are marked with red dots."""
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(solutions)
    if n == 0:
        raise ValueError("no solutions to render")
    interior_set = set(interior_ids or ())
    reflex_pts = _polygon_reflex_vertices(polygon)

    cols = min(cols, n)
    rows = math.ceil(n / cols)
    xs_all = [p[0] for p in polygon]
    ys_all = [p[1] for p in polygon]
    xlim = (min(xs_all) - 1.5, max(xs_all) + 1.5)
    ylim = (min(ys_all) - 1.5, max(ys_all) + 1.5)

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows))
    axes = [axes] if n == 1 else list(axes.flatten())

    bx = [p[0] for p in polygon] + [polygon[0][0]]
    by = [p[1] for p in polygon] + [polygon[0][1]]

    for i, placements in enumerate(solutions):
        ax = axes[i]
        ax.plot(bx, by, "k-", linewidth=1.8, zorder=4)
        if reflex_pts:
            rx_pts = [p[0] for p in reflex_pts]
            ry_pts = [p[1] for p in reflex_pts]
            ax.scatter(rx_pts, ry_pts, color="crimson", s=26, zorder=5, label="reflex vertex")
        for rid, (x, y, w, h) in placements.items():
            is_int = rid in interior_set
            face = "tab:orange" if is_int else "tab:blue"
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=0.9, edgecolor="black",
                facecolor=face, alpha=0.72, zorder=2
            ))
            tag = f"#{rid}\n{w:g}x{h:g}" + ("\n(int)" if is_int else "")
            ax.text(x + w / 2, y + h / 2, tag, ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title_prefix} {i + 1}", fontsize=9.5, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_examples_showcase(example_keys, save_path="packing_t_shape_examples_showcase.png", cols=4, time_limit=20.0):
    """Solves and renders a multi-example showcase grid (1 panel per built-in example)."""
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    solved_panels = []
    for k in example_keys:
        ex = EXAMPLES[k]
        poly = ex["polygon"]
        rects = ex["rects"]
        adj = ex.get("required_adjacency")
        inter = set(ex.get("interior", ()))
        t0 = time.perf_counter()
        found, sol, nodes = pack_rectangles_frontier(
            poly, rects, adj, require_boundary_touch=True, interior_ids=inter, time_limit=time_limit
        )
        dt = time.perf_counter() - t0
        if found and sol:
            solved_panels.append((k, ex.get("name", f"Example {k}"), poly, sol, inter, nodes, dt))

    n = len(solved_panels)
    if n == 0:
        raise ValueError("No examples solved for showcase")
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows))
    axes = [axes] if n == 1 else list(axes.flatten())

    for idx_panel, (k, name, poly, sol, inter, nodes, dt) in enumerate(solved_panels):
        ax = axes[idx_panel]
        bx = [p[0] for p in poly] + [poly[0][0]]
        by = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(bx, by, "k-", linewidth=1.8, zorder=4)
        reflex_pts = _polygon_reflex_vertices(poly)
        if reflex_pts:
            ax.scatter([p[0] for p in reflex_pts], [p[1] for p in reflex_pts], color="crimson", s=28, zorder=5)
        for rid, (x, y, w, h) in sol.items():
            is_int = rid in inter
            face = "tab:orange" if is_int else "tab:blue"
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=0.85, edgecolor="black",
                facecolor=face, alpha=0.72, zorder=2
            ))
            lbl = f"#{rid}\n{w:g}x{h:g}"
            ax.text(x + w / 2, y + h / 2, lbl, ha="center", va="center", fontsize=6, fontweight="bold", zorder=3)
        xs_all = [p[0] for p in poly]
        ys_all = [p[1] for p in poly]
        ax.set_xlim(min(xs_all) - 1.5, max(xs_all) + 1.5)
        ax.set_ylim(min(ys_all) - 1.5, max(ys_all) + 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        short_title = name.split("(")[0].strip()
        ax.set_title(f"Ex {k}: {short_title}\n({len(sol)} rooms, {len(inter)} int | {nodes} nodes, {dt:.2f}s)", fontsize=8.5, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("T-Shape & Multi-Reflex Rectilinear Floorplan Packer — Built-In Examples Showcase\n(Blue = Boundary Rooms, Orange = Interior Rooms, Red Dots = Concave Reflex Vertices)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Built-in Example Plots (1..12: T-Shape & Multi-Reflex; 13..22: Original L-Shape)
# --------------------------------------------------------------------------

EXAMPLES = {
    1: {'polygon': [(10, 0), (20, 0), (20, 15), (30, 15), (30, 25), (0, 25), (0, 15), (10, 15)], 'rects': {0: (8, 5), 1: (8, 5), 2: (6, 10), 3: (8, 6), 4: (8, 4), 5: (8, 10), 6: (10, 6), 7: (4, 9), 8: (6, 9)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 6), (3, 4), (3, 5), (4, 5), (4, 6), (6, 7), (6, 8), (7, 8)], 'name': 'T-Shape: Classic 3-Arm Asymmetric Layout (9 rooms)'},
    2: {'polygon': [(10, 0), (20, 0), (20, 15), (30, 15), (30, 25), (0, 25), (0, 15), (10, 15)], 'rects': {0: (5, 5), 1: (5, 5), 2: (10, 10), 3: (20, 10), 4: (4, 10), 5: (6, 10)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5)], 'interior': [], 'name': 'T-Shape: Stem-to-Crossbar Double-Overhang Bridge (6 rooms)'},
    3: {'polygon': [(10, 0), (20, 0), (20, 14), (30, 14), (30, 26), (0, 26), (0, 14), (10, 14)], 'rects': {0: (5, 5), 1: (5, 5), 2: (10, 9), 3: (18, 4), 4: (6, 7), 5: (9, 5), 6: (12, 5), 7: (9, 5), 8: (6, 7), 9: (18, 3)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 8), (3, 9), (4, 5), (4, 9), (5, 6), (5, 9), (6, 7), (6, 9), (7, 8), (7, 9), (8, 9)], 'interior': [9], 'name': 'T-Shape: Double-Overhang Bridge + 1 Interior Room (10 rooms)'},
    4: {'polygon': [(10, 0), (20, 0), (20, 15), (30, 15), (30, 25), (0, 25), (0, 15), (10, 15)], 'rects': {0: (5, 5), 1: (5, 5), 2: (10, 12), 3: (18, 4), 4: (10, 2), 5: (10, 2), 6: (6, 8), 7: (6, 8), 8: (18, 4)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 6), (5, 7), (6, 8), (7, 8)], 'interior': [3], 'name': 'T-Shape: Protruding Stem + Overhanging Interior Room (9 rooms, 1 int)'},
    5: {'polygon': [(8, 0), (20, 0), (20, 12), (28, 12), (28, 24), (0, 24), (0, 12), (8, 12)], 'rects': {0: (5, 6), 1: (5, 6), 2: (3, 8), 3: (10, 4), 4: (5, 4), 5: (8, 8), 6: (8, 4), 7: (6, 5), 8: (6, 5), 9: (12, 7), 10: (7, 8), 11: (5, 8)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 10), (3, 4), (3, 10), (4, 5), (4, 11), (5, 6), (5, 11), (6, 11), (7, 8), (7, 9), (8, 9), (9, 10), (9, 11), (10, 11)], 'interior': [10, 11], 'name': 'T-Shape: Dual Interior Core Touching Reflex Vertices (12 rooms, 2 int)'},
    6: {'polygon': [(10, 0), (20, 0), (20, 15), (30, 15), (30, 25), (0, 25), (0, 15), (10, 15)], 'rects': {0: (5, 5), 1: (5, 5), 2: (10, 12), 3: (10, 4), 4: (10, 2), 5: (10, 2), 6: (6, 8), 7: (6, 8), 8: (18, 4), 9: (8, 4)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (2, 9), (3, 4), (3, 6), (3, 8), (3, 9), (4, 6), (5, 7), (5, 9), (6, 8), (7, 8), (7, 9), (8, 9)], 'interior': [3, 9], 'name': 'T-Shape: Staggered Brick-Laid Dual Interior Rooms (10 rooms, 2 int)'},
    7: {'polygon': [(10, 0), (22, 0), (22, 12), (32, 12), (32, 26), (0, 26), (0, 12), (10, 12)], 'rects': {0: (6, 7), 1: (6, 7), 2: (4, 10), 3: (9, 4), 4: (11, 4), 5: (6, 8), 6: (10, 6), 7: (4, 4), 8: (6, 10), 9: (6, 10), 10: (12, 6), 11: (4, 6), 12: (4, 6), 13: (4, 6)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 8), (3, 4), (3, 8), (4, 5), (4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (6, 7), (6, 9), (7, 9), (8, 9), (8, 10), (9, 10), (10, 11), (10, 12), (10, 13), (11, 12), (12, 13)], 'interior': [7, 8, 9], 'name': 'T-Shape: Triple Interior Core in Junction Hub (14 rooms, 3 int)'},
    8: {'name': 'T-Shape (Sideways 90°): 3-Layer Interior Core & Vertical Straddle (14 rooms, 3 int)', 'polygon': [(0, 0), (12, 0), (12, 9), (26, 9), (26, 21), (12, 21), (12, 30), (0, 30)], 'rects': {0: (5, 6), 1: (5, 6), 2: (11, 12), 3: (4, 9), 4: (4, 11), 5: (3, 14), 6: (2, 9), 7: (2, 9), 8: (4, 5), 9: (4, 5), 10: (6, 8), 11: (3, 7), 12: (3, 7), 13: (6, 8)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 8), (3, 10), (4, 5), (4, 7), (4, 9), (4, 13), (5, 10), (5, 11), (5, 12), (5, 13), (6, 8), (7, 9), (8, 10), (9, 13), (10, 11), (11, 12), (12, 13)], 'interior': [3, 4, 5]},
    9: {'name': 'T-Shape: Deep-Stem 2x2 Quad Interior Core (16 rooms, 4 int)', 'polygon': [(10, 0), (22, 0), (22, 14), (32, 14), (32, 28), (0, 28), (0, 14), (10, 14)], 'rects': {0: (6, 4), 1: (6, 4), 2: (4, 10), 3: (8, 10), 4: (5, 7), 5: (5, 7), 6: (5, 10), 7: (11, 4), 8: (11, 4), 9: (5, 10), 10: (5, 7), 11: (5, 7), 12: (6, 5), 13: (6, 5), 14: (7, 5), 15: (5, 5)}, 'required_adjacency': [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 12), (3, 12), (3, 13), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 12), (6, 14), (7, 8), (7, 14), (8, 9), (8, 11), (8, 14), (8, 15), (9, 10), (9, 11), (9, 13), (9, 15), (10, 11), (12, 13), (12, 14), (13, 14), (13, 15), (14, 15)], 'interior': [12, 13, 14, 15]},
    10: {'name': 'T-Shape (Asymmetric Long-Wing): 17 Rooms, 3 Interior Rooms', 'polygon': [(8, 0), (18, 0), (18, 12), (34, 12), (34, 24), (0, 24), (0, 12), (8, 12)], 'rects': {0: (5, 4), 1: (5, 4), 2: (10, 4), 3: (10, 4), 4: (4, 6), 5: (4, 6), 6: (4, 8), 7: (10, 4), 8: (10, 4), 9: (6, 4), 10: (6, 4), 11: (6, 4), 12: (4, 6), 13: (4, 6), 14: (10, 8), 15: (6, 4), 16: (6, 4)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (3, 14), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 14), (7, 8), (7, 14), (8, 9), (8, 14), (8, 15), (9, 13), (9, 16), (10, 11), (10, 14), (10, 15), (11, 12), (11, 16), (12, 13), (12, 16), (13, 16), (14, 15), (15, 16)], 'interior': [14, 15, 16]},
    11: {'name': 'Stepped T-Shape (3 Reflex Angles): 13 Rooms, 2 Interior Rooms', 'polygon': [(8, 0), (18, 0), (18, 12), (28, 12), (28, 20), (22, 20), (22, 24), (0, 24), (0, 12), (8, 12)], 'rects': {0: (5, 5), 1: (5, 5), 2: (10, 7), 3: (4, 6), 4: (4, 6), 5: (4, 8), 6: (9, 4), 7: (9, 4), 8: (6, 4), 9: (4, 8), 10: (6, 4), 11: (5, 8), 12: (5, 8)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 11), (2, 12), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 11), (6, 7), (6, 11), (7, 10), (7, 12), (8, 9), (8, 10), (8, 12), (9, 10), (10, 12), (11, 12)], 'interior': [11, 12]},
    12: {'polygon': [(8, 0), (18, 0), (18, 8), (26, 8), (26, 18), (18, 18), (18, 26), (8, 26), (8, 18), (0, 18), (0, 8), (8, 8)], 'rects': {0: (5, 4), 1: (5, 4), 2: (10, 4), 3: (8, 10), 4: (8, 5), 5: (8, 5), 6: (4, 8), 7: (6, 8), 8: (10, 10)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 8), (3, 8), (4, 5), (4, 8), (5, 8), (6, 7), (6, 8), (7, 8)], 'interior': [8], 'name': 'Cross / Plus Shape (4 Reflex Angles): Central Interior Hub (9 rooms, 1 int)'},
    13: {'name': 'Irregular T-Shape (Uneven Arms & Uneven Bottom Stem Rooms: 16 rooms, 5 int)', 'polygon': [(8, 0), (22, 0), (22, 14), (32, 14), (32, 26), (0, 26), (0, 10), (8, 10)], 'rects': {0: (6, 4), 1: (8, 7), 2: (6, 6), 3: (8, 7), 4: (4, 8), 5: (4, 8), 6: (4, 6), 7: (9, 4), 8: (11, 4), 9: (8, 6), 10: (10, 6), 11: (6, 4), 12: (4, 6), 13: (7, 8), 14: (7, 8), 15: (2, 2)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 11), (3, 11), (3, 13), (3, 14), (4, 5), (4, 6), (4, 12), (5, 7), (5, 12), (6, 11), (6, 12), (6, 13), (7, 8), (7, 12), (7, 13), (8, 9), (8, 13), (8, 14), (8, 15), (9, 10), (9, 15), (10, 14), (10, 15), (11, 13), (12, 13), (13, 14), (14, 15)], 'interior': [11, 12, 13, 14, 15]},
    14: {'name': 'Irregular Stepped-Bottom T-Shape (Lower/Higher Bottom Base: 15 rooms, 5 int)', 'polygon': [(8, 0), (15, 0), (15, 4), (22, 4), (22, 14), (32, 14), (32, 26), (0, 26), (0, 12), (8, 12)], 'rects': {0: (7, 4), 1: (7, 10), 2: (7, 8), 3: (4, 7), 4: (4, 7), 5: (4, 10), 6: (11, 4), 7: (11, 4), 8: (6, 8), 9: (10, 4), 10: (7, 5), 11: (7, 5), 12: (7, 4), 13: (7, 4), 14: (4, 4)}, 'required_adjacency': [(0, 2), (1, 2), (1, 10), (1, 12), (2, 10), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 10), (5, 11), (6, 7), (6, 11), (7, 8), (7, 13), (7, 14), (8, 9), (8, 14), (9, 12), (9, 14), (10, 11), (10, 12), (11, 12), (11, 13), (12, 13), (13, 14)], 'interior': [10, 11, 12, 13, 14]},
    15: {'name': 'T-Shape with Unequal Bottom Stem & 6 Interior Rooms (18 rooms, 6 int)', 'polygon': [(8, 0), (22, 0), (22, 12), (32, 12), (32, 26), (0, 26), (0, 12), (8, 12)], 'rects': {0: (5, 4), 1: (9, 6), 2: (5, 8), 3: (9, 6), 4: (4, 7), 5: (4, 7), 6: (4, 10), 7: (10, 4), 8: (12, 4), 9: (6, 8), 10: (10, 4), 11: (6, 2), 12: (5, 5), 13: (5, 5), 14: (5, 5), 15: (5, 5), 16: (4, 10), 17: (4, 6)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 12), (3, 14), (3, 16), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 12), (6, 13), (7, 8), (7, 13), (7, 15), (8, 9), (8, 15), (8, 16), (8, 17), (9, 11), (9, 17), (10, 11), (10, 16), (10, 17), (11, 17), (12, 13), (12, 14), (13, 15), (14, 15), (14, 16), (15, 16), (16, 17)], 'interior': [12, 13, 14, 15, 16, 17]},
    16: {'polygon': [(0, 0), (25, 0), (25, 15), (15, 15), (15, 30), (0, 30)], 'rects': {0: (15, 6), 1: (10, 6), 2: (15, 4), 3: (15, 4), 4: (8, 7), 5: (7, 7), 6: (5, 9), 7: (16, 9), 8: (4, 9)}, 'required_adjacency': [(0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 7), (2, 3), (3, 6), (3, 7), (3, 8), (4, 5), (6, 7), (6, 8)], 'name': 'L-Shape (Original Ex 1: 9 rooms, 0 interior)'},
    17: {'polygon': [(0, 0), (24, 0), (24, 16), (16, 16), (16, 24), (0, 24)], 'rects': {0: (24, 4), 1: (8, 7), 2: (8, 7), 3: (13, 4), 4: (7, 4), 5: (4, 4), 6: (16, 4), 7: (16, 5), 8: (7, 4), 9: (8, 4), 10: (1, 4)}, 'required_adjacency': [(0, 6), (0, 7), (1, 4), (1, 7), (2, 3), (2, 4), (3, 5), (3, 6), (5, 6), (8, 3), (8, 6), (8, 7), (8, 0), (8, 9), (9, 3), (9, 2), (9, 4), (9, 7), (9, 10), (10, 4), (10, 1), (10, 7)], 'interior': [8, 9, 10], 'name': 'L-Shape (Original Ex 2: 11 rooms, 3 interior)'},
    18: {'polygon': [(0, 0), (35, 0), (35, 20), (22, 20), (22, 40), (0, 40)], 'rects': {0: (8, 8), 1: (13, 10), 2: (14, 8), 3: (17, 12), 4: (12, 10), 5: (5, 5), 6: (5, 5), 7: (12, 10), 8: (6, 5), 9: (6, 5), 10: (10, 10), 11: (5, 7), 12: (5, 5), 13: (12, 10)}, 'required_adjacency': [(0, 2), (0, 12), (0, 3), (2, 3), (3, 12), (3, 11), (12, 11), (3, 10), (11, 10), (10, 5), (10, 4), (5, 4), (5, 6), (6, 4), (4, 7), (7, 8), (7, 9), (7, 1), (8, 9), (9, 1), (13, 4), (13, 7), (13, 3), (13, 10), (13, 1)], 'interior': [13], 'name': 'L-Shape (Original Ex 3: 14 rooms, 1 interior)'},
    19: {'polygon': [(0, 0), (24, 0), (24, 8), (12, 8), (12, 20), (0, 20)], 'rects': {0: (6, 8), 1: (6, 6), 2: (3, 7), 3: (6, 4), 4: (6, 2), 5: (4, 6), 6: (11, 4), 7: (9, 2), 8: (3, 8), 9: (3, 3), 10: (6, 4), 11: (5, 5), 12: (3, 9)}, 'required_adjacency': [(0, 3), (0, 5), (1, 2), (1, 4), (2, 6), (3, 4), (3, 5), (6, 8), (7, 8), (9, 2), (9, 6), (9, 8), (9, 7), (9, 11), (10, 5), (10, 3), (10, 4), (10, 1), (10, 2), (10, 11), (11, 5), (11, 2), (11, 7), (11, 12), (12, 7)], 'interior': [9, 10], 'name': 'L-Shape (Original Ex 4: 13 rooms, 2 interior)'},
    20: {'polygon': [(0, 0), (20, 0), (20, 15), (10, 15), (10, 30), (0, 30)], 'rects': {0: (6, 6), 1: (11, 4), 2: (1, 5), 3: (2, 4), 4: (9, 6), 5: (6, 10), 6: (5, 10), 7: (9, 5), 8: (10, 2), 9: (4, 10), 10: (5, 9), 11: (3, 5), 12: (4, 7)}, 'required_adjacency': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0), (9, 0), (9, 1), (9, 7), (9, 8), (9, 10), (9, 11), (10, 2), (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 1), (11, 2), (12, 3), (12, 4), (12, 5)], 'interior': [9, 10, 11, 12], 'name': 'L-Shape (Original Ex 5: 13 rooms, 4 interior)'},
    21: {'polygon': [(0, 0), (16, 0), (16, 6), (9, 6), (9, 14), (0, 14)], 'rects': {0: (6, 4), 1: (3, 3), 2: (3, 4), 3: (5, 3), 4: (3, 8), 5: (3, 6), 6: (5, 2), 7: (3, 4), 8: (2, 3), 9: (6, 2), 10: (4, 5), 11: (3, 1), 12: (1, 3)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (3, 5), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 3), (10, 5), (10, 6), (10, 7), (10, 11), (10, 12), (11, 7), (11, 8), (11, 9), (11, 12), (12, 3), (12, 9)], 'interior': [10, 11, 12], 'name': 'L-Shape (Original Ex 6: 13 rooms, 3 interior)'},
    22: {'polygon': [(0, 0), (16, 0), (16, 8), (6, 8), (6, 23), (0, 23)], 'rects': {0: (3, 2), 1: (3, 2), 2: (6, 2), 3: (3, 2), 4: (3, 2), 5: (4, 6), 6: (5, 6), 7: (8, 5), 8: (8, 3), 9: (3, 2), 10: (3, 2), 11: (3, 4), 12: (5, 4), 13: (3, 4), 14: (4, 2)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (1, 5), (2, 5), (4, 5), (5, 6), (6, 7), (6, 8), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8), (14, 9), (14, 10), (14, 12), (14, 13), (14, 8)], 'interior': [14], 'name': 'L-Shape (Original Ex 7: 15 rooms, 1 interior)'},
    23: {'polygon': [(0, 7), (0, 13), (20, 13), (20, 0), (13, 0), (13, 7)], 'rects': {0: (7, 4), 1: (7, 3), 2: (2, 3), 3: (4, 4), 4: (2, 3), 5: (2, 2), 6: (2, 5), 7: (2, 2), 8: (2, 4), 9: (4, 5), 10: (3, 4), 11: (3, 4), 12: (11, 1), 13: (3, 3), 14: (1, 2)}, 'required_adjacency': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9), (9, 10), (10, 11), (12, 7), (12, 9), (12, 6), (12, 5), (12, 4), (12, 3), (12, 10), (12, 11), (12, 13), (13, 3), (13, 0), (13, 11), (13, 14), (14, 3), (14, 2), (14, 0)], 'interior': [12, 13, 14], 'name': 'L-Shape (Original Ex 8: 15 rooms, 3 interior)'},
    24: {'polygon': [(0, 0), (21, 0), (21, 6), (9, 6), (9, 18), (0, 18)], 'rects': {0: (6, 8), 1: (6, 4), 2: (2, 2), 3: (3, 2), 4: (4, 5), 5: (2, 5), 6: (3, 5), 7: (2, 5), 8: (3, 3), 9: (4, 9), 10: (6, 3), 11: (4, 8), 12: (2, 1)}, 'required_adjacency': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 7), (10, 8), (10, 9), (10, 11), (11, 1), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 12), (12, 1), (12, 2), (12, 3)], 'interior': [12], 'name': 'L-Shape (Original Ex 9: 13 rooms, 1 interior)'},
    25: {'polygon': [(0, 0), (18, 0), (18, 13), (10, 13), (10, 24), (0, 24)], 'rects': {0: (9, 6), 1: (4, 4), 2: (4, 4), 3: (6, 4), 4: (4, 4), 5: (5, 6), 6: (3, 4), 7: (2, 4), 8: (4, 4), 9: (3, 4), 10: (5, 5), 11: (7, 7), 12: (5, 3), 13: (2, 7), 14: (7, 2), 15: (1, 3), 16: (5, 2), 17: (2, 5)}, 'required_adjacency': [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (13, 3), (13, 4), (13, 5), (13, 8), (13, 9), (13, 10), (13, 15), (14, 0), (14, 2), (14, 3), (14, 10), (14, 11), (14, 15), (14, 16), (15, 3), (15, 10), (16, 0), (16, 11), (16, 12), (16, 17), (17, 0), (17, 12)], 'interior': [13, 14, 15, 16], 'name': 'L-Shape (Original Ex 10: 18 rooms, 4 interior)'},
    26: {'name': 'Staggered Multi-Scale T-Shape (20 rooms: 13 exterior, 7 diverse interior)', 'polygon': [(7, 0), (21, 0), (21, 13), (29, 13), (29, 25), (0, 25), (0, 13), (7, 13)], 'rects': {0: (7, 7), 1: (3, 7), 2: (4, 9), 3: (5, 6), 4: (4, 7), 5: (4, 7), 6: (8, 3), 7: (5, 5), 8: (9, 4), 9: (5, 5), 10: (4, 3), 11: (6, 9), 12: (8, 3), 13: (5, 2), 14: (2, 7), 15: (3, 7), 16: (1, 4), 17: (9, 5), 18: (5, 4), 19: (4, 6)}, 'required_adjacency': [(0, 1), (0, 3), (0, 13), (1, 2), (1, 13), (2, 4), (2, 13), (3, 6), (3, 13), (3, 14), (4, 12), (4, 15), (4, 18), (4, 19), (5, 6), (5, 7), (5, 16), (6, 14), (6, 16), (6, 17), (7, 8), (7, 16), (7, 17), (8, 9), (8, 17), (9, 10), (9, 17), (9, 18), (9, 19), (10, 11), (10, 19), (11, 12), (11, 19), (12, 19), (13, 14), (13, 15), (14, 15), (14, 17), (15, 18), (16, 17), (17, 18), (18, 19)], 'interior': [13, 14, 15, 16, 17, 18, 19]},
    27: {'name': 'Staggered Multi-Scale T-Shape (22 rooms: 13 exterior, 9 diverse interior)', 'polygon': [(7, 0), (21, 0), (21, 13), (29, 13), (29, 25), (0, 25), (0, 13), (7, 13)], 'rects': {0: (7, 7), 1: (3, 5), 2: (4, 9), 3: (5, 6), 4: (4, 7), 5: (4, 7), 6: (8, 3), 7: (5, 5), 8: (9, 4), 9: (5, 5), 10: (4, 3), 11: (6, 9), 12: (8, 3), 13: (3, 2), 14: (5, 2), 15: (2, 7), 16: (3, 7), 17: (1, 4), 18: (9, 3), 19: (9, 2), 20: (5, 4), 21: (4, 6)}, 'required_adjacency': [(0, 1), (0, 3), (0, 13), (0, 14), (1, 2), (1, 13), (2, 4), (2, 13), (2, 14), (3, 6), (3, 14), (3, 15), (4, 12), (4, 16), (4, 20), (4, 21), (5, 6), (5, 7), (5, 17), (6, 15), (6, 17), (6, 18), (7, 8), (7, 17), (7, 19), (8, 9), (8, 19), (9, 10), (9, 19), (9, 20), (9, 21), (10, 11), (10, 21), (11, 12), (11, 21), (12, 21), (13, 14), (14, 15), (14, 16), (15, 16), (15, 18), (16, 20), (17, 18), (17, 19), (18, 19), (18, 20), (19, 20), (20, 21)], 'interior': [13, 14, 15, 16, 17, 18, 19, 20, 21]},
    28: {'name': 'Irregular Stepped-Bottom Staggered T-Shape (22 rooms: 13 exterior, 9 diverse interior)', 'polygon': [(7, 0), (14, 0), (14, 3), (21, 3), (21, 13), (29, 13), (29, 25), (0, 25), (0, 11), (7, 11)], 'rects': {0: (7, 7), 1: (3, 4), 2: (4, 6), 3: (5, 4), 4: (4, 7), 5: (7, 2), 6: (4, 7), 7: (5, 5), 8: (9, 4), 9: (5, 5), 10: (4, 3), 11: (6, 9), 12: (8, 3), 13: (5, 2), 14: (2, 7), 15: (3, 7), 16: (5, 2), 17: (8, 3), 18: (1, 4), 19: (9, 5), 20: (5, 4), 21: (4, 6)}, 'required_adjacency': [(0, 1), (0, 3), (0, 13), (1, 2), (1, 13), (2, 4), (2, 13), (3, 13), (3, 14), (3, 16), (4, 12), (4, 15), (4, 20), (4, 21), (5, 6), (5, 16), (5, 17), (6, 7), (6, 17), (6, 18), (7, 8), (7, 18), (7, 19), (8, 9), (8, 19), (9, 10), (9, 19), (9, 20), (9, 21), (10, 11), (10, 21), (11, 12), (11, 21), (12, 21), (13, 14), (13, 15), (14, 15), (14, 16), (14, 17), (14, 19), (15, 20), (16, 17), (17, 18), (17, 19), (18, 19), (19, 20), (20, 21)], 'interior': [13, 14, 15, 16, 17, 18, 19, 20, 21]},
}

DEFAULT_EXAMPLE = 27


# --------------------------------------------------------------------------
# Interactive runner
# --------------------------------------------------------------------------

def run_example(
    choice: int = DEFAULT_EXAMPLE,
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

    _solve_and_report(
        str(choice), polygon, rects, required_adjacency, max_solutions, time_limit,
        interior_ids=example.get("interior", ()),
        title=example.get("name", f"Example {choice}"),
    )


def run_custom_example(max_solutions: int = 5, time_limit: Optional[float] = 60.0):
    """Prompt the user for their own polygon / rects / required_adjacency
    and run the same solve-and-report flow as `run_example`. Each value is
    typed as a plain Python literal and parsed with `ast.literal_eval`."""
    print("\nEnter your own floorplan. Each value is a Python literal, e.g.:")
    print("  polygon: [(10,0), (20,0), (20,15), (30,15), (30,25), (0,25), (0,15), (10,15)]")
    print("  rects:   {0: (10,15), 1: (20,10), 2: (6,10), 3: (4,10)}")
    print("  required_adjacency: [(0,1), (1,2), (1,3)]   (or [] for none)")
    print("  interior rects (must not touch the boundary): []   (or e.g. [3])")

    def _parse_val(raw_s):
        s = raw_s.strip()
        if "=" in s and not s.startswith(("{", "[", "(")):
            s = s.split("=", 1)[1].strip()
        if s.endswith(","):
            s = s[:-1].strip()
        if s.startswith("dict(enumerate(") and s.endswith("))"):
            inner = ast.literal_eval(s[len("dict(enumerate("):-2])
            return dict(enumerate(inner))
        return ast.literal_eval(s)

    try:
        polygon = _parse_val(input("\npolygon: "))
        rects_raw = _parse_val(input("rects (dict {id: (w,h)} or list [(w,h), ...]): "))
        if isinstance(rects_raw, dict):
            rects = {int(k): tuple(v) for k, v in rects_raw.items()}
        else:
            rects = {i: tuple(v) for i, v in enumerate(rects_raw)}
        adj_raw = input("required_adjacency (blank for none): ").strip()
        required_adjacency = _normalize_adjacency_pairs(_parse_val(adj_raw)) if adj_raw else []
        interior_raw = input("interior rects (blank for none): ").strip()
        interior_ids = _parse_val(interior_raw) if interior_raw else []
    except Exception as exc:
        print(f"Could not parse that: {exc}")
        return

    _solve_and_report("custom", polygon, rects, required_adjacency, max_solutions, time_limit,
                      interior_ids=interior_ids, title="Custom Floorplan")


def _solve_and_report(label, polygon, rects, required_adjacency, max_solutions, time_limit, interior_ids=(), title=""):
    n_adj = len(required_adjacency or [])
    n_reflex = len(_polygon_reflex_vertices(polygon))
    header = f"{label} ({title})" if title else str(label)
    print(f"\nRunning example {header}: {len(rects)} rects, {n_adj} adjacencies, {n_reflex} reflex angle(s)")
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

    if found is True or found is False:
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
    render_solution_grid(polygon, solutions, save_path, interior_ids=interior_ids)
    print(f"  saved to {save_path}")



def run_draw_boundary_ui(max_solutions: int = 5, time_limit: Optional[float] = 60.0):
    """Interactive Matplotlib window where the user clicks grid vertices to draw
    a custom orthogonal (T-shape / irregular T-shape / rectilinear) plot boundary,
    then enters room sizes in the terminal and runs `pack_rectangles_frontier`."""
    import matplotlib.pyplot as plt

    print("\nOpening interactive Matplotlib boundary drawer...")
    print("  - Left-click grid points to add orthogonal polygon vertices.")
    print("  - Right-click (or press Enter / close window) once finished to close the polygon.")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 32)
    ax.set_xticks(range(0, 37, 2))
    ax.set_yticks(range(0, 33, 2))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title("Click to Draw Orthogonal Plot Boundary (Right-Click or Close When Done)", fontsize=10)

    pts: List[Tuple[float, float]] = []
    line_plot, = ax.plot([], [], "b-o", linewidth=2)

    def redraw():
        if not pts:
            line_plot.set_data([], [])
        else:
            xs = [p[0] for p in pts] + ([pts[0][0]] if len(pts) >= 4 else [])
            ys = [p[1] for p in pts] + ([pts[0][1]] if len(pts) >= 4 else [])
            line_plot.set_data(xs, ys)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 3 and len(pts) >= 4:
            plt.close(fig)
            return
        wx, wy = float(round(event.xdata)), float(round(event.ydata))
        if pts:
            lx, ly = pts[-1]
            if abs(wx - lx) < abs(wy - ly):
                wx = lx
            else:
                wy = ly
        if not pts or (wx, wy) != pts[-1]:
            pts.append((wx, wy))
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(pts) < 4:
        print("Polygon needs at least 4 vertices. Cancelled.")
        return
    # Ensure closure is orthogonal
    if pts[-1][0] != pts[0][0] and pts[-1][1] != pts[0][1]:
        pts.append((pts[0][0], pts[-1][1]))

    poly_area = Polygon(pts).area
    print(f"\nDrawn polygon ({len(pts)} vertices, total area = {poly_area:g}):")
    print(f"  {pts}")
    try:
        rects_raw = ast.literal_eval(input(f"Enter rects dict summing to area {poly_area:g}, e.g. {{0: (6,4), ...}}: ").strip())
        rects = {int(k): tuple(v) for k, v in rects_raw.items()}
        adj_raw = input("Enter required_adjacency list (or blank for free adjacency): ").strip()
        required_adjacency = ast.literal_eval(adj_raw) if adj_raw else None
        int_raw = input("Enter interior room ids list (or blank for none): ").strip()
        interior_ids = ast.literal_eval(int_raw) if int_raw else []
    except (ValueError, SyntaxError) as exc:
        print(f"Could not parse input: {exc}")
        return

    _solve_and_report(
        "drawn_custom", pts, rects, required_adjacency, max_solutions, time_limit,
        interior_ids=interior_ids, title="Custom Drawn Floorplan"
    )


def run_python_web_ui(port: int = 8765):
    """Starts a local Python HTTP server on `http://127.0.0.1:<port>` where the
    browser UI calls `pack_rectangles_frontier` in THIS Python file directly
    via `POST /api/solve`."""
    import json
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class PackerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            if self.path == "/api/solve":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                polygon = [tuple(p) for p in body["polygon"]]
                rects = {int(k): (float(v[0]), float(v[1])) for k, v in body["rects"].items()}
                adj = [tuple(pair) for pair in body["adj"]] if body.get("adj") else None
                interior = [int(i) for i in body.get("interior", [])]
                t0 = time.perf_counter()
                sols = []
                found, _, nodes = pack_rectangles_frontier(
                    polygon, rects, adj,
                    require_boundary_touch=True,
                    interior_ids=interior,
                    time_limit=float(body.get("time_limit", 15.0)),
                    collect_solutions=sols,
                    max_solutions=int(body.get("max_solutions", 5)),
                )
                dt = time.perf_counter() - t0
                resp = json.dumps({
                    "found": found,
                    "solutions": sols,
                    "nodes": nodes,
                    "time_s": round(dt, 4),
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return
            self.send_response(404)
            self.end_headers()

    print(f"\nStarting Python Floorplan Packer Server at http://127.0.0.1:{port}")
    print("  Every solve request in the UI calls `pack_rectangles_frontier` in Python directly.")
    print("  Press Ctrl+C to stop the server.")
    server = HTTPServer(("127.0.0.1", port), PackerHandler)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped server.")
    finally:
        server.server_close()


def main():
    print("Available built-in examples (1-15: T-Shape, Irregular T & Multi-Reflex; 16-25: L-Shape):")
    for key, ex in EXAMPLES.items():
        n_adj = len(ex.get("required_adjacency") or [])
        n_int = len(ex.get("interior") or [])
        desc = ex.get("name", f"{len(ex['rects'])} rects, {n_adj} adjacencies")
        print(f"  {key:2d}. {desc} [{len(ex['rects'])} rects, {n_adj} adj, {n_int} int]")
    print("   a. render showcase grid of all 15 T-shape, irregular T & multi-reflex examples (1-15)")
    print("   d. draw custom plot boundary interactively in Matplotlib + pack rooms")
    print("   c. enter custom polygon / rooms / adjacencies in terminal")

    try:
        raw = input(
            f"\nChoose an example [{min(EXAMPLES)}-{max(EXAMPLES)}], 'a' (all T-shapes), 'd' (draw boundary), or 'c' [Press Enter for default={DEFAULT_EXAMPLE}]: "
        ).strip()
    except EOFError:
        raw = ""

    if not raw:
        print(f"Using default Irregular T-shape example {DEFAULT_EXAMPLE}...")
        run_example(DEFAULT_EXAMPLE)
        return

    if raw.lower() == "d":
        run_draw_boundary_ui()
        return

    if raw.lower() == "c":
        run_custom_example()
        return

    if raw.lower() == "a":
        out_file = render_examples_showcase(list(range(1, 16)), cols=5)
        print(f"  saved 15-example T-shape showcase grid to {out_file}")
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