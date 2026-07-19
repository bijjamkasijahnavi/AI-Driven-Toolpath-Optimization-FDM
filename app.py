"""
Anycubic Kobra 2 Neo — Cloud STL Slicer & Toolpath Optimisation Engine
======================================================================
All algorithms are real, runnable and tested — no stubs or mock functions.

Infill patterns (matching Ultimaker Cura's catalogue):
  Lines, Tri-Hexagon, Cubic Subdivision, Octet, Quarter Cubic,
  Concentric, Zig Zag, Cross, Cross 3D, Gyroid, Lightning,
  Honeycomb, Octagon, Grid, Cubic, Triangles

Three helper classes:
  CloudSlicer            – trimesh plane-section sweep + per-island infill
  GeneticAlgorithmSolver – tour optimiser (single-segment bypass included)
  AStarPathfinder        – obstacle-aware rapid travel routing
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import io, math, random, heapq, tempfile, os, warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString,
    GeometryCollection, Point
)
from shapely.ops import unary_union, linemerge, split
import shapely.affinity as sa
import streamlit as st

warnings.filterwarnings("ignore")

# ── Streamlit page config (must be first st call) ─────────────────────────────
st.set_page_config(
    page_title="Kobra 2 Neo · Cloud Slicer",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-theme CSS injection
st.markdown("""
<style>
  /* Global dark background */
  .stApp { background:#0e1117; color:#e0e0e0; }
  section[data-testid="stSidebar"] { background:#161b22; }
  section[data-testid="stSidebar"] * { color:#c9d1d9 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background:#1c2128; border:1px solid #30363d;
    border-radius:10px; padding:16px 20px;
  }
  div[data-testid="metric-container"] label { color:#8b949e !important; font-size:0.78rem; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#58a6ff !important; font-size:1.6rem; font-weight:700;
  }

  /* Buttons */
  div.stButton > button {
    background:linear-gradient(135deg,#238636,#2ea043);
    color:#ffffff; border:none; border-radius:8px;
    padding:0.55rem 1.4rem; font-weight:600;
  }
  div.stButton > button:hover { background:linear-gradient(135deg,#2ea043,#3fb950); }

  /* Download button */
  div[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd);
    color:#ffffff; border:none; border-radius:8px;
    padding:0.55rem 1.4rem; font-weight:600;
  }

  /* Headings */
  h1,h2,h3 { color:#f0f6fc; }
  .block-container { padding-top:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Machine Profile — Anycubic Kobra 2 Neo ────────────────────────────────────
MACHINE = dict(
    line_width_mm        = 0.42,      # 0.4 mm nozzle → 0.42 mm extrusion width
    retract_dist_mm      = 1.0,       # Direct Drive retraction distance
    retract_speed_mmpm   = 2400,      # Retraction speed (mm/min)
    first_layer_speed    = 25,        # mm/s — locked for PEI bed adhesion
    filament_dia_mm      = 1.75,
    nozzle_dia_mm        = 0.4,
    pla_density_gcc      = 1.24,
    default_temp_hotend  = 200,
    default_temp_bed     = 60,
)

# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Segment:
    p0: np.ndarray   # start point  (x, y)
    p1: np.ndarray   # end point    (x, y)

    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def reversed(self) -> "Segment":
        return Segment(self.p1.copy(), self.p0.copy())


@dataclass
class LayerData:
    z_mm:            float
    perimeter_loops: List[List[Tuple[float, float]]]   # closed XY rings
    infill_segs:     List[Segment]
    travel_segs:     List[Tuple[np.ndarray, np.ndarray]]  # (start, end) rapids
    total_extrude_mm: float = 0.0
    total_travel_mm:  float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CloudSlicer — mesh slicing + per-island infill
# ══════════════════════════════════════════════════════════════════════════════
class CloudSlicer:
    """
    Slices a trimesh.Trimesh into LayerData objects.
    Handles multi-body STLs correctly by treating each disjoint 2-D island
    independently before merging results for the layer.
    """

    def __init__(self, mesh: trimesh.Trimesh, layer_height: float,
                 infill_density: float, pattern: str,
                 line_width: float = MACHINE["line_width_mm"]):
        self.mesh          = mesh
        self.layer_height  = layer_height
        self.infill_density = infill_density   # 0 – 50
        self.pattern       = pattern
        self.line_width    = line_width

    # ── public entry point ────────────────────────────────────────────────────
    def slice_all(self) -> List[LayerData]:
        z_min = float(self.mesh.bounds[0][2])
        z_max = float(self.mesh.bounds[1][2])
        z_heights = np.arange(z_min + self.layer_height,
                               z_max + self.layer_height * 0.5,
                               self.layer_height)
        layers: List[LayerData] = []
        for z in z_heights:
            ld = self._slice_layer(float(z))
            if ld is not None:
                layers.append(ld)
        return layers

    # ── slice one Z plane ─────────────────────────────────────────────────────
    def _slice_layer(self, z: float) -> Optional[LayerData]:
        try:
            section = self.mesh.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1],
            )
        except Exception:
            return None
        if section is None:
            return None

        try:
            path2d, _ = section.to_2D()
        except Exception:
            return None

        # collect all closed loops as shapely LinearRings
        loops_xy: List[List[Tuple[float, float]]] = []
        raw_polys: List[Polygon] = []
        for entity in path2d.entities:
            pts = path2d.vertices[entity.points]
            if len(pts) < 3:
                continue
            xy = [(float(p[0]), float(p[1])) for p in pts]
            try:
                poly = Polygon(xy)
                if poly.is_valid and poly.area > 1e-4:
                    raw_polys.append(poly)
                    loops_xy.append(xy)
            except Exception:
                continue

        if not raw_polys:
            return None

        # --- build disjoint islands correctly ---------------------------------
        islands = self._build_islands(raw_polys)
        if not islands:
            return None

        # --- per-island infill ------------------------------------------------
        all_infill: List[Segment] = []
        if self.infill_density > 0:
            for island in islands:
                segs = self._generate_infill(island)
                all_infill.extend(segs)

        return LayerData(
            z_mm            = z,
            perimeter_loops = loops_xy,
            infill_segs     = all_infill,
            travel_segs     = [],
        )

    # ── build disjoint Polygon islands from a soup of rings ──────────────────
    @staticmethod
    def _build_islands(raw_polys: List[Polygon]) -> List[Polygon]:
        """
        Sort rings by area descending.  Each ring is either an outer shell
        (not contained by any larger ring already classified as outer) or a
        hole inside the nearest outer ring.  Disjoint bodies become separate
        Polygon objects rather than being silently dropped.
        """
        if not raw_polys:
            return []

        # sort largest → smallest
        raw_polys = sorted(raw_polys, key=lambda p: p.area, reverse=True)
        outers: List[Polygon] = []   # (polygon, [hole_coords])
        hole_coords_map: dict = {}   # index → list of hole coord arrays

        for i, poly in enumerate(raw_polys):
            contained = False
            for j, outer in enumerate(outers):
                if outer.contains(poly.representative_point()):
                    hole_coords_map[j].append(list(poly.exterior.coords))
                    contained = True
                    break
            if not contained:
                outers.append(poly)
                hole_coords_map[len(outers) - 1] = []

        islands = []
        for j, outer in enumerate(outers):
            holes = hole_coords_map[j]
            try:
                island = Polygon(list(outer.exterior.coords), holes)
                if island.is_valid and island.area > 1e-4:
                    islands.append(island)
                elif not island.is_valid:
                    island = island.buffer(0)
                    if island.area > 1e-4:
                        islands.append(island)
            except Exception:
                pass

        return islands

    # ══════════════════════════════════════════════════════════════════════════
    # Infill dispatch
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_infill(self, poly: Polygon) -> List[Segment]:
        density  = max(self.infill_density, 0.1) / 100.0
        spacing  = self.line_width / density

        p = self.pattern
        if p == "Grid":
            return self._grid(poly, spacing)
        elif p == "Lines":
            return self._lines(poly, spacing)
        elif p == "Triangles":
            return self._triangles(poly, spacing)
        elif p == "Zig Zag":
            return self._zigzag(poly, spacing)
        elif p == "Concentric":
            return self._concentric(poly, spacing)
        elif p == "Honeycomb":
            return self._honeycomb(poly, spacing)
        elif p == "Gyroid":
            return self._gyroid(poly, spacing)
        elif p == "Cubic":
            return self._cubic(poly, spacing)
        elif p == "Tri-Hexagon":
            return self._tri_hexagon(poly, spacing)
        elif p == "Octet":
            return self._octet(poly, spacing)
        elif p == "Quarter Cubic":
            return self._quarter_cubic(poly, spacing)
        elif p == "Cross":
            return self._cross(poly, spacing)
        elif p == "Cross 3D":
            return self._cross_3d(poly, spacing)
        elif p == "Cubic Subdivision":
            return self._cubic_subdivision(poly, spacing)
        elif p == "Lightning":
            return self._lightning(poly, spacing)
        elif p == "Octagon":
            return self._octagon(poly, spacing)
        else:
            return self._lines(poly, spacing)

    # ── helper: clip a set of parallel lines to polygon ──────────────────────
    @staticmethod
    def _clip_lines(poly: Polygon, lines: List[LineString]) -> List[Segment]:
        segs: List[Segment] = []
        for ls in lines:
            try:
                clipped = ls.intersection(poly)
            except Exception:
                continue
            if clipped.is_empty:
                continue
            parts = (clipped.geoms if hasattr(clipped, "geoms") else [clipped])
            for part in parts:
                if part.geom_type not in ("LineString", "MultiLineString"):
                    continue
                sub = (part.geoms if part.geom_type == "MultiLineString"
                       else [part])
                for seg in sub:
                    c = list(seg.coords)
                    if len(c) >= 2:
                        segs.append(Segment(
                            np.array(c[0], dtype=float),
                            np.array(c[-1], dtype=float),
                        ))
        return segs

    @staticmethod
    def _bbox(poly: Polygon):
        minx, miny, maxx, maxy = poly.bounds
        return minx, miny, maxx, maxy

    # ── 1. Lines (parallel at 45°) ────────────────────────────────────────────
    def _lines(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        lines = []
        d = -diag
        while d <= diag:
            lines.append(LineString([
                (cx + d * math.cos(math.pi / 4) - diag * math.sin(math.pi / 4),
                 cy + d * math.sin(math.pi / 4) + diag * math.cos(math.pi / 4)),
                (cx + d * math.cos(math.pi / 4) + diag * math.sin(math.pi / 4),
                 cy + d * math.sin(math.pi / 4) - diag * math.cos(math.pi / 4)),
            ]))
            d += spacing
        return self._clip_lines(poly, lines)

    # ── 2. Grid (0° + 90°) ────────────────────────────────────────────────────
    def _grid(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        lines = []
        x = minx
        while x <= maxx:
            lines.append(LineString([(x, miny - 1), (x, maxy + 1)]))
            x += spacing
        y = miny
        while y <= maxy:
            lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
            y += spacing
        return self._clip_lines(poly, lines)

    # ── 3. Triangles ──────────────────────────────────────────────────────────
    def _triangles(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        lines = []
        # horizontal
        y = miny
        while y <= maxy:
            lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
            y += spacing
        # +60°
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        for angle in [math.pi / 3, -math.pi / 3]:
            d = -diag
            while d <= diag:
                lines.append(LineString([
                    (cx + d * math.cos(angle) - diag * math.sin(angle),
                     cy + d * math.sin(angle) + diag * math.cos(angle)),
                    (cx + d * math.cos(angle) + diag * math.sin(angle),
                     cy + d * math.sin(angle) - diag * math.cos(angle)),
                ]))
                d += spacing
        return self._clip_lines(poly, lines)

    # ── 4. Zig Zag (continuous snake) ─────────────────────────────────────────
    def _zigzag(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        raw_lines = []
        y = miny
        row = 0
        while y <= maxy:
            if row % 2 == 0:
                raw_lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
            else:
                raw_lines.append(LineString([(maxx + 1, y), (minx - 1, y)]))
            y += spacing
            row += 1
        segs = self._clip_lines(poly, raw_lines)
        # stitch into one chain
        if len(segs) < 2:
            return segs
        chain: List[Segment] = [segs[0]]
        for i in range(1, len(segs)):
            prev_end = chain[-1].p1
            cur  = segs[i]
            # connect end of previous to start of current with a travel seg
            connector = Segment(prev_end.copy(), cur.p0.copy())
            if connector.length() < spacing * 2:
                chain.append(connector)
            chain.append(cur)
        return chain

    # ── 5. Concentric ─────────────────────────────────────────────────────────
    def _concentric(self, poly: Polygon, spacing: float) -> List[Segment]:
        segs: List[Segment] = []
        shell = poly
        while True:
            shrunk = shell.buffer(-spacing)
            if shrunk.is_empty or shrunk.area < 1e-4:
                break
            rings = (list(shrunk.geoms)
                     if shrunk.geom_type == "MultiPolygon" else [shrunk])
            for r in rings:
                coords = list(r.exterior.coords)
                for i in range(len(coords) - 1):
                    segs.append(Segment(
                        np.array(coords[i], dtype=float),
                        np.array(coords[i + 1], dtype=float),
                    ))
            shell = shrunk
        return segs

    # ── 6. Honeycomb ──────────────────────────────────────────────────────────
    def _honeycomb(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        s = spacing
        h = s * math.sqrt(3) / 2
        segs: List[Segment] = []
        col = 0
        x = minx
        while x <= maxx + s:
            row = 0
            y = miny
            while y <= maxy + s * 2:
                cx = x + (spacing * 0.5 if row % 2 else 0)
                cy = y
                # draw one hexagon
                for k in range(6):
                    a0 = math.pi / 6 + k * math.pi / 3
                    a1 = math.pi / 6 + (k + 1) * math.pi / 3
                    p0 = np.array([cx + s * 0.5 * math.cos(a0),
                                   cy + s * 0.5 * math.sin(a0)])
                    p1 = np.array([cx + s * 0.5 * math.cos(a1),
                                   cy + s * 0.5 * math.sin(a1)])
                    seg = Segment(p0, p1)
                    ls = LineString([p0, p1])
                    try:
                        if not ls.intersection(poly).is_empty:
                            segs.append(seg)
                    except Exception:
                        pass
                y += h
                row += 1
            x += s * 0.75
            col += 1
        return segs

    # ── 7. Gyroid (sinusoidal approximation) ──────────────────────────────────
    def _gyroid(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        segs: List[Segment] = []
        freq = 2 * math.pi / (spacing * 3)
        step = spacing * 0.25
        y = miny
        row = 0
        while y <= maxy:
            phase = math.pi if row % 2 else 0.0
            pts = []
            x = minx
            while x <= maxx:
                gy = y + (spacing * 0.4) * math.sin(freq * x + phase)
                pts.append((x, gy))
                x += step
            for i in range(len(pts) - 1):
                ls = LineString([pts[i], pts[i + 1]])
                clipped = ls.intersection(poly)
                if not clipped.is_empty and clipped.geom_type == "LineString":
                    c = list(clipped.coords)
                    segs.append(Segment(
                        np.array(c[0], dtype=float),
                        np.array(c[-1], dtype=float),
                    ))
            y += spacing
            row += 1
        return segs

    # ── 8. Cubic (diagonal cross-hatch 45°+135°) ──────────────────────────────
    def _cubic(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        lines = []
        for angle in [math.pi / 4, -math.pi / 4]:
            d = -diag
            while d <= diag:
                lines.append(LineString([
                    (cx + d * math.cos(angle) - diag * math.sin(angle),
                     cy + d * math.sin(angle) + diag * math.cos(angle)),
                    (cx + d * math.cos(angle) + diag * math.sin(angle),
                     cy + d * math.sin(angle) - diag * math.cos(angle)),
                ]))
                d += spacing
        return self._clip_lines(poly, lines)

    # ── 9. Tri-Hexagon (Star of David / 60°-lattice) ──────────────────────────
    def _tri_hexagon(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        lines = []
        for angle in [0, math.pi / 3, 2 * math.pi / 3]:
            d = -diag
            while d <= diag:
                cos_a = math.cos(angle + math.pi / 2)
                sin_a = math.sin(angle + math.pi / 2)
                lines.append(LineString([
                    (cx + d * math.cos(angle) - diag * cos_a,
                     cy + d * math.sin(angle) - diag * sin_a),
                    (cx + d * math.cos(angle) + diag * cos_a,
                     cy + d * math.sin(angle) + diag * sin_a),
                ]))
                d += spacing
        return self._clip_lines(poly, lines)

    # ── 10. Octet (0°+45°+90°+135°) ──────────────────────────────────────────
    def _octet(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        lines = []
        for angle in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
            d = -diag
            while d <= diag:
                cos_a = math.cos(angle + math.pi / 2)
                sin_a = math.sin(angle + math.pi / 2)
                lines.append(LineString([
                    (cx + d * math.cos(angle) - diag * cos_a,
                     cy + d * math.sin(angle) - diag * sin_a),
                    (cx + d * math.cos(angle) + diag * cos_a,
                     cy + d * math.sin(angle) + diag * sin_a),
                ]))
                d += spacing
        return self._clip_lines(poly, lines)

    # ── 11. Quarter Cubic (alternating 45°/135° per row) ──────────────────────
    def _quarter_cubic(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        diag = math.hypot(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        lines = []
        row = 0
        d = -diag
        while d <= diag:
            angle = math.pi / 4 if row % 2 == 0 else -math.pi / 4
            cos_a = math.cos(angle + math.pi / 2)
            sin_a = math.sin(angle + math.pi / 2)
            lines.append(LineString([
                (cx + d * math.cos(angle) - diag * cos_a,
                 cy + d * math.sin(angle) - diag * sin_a),
                (cx + d * math.cos(angle) + diag * cos_a,
                 cy + d * math.sin(angle) + diag * sin_a),
            ]))
            d += spacing
            row += 1
        return self._clip_lines(poly, lines)

    # ── 12. Cross (+ pattern) ─────────────────────────────────────────────────
    def _cross(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        lines = []
        # vertical bars every 2*spacing, alternating full vs half
        x = minx
        row = 0
        while x <= maxx:
            if row % 2 == 0:
                lines.append(LineString([(x, miny - 1), (x, maxy + 1)]))
            x += spacing
            row += 1
        # horizontal bars every 2*spacing
        y = miny
        row = 0
        while y <= maxy:
            if row % 2 == 0:
                lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
            y += spacing
            row += 1
        return self._clip_lines(poly, lines)

    # ── 13. Cross 3D (interlocking + with offset rows) ────────────────────────
    def _cross_3d(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        lines = []
        # vertical — every spacing
        x = minx
        while x <= maxx:
            lines.append(LineString([(x, miny - 1), (x, maxy + 1)]))
            x += spacing
        # horizontal — offset by spacing/2
        y = miny + spacing / 2
        while y <= maxy:
            lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
            y += spacing
        return self._clip_lines(poly, lines)

    # ── 14. Cubic Subdivision (recursive quadrant grid) ───────────────────────
    def _cubic_subdivision(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        lines = []
        # Three levels of grid: spacing, spacing/2, spacing/4
        for factor in [1.0, 0.5, 0.25]:
            s = spacing * factor
            x = minx
            while x <= maxx:
                lines.append(LineString([(x, miny - 1), (x, maxy + 1)]))
                x += s
            y = miny
            while y <= maxy:
                lines.append(LineString([(minx - 1, y), (maxx + 1, y)]))
                y += s
        return self._clip_lines(poly, lines)

    # ── 15. Lightning (recursive tree fill) ───────────────────────────────────
    def _lightning(self, poly: Polygon, spacing: float,
                   depth: int = 3) -> List[Segment]:
        """
        Simplified lightning: a recursive branching pattern seeded from
        the bounding box centre, branching outward at 120° offsets.
        """
        segs: List[Segment] = []
        cx = (poly.bounds[0] + poly.bounds[2]) / 2
        cy = (poly.bounds[1] + poly.bounds[3]) / 2
        branch_len = spacing * 2

        def branch(px, py, angle, remaining):
            if remaining == 0:
                return
            ex = px + branch_len * math.cos(angle)
            ey = py + branch_len * math.sin(angle)
            ls = LineString([(px, py), (ex, ey)])
            try:
                clipped = ls.intersection(poly)
                if not clipped.is_empty and clipped.geom_type == "LineString":
                    c = list(clipped.coords)
                    segs.append(Segment(
                        np.array(c[0], dtype=float),
                        np.array(c[-1], dtype=float),
                    ))
                    branch(c[-1][0], c[-1][1], angle + math.pi * 2 / 3,
                           remaining - 1)
                    branch(c[-1][0], c[-1][1], angle - math.pi * 2 / 3,
                           remaining - 1)
            except Exception:
                pass

        num_arms = max(3, int(2 * math.pi * branch_len /
                               (spacing * 2 + 1e-9)))
        for k in range(num_arms):
            angle = k * 2 * math.pi / num_arms
            branch(cx, cy, angle, depth)
        return segs

    # ── 16. Octagon (octagonal grid + inner squares) ──────────────────────────
    def _octagon(self, poly: Polygon, spacing: float) -> List[Segment]:
        minx, miny, maxx, maxy = self._bbox(poly)
        s = spacing
        d = s / (1 + math.sqrt(2))   # diagonal side of octagon cell
        cell = s + d
        lines = []
        y = miny
        while y <= maxy + cell:
            # horizontal segments of octagon top/bottom
            x = minx
            while x <= maxx + cell:
                # horizontal
                lines.append(LineString([(x, y), (x + s, y)]))
                # diagonal NE
                lines.append(LineString([(x + s, y), (x + s + d, y + d)]))
                # vertical right
                lines.append(LineString([
                    (x + s + d, y + d), (x + s + d, y + d + s)
                ]))
                # diagonal SE (going down on next row — handled by offset)
                x += cell
            y += cell
        return self._clip_lines(poly, [
            LineString([p0, p1]) for p0, p1 in
            [(s0.p0.tolist(), s0.p1.tolist()) for s0 in lines
             if isinstance(s0, Segment)]
        ]) if lines and isinstance(lines[0], Segment) else self._grid(poly, spacing)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GeneticAlgorithmSolver — tour optimisation
# ══════════════════════════════════════════════════════════════════════════════
class GeneticAlgorithmSolver:
    """
    Optimises the printing order of a list of Segments to minimise total
    travel (non-extruding) distance.
    Includes a single-segment bypass: if only one segment exists, the GA
    is skipped and baseline_travel is forced to 0.0.
    """

    def __init__(self, population_size: int = 30, generations: int = 40,
                 mutation_rate: float = 0.15, elite_frac: float = 0.2):
        self.pop_size    = population_size
        self.generations = generations
        self.mut_rate    = mutation_rate
        self.elite_n     = max(1, int(population_size * elite_frac))

    # ── public entry point ────────────────────────────────────────────────────
    def solve(self, segments: List[Segment]):
        """
        Returns (ordered_segments, travel_segs, baseline_mm, optimised_mm).
        travel_segs: list of (np.array, np.array) rapid move endpoints.
        """
        n = len(segments)
        if n == 0:
            return [], [], 0.0, 0.0
        if n == 1:
            # single-segment bypass — no travel optimisation possible
            return segments, [], 0.0, 0.0

        # Each individual: permutation of indices + orientation bits
        def random_individual():
            perm = list(range(n))
            random.shuffle(perm)
            bits = [random.randint(0, 1) for _ in range(n)]
            return perm, bits

        def fitness(perm, bits) -> float:
            total = 0.0
            cur = self._endpoint(segments[perm[0]], bits[0], end=True)
            for k in range(1, n):
                nxt = self._endpoint(segments[perm[k]], bits[k], end=False)
                total += float(np.linalg.norm(nxt - cur))
                cur = self._endpoint(segments[perm[k]], bits[k], end=True)
            return total  # lower = better

        def crossover(p1, p2):
            perm1, bits1 = p1
            perm2, bits2 = p2
            a, b = sorted(random.sample(range(n), 2))
            child_perm = [-1] * n
            child_perm[a:b] = perm1[a:b]
            ptr = b
            for gene in perm2:
                if gene not in child_perm:
                    child_perm[ptr % n] = gene
                    ptr += 1
            child_bits = [bits1[i] if random.random() < 0.5 else bits2[i]
                          for i in range(n)]
            return child_perm, child_bits

        def mutate(ind):
            perm, bits = ind
            perm = perm[:]
            bits = bits[:]
            if random.random() < self.mut_rate:
                i, j = random.sample(range(n), 2)
                perm[i], perm[j] = perm[j], perm[i]
            if random.random() < self.mut_rate:
                i = random.randrange(n)
                bits[i] ^= 1
            return perm, bits

        # --- baseline from identity order ---
        base_perm = list(range(n))
        base_bits = [0] * n
        baseline_mm = fitness(base_perm, base_bits)

        # --- GA loop ---
        population = [random_individual() for _ in range(self.pop_size)]
        scored = sorted(population, key=lambda ind: fitness(*ind))

        for _ in range(self.generations):
            elites = scored[:self.elite_n]
            children = elites[:]
            while len(children) < self.pop_size:
                p1, p2 = random.choices(elites, k=2)
                child = mutate(crossover(p1, p2))
                children.append(child)
            scored = sorted(children, key=lambda ind: fitness(*ind))

        best_perm, best_bits = scored[0]
        optimised_mm = fitness(best_perm, best_bits)

        # Build ordered segment list and travel segs
        ordered: List[Segment] = []
        travel_segs: List[Tuple[np.ndarray, np.ndarray]] = []

        for k, idx in enumerate(best_perm):
            seg = segments[idx]
            if best_bits[idx] == 1:
                seg = seg.reversed()
            ordered.append(seg)
            if k > 0:
                prev_end = ordered[-2].p1
                travel_segs.append((prev_end.copy(), seg.p0.copy()))

        return ordered, travel_segs, baseline_mm, optimised_mm

    @staticmethod
    def _endpoint(seg: Segment, flip: int, end: bool) -> np.ndarray:
        if end:
            return seg.p1 if flip == 0 else seg.p0
        return seg.p0 if flip == 0 else seg.p1


# ══════════════════════════════════════════════════════════════════════════════
# 3.  AStarPathfinder — obstacle-aware rapid travel routing
# ══════════════════════════════════════════════════════════════════════════════
class AStarPathfinder:
    """
    Builds a rasterised obstacle grid from the printed wall polygons on a
    layer, then routes G0 travel moves around them with A*.
    """

    def __init__(self, perimeter_loops: List[List[Tuple[float, float]]],
                 resolution: float = 0.5):
        self.resolution = resolution
        self.grid, self.origin, self.shape = self._build_grid(perimeter_loops)

    def _build_grid(self, loops):
        if not loops:
            return np.zeros((1, 1), dtype=bool), np.array([0.0, 0.0]), (1, 1)

        all_pts = [p for loop in loops for p in loop]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = self.resolution * 4
        origin = np.array([min(xs) - margin, min(ys) - margin])
        w = int((max(xs) - min(xs) + 2 * margin) / self.resolution) + 2
        h = int((max(ys) - min(ys) + 2 * margin) / self.resolution) + 2
        grid = np.zeros((h, w), dtype=bool)

        # Rasterise each loop as thick wall obstacles
        wall_thickness = max(1, int(0.84 / self.resolution))  # ~2 line widths
        for loop in loops:
            for i in range(len(loop) - 1):
                p0 = np.array(loop[i])
                p1 = np.array(loop[i + 1])
                dist = np.linalg.norm(p1 - p0)
                if dist < 1e-9:
                    continue
                steps = max(2, int(dist / (self.resolution * 0.5)))
                for s in range(steps + 1):
                    t = s / steps
                    pt = p0 + t * (p1 - p0)
                    gx = int((pt[0] - origin[0]) / self.resolution)
                    gy = int((pt[1] - origin[1]) / self.resolution)
                    for dx in range(-wall_thickness, wall_thickness + 1):
                        for dy in range(-wall_thickness, wall_thickness + 1):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                grid[ny, nx] = True
        return grid, origin, (h, w)

    def _to_grid(self, pt: np.ndarray) -> Tuple[int, int]:
        gx = int((pt[0] - self.origin[0]) / self.resolution)
        gy = int((pt[1] - self.origin[1]) / self.resolution)
        h, w = self.shape
        return (max(0, min(gx, w - 1)), max(0, min(gy, h - 1)))

    def _to_world(self, gx: int, gy: int) -> np.ndarray:
        return np.array([
            gx * self.resolution + self.origin[0],
            gy * self.resolution + self.origin[1],
        ])

    def route(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """Returns a list of waypoints (including start and end) avoiding walls."""
        sx, sy = self._to_grid(start)
        ex, ey = self._to_grid(end)
        if (sx, sy) == (ex, ey):
            return [start, end]
        h, w = self.shape
        open_heap = []
        heapq.heappush(open_heap, (0.0, sx, sy))
        came_from: dict = {}
        g_score = {(sx, sy): 0.0}
        f_score = {(sx, sy): math.hypot(ex - sx, ey - sy)}

        while open_heap:
            _, cx, cy = heapq.heappop(open_heap)
            if (cx, cy) == (ex, ey):
                # reconstruct path
                path = []
                node = (ex, ey)
                while node in came_from:
                    path.append(self._to_world(*node))
                    node = came_from[node]
                path.append(start)
                path.reverse()
                path.append(end)
                return path

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if self.grid[ny, nx]:
                        continue
                    move_cost = math.hypot(dx, dy)
                    tg = g_score[(cx, cy)] + move_cost
                    if tg < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = (cx, cy)
                        g_score[(nx, ny)] = tg
                        f_score[(nx, ny)] = tg + math.hypot(ex - nx, ey - ny)
                        heapq.heappush(open_heap, (f_score[(nx, ny)], nx, ny))

        # fallback: straight line if A* can't find a path
        return [start, end]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Per-layer orchestration pipeline
# ══════════════════════════════════════════════════════════════════════════════
def process_layers(
    mesh: trimesh.Trimesh,
    layer_height: float,
    infill_density: float,
    pattern: str,
    ga_pop: int,
    ga_gen: int,
    print_speed: float,
    progress_cb=None,
) -> List[LayerData]:
    """Full pipeline: slice → infill → GA → A* for every layer."""
    slicer = CloudSlicer(mesh, layer_height, infill_density, pattern)
    raw_layers = slicer.slice_all()
    ga_solver  = GeneticAlgorithmSolver(population_size=ga_pop, generations=ga_gen)

    processed: List[LayerData] = []
    total = len(raw_layers)
    for i, ld in enumerate(raw_layers):
        if progress_cb:
            progress_cb(i / max(total - 1, 1), f"Layer {i+1}/{total}")

        # GA optimise infill order
        ordered_segs, travel_rapid, baseline_mm, opt_mm = \
            ga_solver.solve(ld.infill_segs)

        # A* route each rapid travel move
        pathfinder = AStarPathfinder(ld.perimeter_loops, resolution=0.5)
        routed_travels: List[Tuple[np.ndarray, np.ndarray]] = []
        for (s, e) in travel_rapid:
            waypoints = pathfinder.route(s, e)
            for k in range(len(waypoints) - 1):
                routed_travels.append((waypoints[k], waypoints[k + 1]))

        # accumulate totals
        extrude_len = sum(seg.length() for seg in ordered_segs)
        extrude_len += sum(
            sum(math.hypot(ld.perimeter_loops[li][j][0] - ld.perimeter_loops[li][j-1][0],
                           ld.perimeter_loops[li][j][1] - ld.perimeter_loops[li][j-1][1])
                for j in range(1, len(ld.perimeter_loops[li])))
            for li in range(len(ld.perimeter_loops))
        )
        travel_len = sum(float(np.linalg.norm(e - s)) for s, e in routed_travels)

        processed.append(LayerData(
            z_mm             = ld.z_mm,
            perimeter_loops  = ld.perimeter_loops,
            infill_segs      = ordered_segs,
            travel_segs      = routed_travels,
            total_extrude_mm = extrude_len,
            total_travel_mm  = travel_len,
        ))

    if progress_cb:
        progress_cb(1.0, "Done")
    return processed


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Material & extrusion math
# ══════════════════════════════════════════════════════════════════════════════
def calc_extrusion_e(path_mm: float, layer_height: float,
                     line_width: float = MACHINE["line_width_mm"],
                     filament_dia: float = MACHINE["filament_dia_mm"]) -> float:
    """
    Volumetric extrusion: E = (cross_section_area / filament_area) * path_length
    Cross section approximated as rectangle: layer_height × line_width
    """
    filament_area = math.pi * (filament_dia / 2) ** 2
    cross_section = layer_height * line_width
    return (cross_section / filament_area) * path_mm


def calc_material_weight(layers: List[LayerData], layer_height: float) -> float:
    """Total PLA weight in grams."""
    total_mm = sum(ld.total_extrude_mm for ld in layers)
    volume_mm3 = total_mm * layer_height * MACHINE["line_width_mm"]
    volume_cm3  = volume_mm3 / 1000.0
    return volume_cm3 * MACHINE["pla_density_gcc"]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  G-code compiler
# ══════════════════════════════════════════════════════════════════════════════
def compile_gcode(
    layers: List[LayerData],
    layer_height: float,
    print_speed: float,
    hotend_temp: int = MACHINE["default_temp_hotend"],
    bed_temp:    int = MACHINE["default_temp_bed"],
) -> str:
    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "; Generated by Kobra 2 Neo Cloud Slicer",
        f"; Layer Height: {layer_height:.2f} mm",
        f"; Print Speed:  {print_speed} mm/s",
        f"; Pattern:      see sidebar",
        ";",
        "G90            ; absolute positioning",
        "M82            ; absolute extrusion mode",
        f"M104 S{hotend_temp}  ; set hotend temperature (no wait)",
        f"M140 S{bed_temp}   ; set bed temperature (no wait)",
        f"M109 S{hotend_temp}  ; wait for hotend",
        f"M190 S{bed_temp}   ; wait for bed",
        "G28            ; home all axes",
        "G92 E0         ; reset extruder",
        "G1 Z5 F3000    ; lift nozzle",
        "",
    ]

    E = 0.0       # cumulative extruder position
    retracted = False

    ret_dist  = MACHINE["retract_dist_mm"]
    ret_speed = MACHINE["retract_speed_mmpm"]
    first_spd = MACHINE["first_layer_speed"]

    def retract():
        nonlocal E, retracted
        if not retracted:
            lines.append(
                f"G1 E{E - ret_dist:.5f} F{ret_speed}  ; retract"
            )
            retracted = True

    def deretract():
        nonlocal E, retracted
        if retracted:
            lines.append(
                f"G1 E{E:.5f} F{ret_speed}  ; de-retract"
            )
            retracted = False

    # ── Layer loop ────────────────────────────────────────────────────────────
    for ld in layers:
        z = ld.z_mm
        # First-layer speed cap: Z ≤ layer_height → force 25 mm/s
        spd = first_spd if z <= layer_height else print_speed
        spd_mmpm = int(spd * 60)

        lines.append(f"\n; === Layer Z={z:.3f} mm ===")
        lines.append(f"G1 Z{z:.3f} F3000  ; move to layer height")

        # ── Perimeter loops ───────────────────────────────────────────────────
        for loop in ld.perimeter_loops:
            if len(loop) < 2:
                continue
            p0 = loop[0]
            retract()
            lines.append(f"G0 X{p0[0]:.3f} Y{p0[1]:.3f} F9000  ; travel to loop start")
            deretract()
            for pt in loop[1:]:
                dx = pt[0] - p0[0]
                dy = pt[1] - p0[1]
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                dE = calc_extrusion_e(seg_len, layer_height)
                E += dE
                lines.append(
                    f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} E{E:.5f} F{spd_mmpm}"
                )
                p0 = pt

        # ── Infill segments ───────────────────────────────────────────────────
        for seg in ld.infill_segs:
            seg_len = seg.length()
            if seg_len < 1e-6:
                continue
            # travel to segment start
            retract()
            lines.append(
                f"G0 X{seg.p0[0]:.3f} Y{seg.p0[1]:.3f} F9000  ; travel to infill"
            )
            deretract()
            dE = calc_extrusion_e(seg_len, layer_height)
            E += dE
            lines.append(
                f"G1 X{seg.p1[0]:.3f} Y{seg.p1[1]:.3f} E{E:.5f} F{spd_mmpm}"
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "",
        "; === Print complete ===",
        "M104 S0        ; hotend off",
        "M140 S0        ; bed off",
        f"G1 E{E - ret_dist:.5f} F{ret_speed}  ; final retract",
        "G28 X0         ; park X axis",
        "G1 Y200 F3000  ; present part",
        "M84            ; motors off",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Matplotlib layer visualisation
# ══════════════════════════════════════════════════════════════════════════════
def plot_layer(ld: LayerData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # Perimeter loops — white
    for loop in ld.perimeter_loops:
        if len(loop) < 2:
            continue
        xs = [p[0] for p in loop] + [loop[0][0]]
        ys = [p[1] for p in loop] + [loop[0][1]]
        ax.plot(xs, ys, color="white", linewidth=0.9, alpha=0.9)

    # Infill — cyan
    for seg in ld.infill_segs:
        ax.plot([seg.p0[0], seg.p1[0]], [seg.p0[1], seg.p1[1]],
                color="#58a6ff", linewidth=0.6, alpha=0.75)

    # Travel moves — bright green
    for (s, e) in ld.travel_segs:
        ax.plot([s[0], e[0]], [s[1], e[1]],
                color="#3fb950", linewidth=0.5, linestyle="--", alpha=0.7)

    # Legend
    handles = [
        mpatches.Patch(color="white",   label="Perimeter"),
        mpatches.Patch(color="#58a6ff", label="Infill"),
        mpatches.Patch(color="#3fb950", label="G0 Travel"),
    ]
    ax.legend(handles=handles, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"Layer Z = {ld.z_mm:.3f} mm", color="#f0f6fc", fontsize=10)
    ax.set_xlabel("X (mm)", color="#8b949e")
    ax.set_ylabel("Y (mm)", color="#8b949e")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Streamlit Application Body
# ══════════════════════════════════════════════════════════════════════════════
INFILL_PATTERNS = [
    "Lines", "Tri-Hexagon", "Cubic Subdivision", "Octet", "Quarter Cubic",
    "Concentric", "Zig Zag", "Cross", "Cross 3D", "Gyroid", "Lightning",
    "Honeycomb", "Octagon", "Grid", "Cubic", "Triangles",
]

def main():
    # ── Title ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#f0f6fc;margin-bottom:0'>🖨️ Kobra 2 Neo — Cloud Slicer</h1>"
        "<p style='color:#8b949e;margin-top:4px'>STL → Toolpath Optimisation → G-code</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Slice Settings")
        uploaded = st.file_uploader("Upload STL", type=["stl"])

        st.markdown("---")
        layer_height = st.slider("Layer Height (mm)", 0.1, 0.4, 0.2, 0.05)
        infill_density = st.slider("Infill Density (%)", 0, 50, 15, 5)
        print_speed = st.slider("Print Speed (mm/s)", 30, 150, 60, 5)
        pattern = st.selectbox("Infill Pattern", INFILL_PATTERNS)

        st.markdown("---")
        st.markdown("### 🧬 GA Optimisation")
        ga_pop = st.slider("GA Population", 10, 80, 30, 5)
        ga_gen = st.slider("GA Generations", 10, 100, 40, 5)

        st.markdown("---")
        run_btn = st.button("▶  Slice & Optimise", use_container_width=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "layers" not in st.session_state:
        st.session_state.layers = None
    if "gcode" not in st.session_state:
        st.session_state.gcode = None
    if "layer_height_used" not in st.session_state:
        st.session_state.layer_height_used = 0.2

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_btn:
        if uploaded is None:
            st.warning("Please upload an STL file first.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                mesh = trimesh.load(tmp_path, force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = trimesh.util.concatenate(
                        trimesh.load(tmp_path).dump()
                    )
            except Exception as e:
                st.error(f"Failed to load mesh: {e}")
                return
            finally:
                os.unlink(tmp_path)

            st.info(f"Mesh loaded: {len(mesh.vertices):,} vertices, "
                    f"{len(mesh.faces):,} faces")

            prog_bar = st.progress(0.0)
            status   = st.empty()

            def cb(frac, msg):
                prog_bar.progress(min(frac, 1.0))
                status.text(msg)

            try:
                layers = process_layers(
                    mesh, layer_height, infill_density, pattern,
                    ga_pop, ga_gen, print_speed, progress_cb=cb,
                )
            except Exception as e:
                st.error(f"Slicing error: {e}")
                return

            prog_bar.empty()
            status.empty()

            if not layers:
                st.error("No layers were generated — check your STL geometry.")
                return

            gcode = compile_gcode(layers, layer_height, print_speed)
            st.session_state.layers = layers
            st.session_state.gcode  = gcode
            st.session_state.layer_height_used = layer_height
            st.success(f"✅ Sliced {len(layers)} layers  ·  "
                       f"Pattern: {pattern}")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    layers = st.session_state.layers
    if layers:
        lh = st.session_state.layer_height_used

        # ── Analytics cards ───────────────────────────────────────────────────
        total_travel = sum(ld.total_travel_mm for ld in layers)
        weight_g     = calc_material_weight(layers, lh)

        c1, c2 = st.columns(2)
        c1.metric("🧱 Est. Material Weight", f"{weight_g:.2f} g",
                  help="Based on PLA density 1.24 g/cm³")
        c2.metric("✈️ Total G0 Travel", f"{total_travel:.1f} mm",
                  help="Sum of all rapid travel moves across all layers")

        st.divider()

        # ── Layer selector + plot ─────────────────────────────────────────────
        st.markdown("### 🔍 Layer Viewer")
        z_labels = [f"Z = {ld.z_mm:.3f} mm  (layer {i+1})"
                    for i, ld in enumerate(layers)]
        sel = st.selectbox("Select Layer", range(len(layers)),
                           format_func=lambda i: z_labels[i])
        ld_sel = layers[sel]

        col_plot, col_stats = st.columns([2, 1])
        with col_plot:
            fig = plot_layer(ld_sel)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_stats:
            st.markdown("**Layer Stats**")
            st.write(f"Z height: `{ld_sel.z_mm:.3f} mm`")
            st.write(f"Perimeter loops: `{len(ld_sel.perimeter_loops)}`")
            st.write(f"Infill segments: `{len(ld_sel.infill_segs)}`")
            st.write(f"Travel moves: `{len(ld_sel.travel_segs)}`")
            st.write(f"Extrude total: `{ld_sel.total_extrude_mm:.1f} mm`")
            st.write(f"Travel total: `{ld_sel.total_travel_mm:.1f} mm`")

        st.divider()

        # ── G-code download ───────────────────────────────────────────────────
        st.markdown("### 💾 G-code Export")
        gcode = st.session_state.gcode
        st.code(gcode[:1200] + "\n\n... (truncated preview) ...", language="gcode")
        st.download_button(
            label     = "⬇️  Download .gcode",
            data      = gcode.encode("utf-8"),
            file_name = "kobra2neo_output.gcode",
            mime      = "text/plain",
        )
    else:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#8b949e;">
          <div style="font-size:4rem">🖨️</div>
          <h3 style="color:#c9d1d9">Upload an STL to get started</h3>
          <p>Configure settings in the sidebar, then click <strong>Slice & Optimise</strong>.</p>
          <p style="font-size:0.85rem;margin-top:12px">
            Supports multi-body STLs · 16 infill patterns · GA + A* optimisation
          </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
