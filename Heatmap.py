from numpy.typing import NDArray

import numpy as np
import pygame
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def project(a, b):
    return b * (np.dot(a, b) / np.dot(b, b))

def turn(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]

def colinearPointSegmentIntersect(seg: np.ndarray, point: np.ndarray, segsq = None):
    if not segsq:
        segsq = np.dot(seg, seg)
    sdp = np.dot(seg, point)
    return 0 <= sdp and sdp < segsq

def segSegIntersect(seg1: np.ndarray, seg2: np.ndarray):
    s1l = seg1[1] - seg1[0]
    s2l = seg2[1] - seg2[0]
    t12_0 = turn(s1l, seg2[0] - seg1[0])
    t12_1 = turn(s1l, seg2[1] - seg1[0])
    t21_0 = turn(s2l, seg1[0] - seg2[0])
    t21_1 = turn(s2l, seg1[1] - seg2[0])
    return ((((t12_0 < 0 and 0 < t12_1) or (t12_1 < 0 and 0 < t12_0)) and
        ((t21_0 < 0 and 0 < t21_1) or (t21_1 < 0 and 0 < t21_0))) or
        (t12_0 == 0 and colinearPointSegmentIntersect(s1l, seg2[0] - seg1[0])) or 
        (t12_1 == 0 and colinearPointSegmentIntersect(s1l, seg2[1] - seg1[0])) or 
        (t21_0 == 0 and colinearPointSegmentIntersect(s2l, seg1[0] - seg2[0])) or 
        (t21_1 == 0 and colinearPointSegmentIntersect(s2l, seg1[1] - seg2[0])))

def lineCircleIntersectionPoints(line: np.ndarray, center: np.ndarray, radius):
    unitLine = line / np.linalg.norm(line)
    projectCenterToLine = project(center, line)
    clDiffVec = center - projectCenterToLine
    clDiffVecMagsq = np.dot(clDiffVec, clDiffVec)
    if radius**2 < clDiffVecMagsq:
        return []
    midDist = np.sqrt(radius**2 - clDiffVecMagsq)
    return [projectCenterToLine + midDist * unitLine, projectCenterToLine - midDist * unitLine]

def segmentCircleIntersectionPoints(segPs: np.ndarray, center: np.ndarray, radius):
    origin = segPs[0]
    line = segPs[1] - origin
    intersectionPoints = lineCircleIntersectionPoints(line, center - origin, radius)
    lineSq = np.dot(line, line)
    onSegmentIntersectionPoints = [p for p in intersectionPoints if colinearPointSegmentIntersect(line, p, lineSq)]
    globalIntersectionPoints = [p + origin for p in onSegmentIntersectionPoints]
    return globalIntersectionPoints

def seg_seg_intersection_point(seg_a: NDArray, seg_b: NDArray):
    if not segSegIntersect(seg_a, seg_b):
        return None

    start_a, end_a = seg_a
    start_b, end_b = seg_b
    m_a, m_b = end_a - start_a, end_b - start_b

    # if abs(np.linalg.det(mat)) <= 1e-5:
    det = m_a[0] * (-m_b[1]) - m_a[1] * (-m_b[0])
    if abs(det) <= 1e-5:
        return None

    mat = [ [m_a[0], -m_b[0]], [m_a[1], -m_b[1]] ]
    # solutions
    s1, s2 = np.linalg.inv(mat) @ np.array(start_b - start_a)
    return start_a + s1 * (end_a - start_a)

def aabb_overlap_2d(a, b) -> bool:
    return (a[0] <= b[2] and a[2] >= b[0] and
            a[1] <= b[3] and a[3] >= b[1])


class Heatmap:
    def __init__(self, rect: tuple[float, float, float, float], cell_size=0.2) -> None:
        self.rect = rect
        x, y, w, h = self.rect
        side_len = max(w, h)

        ww, wh = np.array((side_len, side_len))
        cols, rows = np.array((ww, wh)) / cell_size
        cols, rows = int(cols), int(rows)

        self.tl = np.asarray([x, y])
        self.cell_size = np.ones((2,)) * cell_size
        self.cells = np.ones((rows, cols), dtype=np.uint32)
        self.wall_range: list[tuple[int, int, int]] = []
        self.path = []
        self.occupied_color = (255, 0, 0)
        self.empty_color = (100, 100, 100)
        self.cell_render_fill_pct = 0.8

        self.color_grid = np.zeros_like(self.cells)
        self.colors = {
            "path" : pygame.Color(0xff, 0x00, 0xff, 0xff),
            "open" : pygame.Color(0xff, 0x33, 0x33, 0x33),
            "wall" : pygame.Color(0xff, 0xff, 0x00, 0x00),
            "start": pygame.Color(0xff, 0x00, 0xff, 0x00),
            "end"  : pygame.Color(0xff, 0xff, 0x00, 0x00)
        }

    def point_to_index(self, point) -> tuple[int, int] | None:
        rows, cols = self.cells.shape
        c0, r0 = (point - self.tl) / self.cell_size
        if 0 <= c0 < cols and 0 <= r0 < rows:
            return (int(c0), int(r0))
        else:
            return None

    def index_to_point(self, index: tuple[int, int]) -> NDArray | None:
        rows, cols = self.cells.shape
        c0, r0 = index
        if 0 <= c0 < cols and 0 <= r0 < rows:
            return self.tl + np.array((c0, r0)) * self.cell_size
        else:
            return None

    def update(self, world, defenders):
        self.cells.fill(1)
        self.wall_range.clear()
        self.compute_occupation(world, defenders)
        # self.color_grid.fill(self.colors["open"])
        self.matrix_to_color_grid()

    def draw(self, screen, zoom, pan, opacity=0.5):

        rows, cols = self.cells.shape
        surface_size = self.cell_size * np.array((cols, rows)) * zoom
        surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        surf_cell_size = self.cell_size * zoom
        surf_fill_size = self.cell_render_fill_pct * surf_cell_size
        padding = 0.5 * (surf_cell_size - surf_fill_size)

        for r in range(rows):
            for c in range(cols):
                surf_pos = np.array((c, r)) * surf_cell_size + padding
                pygame.draw.rect(surface, int(self.color_grid[r][c]), (*surf_pos, *surf_fill_size))

        surface.set_alpha(int(opacity * 255.))
        screen.blit(surface, self.tl * zoom + pan)

    def matrix_to_color_grid(self, start_pos=None, end_pos=None, reverse=True):
        # Apply wall weights
        l0, l1, l2 = 1.2, 1, 0
        weight = np.array([
            [l2, l2, l2, l2, l2],
            [l2, l1, l1, l1, l2],
            [l2, l1, l0, l1, l2],
            [l2, l1, l1, l1, l2],
            [l2, l2, l2, l2, l2],
        ], dtype=self.cells.dtype)

        padded = np.pad(self.cells, pad_width=2, mode="constant", constant_values=0)
        for wr, wcs, wce in self.wall_range:
            for wc in range(wcs, wce+1):
                padded[wr:wr+5, wc:wc+5] += weight

        # Remove padding, return back to original size
        self.cells = padded[2:-2, 2:-2]
        norm_grid = (self.cells - self.cells.min()) / (self.cells.max() - self.cells.min())

        for wr, wcs, wce in self.wall_range:
            self.cells[wr, wcs:wce+1].fill(0)

        should_pathfind = not (start_pos is None or end_pos is None)
        if should_pathfind:
            grid = Grid(matrix=self.cells)
            if reverse:
                start_pt, end_pt = self.point_to_index(end_pos), self.point_to_index(start_pos)
            else:
                start_pt, end_pt = self.point_to_index(start_pos), self.point_to_index(end_pos)

            assert start_pt is not None
            assert end_pt is not None

            start = grid.node(*start_pt)
            end = grid.node(*end_pt)

            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            self.path, runs = finder.find_path(start, end, grid)

        rows, cols = self.cells.shape
        for r in range(rows):
            for c in range(cols):
                self.color_grid[r][c] = self.colors["open"].lerp(
                    self.colors["wall"], norm_grid[r][c])

        for pn in self.path:
            self.color_grid[pn.y][pn.x] = self.colors["path"]

        for wr, wcs, wce in self.wall_range:
            self.color_grid[wr, wcs:wce+1].fill(self.colors["wall"])

        if should_pathfind:
            self.color_grid[start.y][start.x] = self.colors["start"]
            self.color_grid[end.y][end.x] = self.colors["end"]

    def compute_occupation(self, world, defenders, ray_heights=[0.2, 0.8]):
        combined_aabb, indiv_aabb = self.defender_sensor_aabb(world, defenders)

        rows, cols = self.cells.shape
        for r in range(rows):
            for c in range(cols):
                pos = self.tl + np.array((c, r)) * self.cell_size
                cell_aabb = (*pos, *(pos + self.cell_size))

                # Broad phase
                if not aabb_overlap_2d(combined_aabb, cell_aabb):
                    continue

                # Narrow phase
                for ray_h in ray_heights:
                    self.send_rays(world, defenders, indiv_aabb, r, ray_h)

                break

    def send_rays(self, world, defenders, def_aabbs, row, ray_h_pct):
        rows, cols = self.cells.shape

        tl = self.tl + np.array((0, row)) * self.cell_size
        miny, maxy = tl[1], tl[1] + self.cell_size[1]
        ray_h_pct = np.clip(ray_h_pct, 0, 1)
        ray_y = miny + ray_h_pct * self.cell_size[1]
        ray_start_x, ray_end_x = tl[0], tl[0] + cols * self.cell_size[0]

        _range = []
        for i, defender in enumerate(defenders):
            _range.clear()

            sensor = defender.sensors[1]
            _, miny, _, maxy = def_aabbs[i]

            if not (miny <= ray_y <= maxy):
                continue

            ray_seg = np.array([(ray_start_x, ray_y), (ray_end_x, ray_y)])
            points = segmentCircleIntersectionPoints(
                segPs=ray_seg,
                center=sensor.position,
                radius=sensor.r
            )

            origin = sensor.position
            e_left, e_right = sensor.getSectorVectors()
            e_left, e_right = e_left[:2], e_right[:2]
            for point in points:
                if not (turn(point - origin, e_right) <= 0 and 0 <= turn(point - origin, e_left)):
                    continue

                c0 = int(((point - self.tl) / self.cell_size)[0])
                if 0 <= c0 < cols:
                    _range.append(c0)

            left_int = seg_seg_intersection_point(
                seg_a=ray_seg,
                seg_b=np.array([origin, origin + sensor.r * e_left])
            )
            right_int = seg_seg_intersection_point(
                seg_a=ray_seg,
                seg_b=np.array([origin, origin + sensor.r * e_right])
            )
            
            for point in [left_int, right_int]:
                if point is None:
                    continue

                c0 = int(((point - self.tl) / self.cell_size)[0])
                if 0 <= c0 < cols:
                    _range.append(c0)

            if len(_range) > 0:
                self.wall_range.append((row, np.min(_range), np.max(_range)))


    def defender_sensor_aabb(self, world, defenders) -> tuple[NDArray, NDArray]:
        sens_aabbs = np.zeros((len(defenders), 4))
        for i, defender in enumerate(defenders):
            sensor = defender.sensors[1]
            angle = defender.angle + sensor.bias
            aabb = sensor.getAARectContainingSector(
                world, aabb_padding=0.25 * self.cell_size[0])
            sens_aabbs[i] = aabb

        minx, miny = np.min(sens_aabbs.T[:2], axis=1)
        maxx, maxy = np.max(sens_aabbs.T[2:], axis=1)
        return (np.array((minx, miny, maxx, maxy)), sens_aabbs)

