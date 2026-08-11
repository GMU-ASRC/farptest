from numpy.typing import NDArray

import numpy as np
import pygame
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from scipy.ndimage import gaussian_filter


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
    def __init__(self,
        rect: tuple[float, float, float, float],
        cell_size=0.2, decay_rate=0.8
    ) -> None:
        self.rect = rect
        x, y, w, h = self.rect
        side_len = max(w, h)

        ww, wh = np.array((side_len, side_len))
        cols, rows = np.array((ww, wh)) / cell_size
        cols, rows = int(cols), int(rows)

        self.tl = np.asarray([x, y])
        self.cell_size = np.ones((2,)) * cell_size
        self.walls = np.zeros((rows, cols), dtype=np.int_)
        self.cell_wts = np.zeros_like(self.walls)
        self.wall_range: list[tuple[int, int, int]] = []
        self.path = []
        self.decay_rate = decay_rate

        self.occupied_color = (255, 0, 0)
        self.empty_color = (100, 100, 100)
        self.cell_render_fill_pct = 0.8
        self.color_grid = np.zeros_like(self.cell_wts)
        self.colors = {
            # abgr
            "path" : pygame.Color("#ff00ffff"),
            "open" : pygame.Color("#ff333333"),
            "wall" : pygame.Color("#ffff0000"),
            "start": pygame.Color("#ff00ff00"),
            "end"  : pygame.Color("#ffff0000"),
        }

    def point_to_index(self, point) -> tuple[int, int] | None:
        rows, cols = self.cell_wts.shape
        c0, r0 = (point - self.tl) / self.cell_size
        if 0 <= c0 < cols and 0 <= r0 < rows:
            return (int(c0), int(r0))
        else:
            return None

    def index_to_point(self, r: int, c: int) -> NDArray | None:
        rows, cols = self.cell_wts.shape
        if 0 <= c < cols and 0 <= r < rows:
            return self.tl + np.array((c, r)) * self.cell_size
        else:
            return None

    def update(self, world, defenders):
        self.wall_range.clear()
        self.compute_occupation(world, defenders)
        
        self.walls.fill(0)
        for wr, wcs, wce in self.wall_range:
            self.walls[wr, wcs:wce+1] = 1.

        curr_wts = gaussian_filter(self.walls.astype(np.float64), sigma=1, mode="constant")
        self.cell_wts = curr_wts + self.decay_rate * self.cell_wts
        self.cell_wts /= self.cell_wts.max()

    def draw(self, screen, zoom, pan, goal, opacity=0.5):
        self.color_grid.fill(self.colors["open"])
        rows, cols = self.cell_wts.shape
        for r in range(rows):
            for c in range(cols):
                self.color_grid[r][c] = self.colors["open"].lerp(
                    self.colors["wall"], self.cell_wts[r][c])

        for pn in self.path:
            self.color_grid[pn.y][pn.x] = self.colors["path"]

        for wr, wcs, wce in self.wall_range:
            self.color_grid[wr, wcs:wce+1].fill(self.colors["wall"])


        rows, cols = self.cell_wts.shape
        surface_size = self.cell_size * np.array((cols, rows)) * zoom
        surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        surf_cell_size = self.cell_size * zoom
        surf_fill_size = self.cell_render_fill_pct * surf_cell_size
        padding = 0.5 * (surf_cell_size - surf_fill_size)

        for r in range(rows):
            for c in range(cols):
                surf_pos = np.array((c, r)) * surf_cell_size + padding
                pygame.draw.rect(surface, int(self.color_grid[r][c]), (*surf_pos, *surf_fill_size))

        center, radius = goal.position, goal.radius * 1.2
        vec = self.goal_heatmap_vector(center, radius)
        vec = radius * vec / np.linalg.norm(vec)
        pygame.draw.circle(screen, "#aaff00", center * zoom + pan, radius * zoom, width=2)
        pygame.draw.line(screen, "#aaff00", center * zoom + pan, (center + vec) * zoom + pan, width=2)

        surface.set_alpha(int(opacity * 255.))
        screen.blit(surface, self.tl * zoom + pan)

    def goal_heatmap_vector(self, goal_center, goal_radius):
        center, radius = np.asarray(goal_center), goal_radius
        goal_aabb = (*(center - radius), *(center + radius))

        rows, cols = self.cell_wts.shape
        sum_vec = np.zeros((2,))
        for r in range(rows):
            for c in range(cols):
                pos = self.index_to_point(r, c)
                assert pos is not None
                cell_aabb = (*pos, *(pos + self.cell_size))
                if not aabb_overlap_2d(goal_aabb, cell_aabb):
                    continue

                cell_center = pos + 0.5 * self.cell_size
                dist = np.linalg.norm(cell_center - center)
                if dist <= radius:
                    mag = self.cell_wts[r][c]
                    sum_vec += (cell_center - center) / dist * mag

        return sum_vec

    def compute_occupation(self, world, defenders, ray_heights=[0.2, 0.8]):
        combined_aabb, indiv_aabb = self.defender_sensor_aabb(world, defenders)

        rows, cols = self.cell_wts.shape
        for r in range(rows):
            for c in range(cols):
                # pos = self.tl + np.array((c, r)) * self.cell_size
                pos = self.index_to_point(r, c)
                cell_aabb = (*pos, *(pos + self.cell_size))

                # Broad phase
                if not aabb_overlap_2d(combined_aabb, cell_aabb):
                    continue

                # Narrow phase
                for ray_h in ray_heights:
                    self.send_rays(world, defenders, indiv_aabb, r, ray_h)

                break

    def send_rays(self, world, defenders, def_aabbs, row, ray_h_pct):
        rows, cols = self.cell_wts.shape

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

