from numpy.typing import NDArray
import numpy as np
import pygame
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.AbstractController import AbstractController

V, W = 0.3, 0.6

def vectorize(angle):
    return np.array((np.cos(angle), np.sin(angle)))

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

# determine if the sector of an infinite circle defined by the first three arguments intersects the fourth argument point
def sectorPointIntersect(center, angleLeft, angleRight, point):

    u = point - center # vector to agent
    leftTurn = turn(u, vectorize(angleLeft))
    rightTurn = turn(u, vectorize(angleRight))
    
    l180 = (angleLeft - angleRight) % (np.pi * 2) < np.pi

    # if fov < 180 use between minor arc, otherwise use not between minor arc
    return rightTurn <= 0 and 0 <= leftTurn if l180 else not (leftTurn < 0 and 0 < rightTurn)

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

def sector_aabb_not_sure_if_it_works(origin, r, start_angle, end_angle) -> tuple[float, float, float, float]:
    xc, yc = origin
    # Ensure start_angle < end_angle
    if end_angle < start_angle:
        temp = start_angle
        start_angle = end_angle
        end_angle = temp

    # Base candidate points: Center, Start, and End
    angles = [start_angle, end_angle]
    
    # Base cardinal angles (0, pi/2, pi, 3pi/2)
    cardinals = np.array([0, 0.5, 1.0, 1.5]) * np.pi
    
    # Check which cardinal angles fall within [start_angle, end_angle]
    # Shift cardinal angles into the current loop range
    k = np.ceil((start_angle - cardinals) / (2 * np.pi))
    test_angles = cardinals + k * 2 * np.pi
    
    # Filter cardinal angles that fall within the sweep interval
    valid_cardinals = test_angles[(test_angles >= start_angle) & (test_angles <= end_angle)]
    
    # Combine all relevant angles into a single array
    all_angles = np.concatenate([angles, valid_cardinals])
    
    # Compute X and Y points using vectorized operations
    points_x = xc + r * np.cos(all_angles)
    points_y = yc + r * np.sin(all_angles)
    
    # Include center point (xc, yc) in the min/max calculations
    min_x = min(xc, np.min(points_x))
    max_x = max(xc, np.max(points_x))
    min_y = min(yc, np.min(points_y))
    max_y = max(yc, np.max(points_y))

    return (min_x, min_y, max_x, max_y)

def sensor_rect_intersection(rect, def_angle, sensor, world) -> bool:
    x, y, w, h = rect
    points = [(x, y), (x+w, y), (x+w,y+h), (x, y+h)]

    e_left, e_right = sensor.getSectorVectors()
    e_left, e_right = e_left[:2], e_right[:2]
    radiusSq = sensor.r * sensor.r

    # Early exit if entire aabb is enclosed in rect
    minx, miny, maxx, maxy = sensor.getAARectContainingSector(world)
    x, y, w, h = rect
    if (x < minx < maxx < x + w) and (y < miny < maxy < y + h):
        return True

    origin = sensor.position
    angle, span = def_angle + sensor.bias, sensor.theta
    for i in range(-1, len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        segment = np.array([p1, p2])
        cont = False
        for p in segmentCircleIntersectionPoints(segment, origin, sensor.r):
            if sectorPointIntersect(origin, angle + span, angle - span, p):
                return True

        if cont:
            break

        if segSegIntersect(segment, np.array([origin, origin + e_left[:2] * sensor.r])):
            return True

        p2Dist = origin - p2
        if np.dot(p2Dist, p2Dist) <= radiusSq and sectorPointIntersect(origin, angle + span, angle - span, p2):
            return True

    # return False

    # TODO: handle the case where the sector is fully enclosed within the rect
    raise NotImplementedError("handle the case where the sector is fully enclosed within the rect")


def aabb_overlap_2d(a, b) -> bool:
    return (a[0] <= b[2] and a[2] >= b[0] and
            a[1] <= b[3] and a[3] >= b[1])


class CustomEvader(AbstractController):
    def __init__(self, agent, cell_size=0.2, parent=None):
        super().__init__(agent, parent)
        assert agent is not None

        my_pos = self.agent.position
        self.goal = self.agent.world.population[0]
        self.side_len = np.linalg.norm(self.goal.position - my_pos) + self.goal.radius
        # Additional padding
        self.side_len *= 1.2
        ww, wh = np.array((self.side_len, self.side_len))
        cols, rows = np.array((ww, wh)) / cell_size
        cols, rows = int(cols), int(rows)

        center = 0.5 * self.goal.position + 0.5 * self.agent.position
        self.tl = center - np.array((self.side_len * 0.5, self.side_len * 0.5))

        self.cell_size = np.array((cell_size, cell_size))
        self.cells = np.ones((rows, cols), dtype=np.uint32)
        self.color_grid = np.zeros_like(self.cells)
        self.walls: list[tuple[int, int]] = []
        self.occupied_color = (255, 0, 0)
        self.empty_color = (100, 100, 100)
        self.cell_render_fill_pct = 0.8

        self.first = True

    def get_actions(self, agent):
        self.compute_occupation()
        return (V, W)

    def draw(self, screen, offset):
        # if not self.agent.is_highlighted:
        #     return

        world = self.agent.world

        defenders = [a for a in world.population if a.team == "blue"]
        rows, cols = self.cells.shape

        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])

        for defender in defenders:
            defender.is_highlighted = True

        combined_aabb, _ = self.defender_sensor_aabb(defenders)
        minx, miny, maxx, maxy = combined_aabb
        min_coord = minx, miny
        max_coord = maxx, maxy
        # wminx, wminy, wmaxx, wmaxy = minx, miny, maxx, maxy
        wminx, wminy = min_coord * zoom + pan
        wmaxx, wmaxy = max_coord * zoom + pan
        pygame.draw.rect(screen, "#00ff00", (
            wminx, wminy, wmaxx - wminx, wmaxy - wminy
        ), width=2)

        surface_size = self.cell_size * np.array((cols, rows)) * zoom
        if self.first:
            print(surface_size)
            print("tl:", self.tl)
            self.first = False

        surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        surf_cell_size = self.cell_size * zoom
        surf_fill_size = self.cell_render_fill_pct * surf_cell_size
        padding = 0.5 * (surf_cell_size - surf_fill_size)

        for r in range(rows):
            for c in range(cols):
                # pos = self.tl + np.array((c, r)) * self.cell_size
                # world_pos = pos * zoom + pan
                # world_size = (self.cell_render_fill_pct * self.cell_size) * zoom
                # world_pos = world_pos + 0.5 * (self.cell_size * zoom - world_size)


                surf_pos = np.array((c, r)) * surf_cell_size + padding
                # color = self.empty_color
                # if self.cells[r][c] == 1:
                #     color = self.occupied_color
                # elif self.cells[r][c] == 2:
                #     color = "#ff00ff"
                # pygame.draw.rect(surface, color, (*surf_pos, *surf_fill_size))

                pygame.draw.rect(surface, int(self.color_grid[r][c]), (*surf_pos, *surf_fill_size))

                # --- RAY CASTING: DEBUG VIEW ---
                # tl = np.array((0, r)) * surf_cell_size
                # miny, maxy = tl[1], tl[1] + surf_cell_size[1]
                # ray_start_x, ray_end_x = tl[0], tl[0] + (cols-1) * surf_cell_size[0]
                # ray_h = miny + 0.5 * surf_cell_size[1]
                # pygame.draw.line(surface, "#0000ff", tl, (tl[0] + cols * surf_cell_size[0], ray_h), width=2)


        surface.set_alpha(128)
        screen.blit(surface, self.tl * zoom + pan)

    def compute_occupation(self):
        self.cells.fill(1)
        self.walls.clear()
        # self.compute_occupation_v1()
        self.compute_occupation_v2()
        self.matrix_to_color_grid()

    def compute_occupation_v2(self):
        world = self.agent.world
        defenders = [a for a in world.population if a.team == "blue"]
        combined_aabb, indiv_aabb = self.defender_sensor_aabb(defenders)

        rows, cols = self.cells.shape
        for r in range(rows):
            for c in range(cols):
                pos = self.tl + np.array((c, r)) * self.cell_size
                cell_aabb = (*pos, *(pos + self.cell_size))

                # Broad phase
                if not aabb_overlap_2d(combined_aabb, cell_aabb):
                    continue

                # Narrow phase
                self.send_rays(world, defenders, indiv_aabb, r, 0)
                self.send_rays(world, defenders, indiv_aabb, r, 0.50)
                self.send_rays(world, defenders, indiv_aabb, r, 1)
                break

    def send_rays(self, world, defenders, def_aabbs, r, ray_h_pct):
        rows, cols = self.cells.shape

        tl = self.tl + np.array((0, r)) * self.cell_size
        miny, maxy = tl[1], tl[1] + self.cell_size[1]
        ray_h_pct = np.clip(ray_h_pct, 0, 1)
        ray_y = miny + ray_h_pct * self.cell_size[1]
        ray_start_x, ray_end_x = tl[0], tl[0] + cols * self.cell_size[0]

        for i, defender in enumerate(defenders):
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

                c0, r0 = (point - self.tl) / self.cell_size
                c0, r0 = int(c0), int(r0)
                if 0 <= c0 < cols and 0 <= r0 < rows:
                    self.walls.append((c0, r0))
                    # self.cells[r0][c0] = 1

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

                c0, r0 = (point - self.tl) / self.cell_size
                c0, r0 = int(c0), int(r0)
                if 0 <= c0 < cols and 0 <= r0 < rows:
                    self.walls.append((c0, r0))
                    # self.cells[r0][c0] = 2

    def compute_occupation_v1(self):
        world = self.agent.world
        defenders = [a for a in world.population if a.team == "blue"]
        combined_aabb, indiv_aabb = self.defender_sensor_aabb(defenders)

        rows, cols = self.cells.shape
        for r in range(rows):
            for c in range(cols):
                pos = self.tl + np.array((c, r)) * self.cell_size
                cell_aabb = (*pos, *(pos + self.cell_size))

                if not aabb_overlap_2d(combined_aabb, cell_aabb):
                    continue

                for defender in defenders:
                    sensor = defender.sensors[1]
                    intersection = sectorRectIntersection(
                        (*pos, *self.cell_size),
                        defender.position, sensor.r,
                        defender.angle + sensor.bias, sensor.theta
                    )
                    self.cells[r][c] = int(intersection)
                    if intersection:
                        break

    def defender_sensor_aabb(self, defenders) -> tuple[NDArray, NDArray]:
        world = self.agent.world

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

    def matrix_to_color_grid(self):
        for wn in self.walls:
            self.cells[wn[1]][wn[0]] = 0

        grid = Grid(matrix=self.cells)
        end_pt, start_pt = self.point_to_index(self.agent.position), self.point_to_index(self.goal.position)
        # start_pt, end_pt = self.point_to_index(self.agent.position), self.point_to_index(self.goal.position)
        assert start_pt is not None
        assert end_pt is not None

        start = grid.node(*start_pt)
        end = grid.node(*end_pt)

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        self.color_grid.fill(0xff777777)
        for pn in path:
            self.color_grid[pn.y][pn.x] = 0xff00ffff

        for wc in self.walls:
            self.color_grid[wc[1]][wc[0]] = 0xff333333

        self.color_grid[start.y][start.x] = 0xff00ff00
        self.color_grid[end.y][end.x] = 0xffff0000

    def point_to_index(self, point) -> tuple[int, int] | None:
        rows, cols = self.cells.shape
        c0, r0 = (point - self.tl) / self.cell_size
        if 0 <= c0 < cols and 0 <= r0 < rows:
            return (int(c0), int(r0))
        else:
            return None
