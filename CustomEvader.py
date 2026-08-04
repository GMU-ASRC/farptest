from tokenize import endpats

from numpy.typing import NDArray
from swarmsim.agent.control.AbstractController import AbstractController
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor

import numpy as np
import pygame

V, W = 0.3, 0.6

def vectorize(angle):
    return np.array((np.cos(angle), np.sin(angle)))

def project(a, b):
    return b * (np.dot(a, b) / np.dot(b, b))

def turn(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]

def getSectorVectors(angle, span):
    leftBorder = vectorize(angle + span)
    rightBorder = vectorize(angle - span)
    return np.append(leftBorder, 0), np.append(rightBorder, 0)

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

def sectorAABB(origin, r, start_angle, end_angle) -> tuple[float, float, float, float]:
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

def sectorRectIntersection(rect, sensor_origin, r, angle, span) -> bool:
    # TODO: handle the case where the sector is fully enclosed within the rect

    x, y, w, h = rect
    points = [(x, y), (x+w, y), (x+w,y+h), (x, y+h)]

    e_left, e_right = getSectorVectors(angle, span)
    e_left, e_right = np.asarray(e_left[:2]), np.asarray(e_right[:2])
    radiusSq = r * r

    # Early exit if entire aabb is enclosed in rect
    minx, miny, maxx, maxy = sectorAABB(sensor_origin, r, angle + span, angle - span)
    x, y, w, h = rect
    if (x < minx < maxx < x + w) and (y < miny < maxy < y + h):
        return True

    for i in range(-1, len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        segment = np.array([p1, p2])
        cont = False
        for p in segmentCircleIntersectionPoints(segment, sensor_origin, r):
            if sectorPointIntersect(sensor_origin, angle + span, angle - span, p):
                return True

        if cont:
            break

        if segSegIntersect(segment, np.array([sensor_origin, sensor_origin + e_left[:2] * r])):
            return True

        p2Dist = sensor_origin - p2
        if np.dot(p2Dist, p2Dist) <= radiusSq and sectorPointIntersect(sensor_origin, angle + span, angle - span, p2):
            return True

    return False

def aabb_overlap_2d(a, b) -> bool:
    # return (a[0] <= b[3] and a[2] >= b[0] and
    #         a[1] <= b[3] and a[3] >= b[1])
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
        self.grid = np.zeros((rows, cols), dtype=np.float32)
        self.cells = np.zeros((rows, cols), dtype=np.bool)
        self.occupied_color = (255, 0, 0)
        self.empty_color = (100, 100, 100)
        self.cell_render_fill_pct = 0.8

        self.first = True

    def get_actions(self, agent):
        self.cells.fill(False)
        self.compute_occupation()
        return (V, W)

    def draw(self, screen, offset):
        # if not self.agent.is_highlighted:
        #     return

        world = self.agent.world

        defenders = [a for a in world.population if a.team == "blue"]
        rows, cols = self.grid.shape

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
                color = self.occupied_color if self.cells[r][c] else self.empty_color
                pygame.draw.rect(surface, color, (*surf_pos, *surf_fill_size))

                # mx, my, Mx, My = np.array((*pos, *(pos + self.cell_size)))
                # mx = mx * zoom + pan[0]
                # my = my * zoom + pan[1]
                # Mx = Mx * zoom + pan[0]
                # My = My * zoom + pan[1]
                # pygame.draw.rect(screen, "#0000ff", (mx, my, Mx - mx, My - my), width=1)

        surface.set_alpha(128)
        screen.blit(surface, self.tl * zoom + pan)

    def compute_occupation(self):
        world = self.agent.world
        defenders = [a for a in world.population if a.team == "blue"]
        combined_aabb, indiv_aabb = self.defender_sensor_aabb(defenders)

        rows, cols = self.grid.shape
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
                        defender.angle + sensor.angle, sensor.theta
                    )
                    self.cells[r][c] = intersection
                    if intersection:
                        break

    def defender_sensor_aabb(self, defenders) -> tuple[NDArray, NDArray]:
        world = self.agent.world
        def_n = len(defenders)

        sens_aabbs = np.zeros((def_n, 4))
        for i, defender in enumerate(defenders):
            sensor = defender.sensors[1]
            angle = defender.angle + sensor.angle
            aabb = sectorAABB(
                origin=defender.position,
                r=sensor.r,
                start_angle=angle + sensor.theta,
                end_angle=angle - sensor.theta,
            )
            sens_aabbs[i] = aabb

        minx, miny = np.min(sens_aabbs.T[:2], axis=1)
        maxx, maxy = np.max(sens_aabbs.T[2:], axis=1)
        return (np.array((minx, miny, maxx, maxy)), sens_aabbs)
