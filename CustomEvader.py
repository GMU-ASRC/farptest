from numpy.typing import NDArray
import numpy as np
import pygame
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.AbstractController import AbstractController

from Heatmap import Heatmap

V, W = 0.3, 0.6

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

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
    def __init__(self, agent, parent=None):
        super().__init__(agent, parent)

        self.goal = self.agent.world.population[0]
        self.dbg_center = self.agent.position + np.asarray([0.7, 0.6]) * (self.goal.position - self.agent.position)
        self.dbg_radius = np.linalg.norm(self.goal.position - self.agent.position) * 0.5
        self.dbg_radius += self.goal.radius
        self.dbg_radius *= 1.1
        self.dbg_rect = (
            *(self.dbg_center - self.dbg_radius),
            self.dbg_radius * 2,
            self.dbg_radius * 2,
        )
        self.heatmap = Heatmap(rect=self.dbg_rect)
        self.defenders = []

    def get_actions(self, agent):
        world = agent.world
        self.defenders = [a for a in world.population if a.team == "blue"]
        self.heatmap.update(world, self.defenders)
        return 0., 0.

    def draw(self, screen, offset):
        # if not self.agent.is_highlighted:
        #     return

        for d in self.defenders:
            d.is_highlighted = True

        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
        self.heatmap.draw(screen, zoom, pan)
        pygame.draw.circle(
            screen, "#ff00ff", self.dbg_center * zoom + pan, radius=self.dbg_radius * zoom, width=2)
        pygame.draw.rect(screen, "#00ffff", (
            self.dbg_rect[0] * zoom + pan[0],
            self.dbg_rect[1] * zoom + pan[1],
            self.dbg_rect[2] * zoom,
            self.dbg_rect[3] * zoom,
        ), width=2)

