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


class CustomEvader(AbstractController):
    def __init__(self, agent, cell_size=0.2, parent=None):
        super().__init__(agent, parent)
        assert agent is not None

        my_pos = self.agent.position
        self.goal = self.agent.world.population[0]
        self.side_len = np.linalg.norm(self.goal.position - my_pos) + self.goal.radius
        self.side_len *= 1.2
        ww, wh = np.ones((2,)) * self.side_len
        cols, rows = np.array((ww, wh)) / cell_size
        cols, rows = int(cols), int(rows)

        center = 0.5 * self.goal.position + 0.5 * self.agent.position
        self.tl = center - np.ones((2,)) * self.side_len * 0.5

        self.cell_size = np.ones((2,)) * cell_size
        self.grid = np.zeros((rows, cols), dtype=np.float32)
        self.occupied_color = (255, 0, 0)
        self.empty_color = (100, 100, 100)
        self.cell_render_fill_pct = 0.3

        self.first = True

    def get_actions(self, agent):
        return (V, W)

    def draw(self, screen, offset):
        world = self.agent.world
        rows, cols = self.grid.shape


        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
        tl_world = self.tl * zoom + pan
        # abc = (self.cell_render_fill_pct * cell_size) * zoom * 0.5
        # pygame.draw.circle(screen, self.empty_color, tl_world, float(0.1 * zoom), width=2)
        # pygame.draw.circle(screen, self.empty_color, center * zoom + pan + abc, float(0.1 * zoom), width=2)
        # pygame.draw.circle(screen, self.empty_color, center * zoom + pan + abc, radius * zoom, width=2)

        x = self.cell_size * np.array((rows, cols)) * zoom + pan
        if self.first:
            print(x)
            self.first = False


        surface = pygame.Surface(x, pygame.SRCALPHA)
        for r in range(rows):
            for c in range(cols):
                x, y = (np.array((r, c)) * self.cell_size)
                x_world, y_world = np.array((x, y)) * zoom + tl_world
                w, h = (self.cell_render_fill_pct * self.cell_size) * zoom
                pos = np.array((x, y)) + 0.5 * np.array((w, h))


                color = self.occupied_color if self.grid[r][c] else self.empty_color
                pygame.draw.rect(surface, color, (x_world, y_world, w, h))
                # pygame.draw.rect(surface, color, (x_world, y_world, w, h), width=2)

        surface.set_alpha(128)
        screen.blit(surface)

        # screen_copy = screen.copy()
        # screen.fill((0, 0, 0))
        # screen.blit(surface)
        # screen.blit(screen_copy)

    # def intersect(self):
    #     x, y, s = self.rect
    #     points = [(x, y), (x+s, y), (x+s,y+s), (x, y+s)]
    #     for i in range(-1, len(points) - 1):
    #         p1 = points[i]
    #         p2 = points[i + 1]
    #         segment = np.array([p1, p2])
    #         cont = False
    #         for p in segmentCircleIntersectionPoints(segment, self.agent.position, self.r):
    #             if sectorPointIntersect(sensor_origin, angle + self.theta, angle - self.theta, p):
    #                 if self.senseObject(obj, world):
    #                     return
    #                 else:
    #                     cont = True
    #                     break
    #         if cont:
    #             break
    #         if segSegIntersect(segment, np.array([sensor_origin, sensor_origin + e_left[:2] * self.r])): # or segSegIntersect(segment, np.array([sensor_origin, sensor_origin + e_right[:2] * self.r])):
    #             if self.senseObject(obj, world):
    #                 return
    #             else:
    #                 break
    #         p2Dist = sensor_origin - p2
    #         if np.dot(p2Dist, p2Dist) <= radiusSq and sectorPointIntersect(sensor_origin, angle + self.theta, angle - self.theta, p2):
    #             if self.senseObject(obj, world):
    #                 return
    #             else:
    #                 break