from swarmsim.world.RectangularWorld import RectangularWorld
from swarmsim.agent.MazeAgent import MazeAgent
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor, vectorize, turn, project, lineCircleIntersect
from swarmsim.agent.control.AbstractController import AbstractController
import pygame
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

GOAL_ATTRACTION = 10
DEFENDER_REPULSION = 1

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

class CustomEvader(AbstractController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])

        sight_color = (200, 100, 0)

        for defender in [a for a in self.agent.world.population if a.team == "blue"]:
            bfovs: BinaryFOVSensor = defender.sensors[1]
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + vec / mag) * zoom + pan
            pygame.draw.line(screen, sight_color, head, tail, np.clip((3 / mag**2), 1, 5).astype(np.int16))
        
        if hasattr(self, "view_vector"):
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + 2 * self.view_vector / np.linalg.norm(self.view_vector)) * zoom + pan
            pygame.draw.line(screen, (100, 0, 200), head, tail)
    
    def __init__(self, agent=None, parent=None, **kwargs):
        super().__init__(agent, parent)

    def point_normal_to_segment(self, segvec, point):
        orthovec = np.array([segvec[1], -segvec[0]])
        return np.sign(turn(orthovec, segvec - point)) != np.sign(turn(orthovec, -point))

    def get_nearest_point_of_sensor(self, sensor: BinaryFOVSensor):
        options = []
        pos = self.agent.position
        agent: MazeAgent = sensor.agent
        options.append(agent.position - pos) # cone point / sensor origin
        left_rel, right_rel = sensor.getSectorVectors()
        left_rel, right_rel = sensor.r * np.asarray(left_rel[:2]), sensor.r * np.asarray(right_rel[:2])
        left_abs, right_abs = left_rel + agent.position, right_rel + agent.position
        options.append(left_abs - pos) # left whisker end
        options.append(right_abs - pos) # right whisker end
        left_ortho, right_ortho = np.array([left_rel[1], -left_rel[0]]), np.array([right_rel[1], -right_rel[0]])
        if self.point_normal_to_segment(left_rel, pos - agent.position):
            options.append(project(left_abs - pos, left_ortho)) # if applicable, intermediate point on left whisker
        if self.point_normal_to_segment(right_rel, pos - agent.position):
            options.append(project(right_abs - pos, right_ortho)) # if applicable, intermediate point on right whisker
        
        l180 = sensor.theta * 2 < np.pi
        u = pos - agent.position  # vector to agent
        leftTurn = turn(u, left_rel)
        rightTurn = turn(u, right_rel)
        # if fov < 180 use between minor arc, otherwise use not between minor arc
        if rightTurn <= 0 and 0 <= leftTurn if l180 else not (leftTurn < 0 and 0 < rightTurn):
            options.append(agent.position - pos + (sensor.r / np.linalg.norm(agent.position - pos)) * (pos - agent.position))
        return min(options, key=lambda v : np.sum(np.square(v)))

    def get_actions(self, agent: MazeAgent):
        pos = agent.position
        vector_sum = np.array([0, 0], dtype=np.float64)
        for defender in [a for a in agent.world.population if a.team == "blue"]:
            bfovs: BinaryFOVSensor = defender.sensors[1]
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            vector_sum += (-DEFENDER_REPULSION / mag**2) * vec / mag
        
        goal = agent.world.population[0]
        gvec = goal.position - pos
        vector_sum += GOAL_ATTRACTION * (gvec / np.linalg.norm(gvec))

        angle = np.atan2(vector_sum[1], vector_sum[0])
        sad = smallest_angular_difference(agent.angle, angle)
        
        self.view_vector = vector_sum

        if abs(sad) < np.pi / 2:
            v, w = SPEED_LIMIT, -np.sign(sad)
        else:
            v, w = -SPEED_LIMIT, np.sign(sad)
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
