from swarmsim.util.collider.AABB import AABB
from swarmsim.world.RectangularWorld import RectangularWorld
from swarmsim.agent.MazeAgent import MazeAgent
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor, vectorize, turn, project, lineCircleIntersect
from swarmsim.agent.control.AbstractController import AbstractController
import pygame
import numpy as np
import types

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

GOAL_ATTRACTION = 20
DEFENDER_REPULSION = 1
PROJECTION_DELTA = 20

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

def draw_sensor_cone(sensor: BinaryFOVSensor, screen, offset=((0, 0), 1.0), color=(255, 0, 0)):
    pan, zoom = offset
    agent = sensor.agent
    magnitude = sensor.r
    head = agent.position * zoom + pan
    e_left, e_right = sensor.getSectorVectors()
    e_left, e_right = np.asarray(e_left[:2]), np.asarray(e_right[:2])
    tail_l = head + magnitude * e_left * zoom
    tail_r = head + magnitude * e_right * zoom
    pygame.draw.line(screen, color, head, tail_l)
    pygame.draw.line(screen, color, head, tail_r)
    width = max(1, round(0.01 * zoom))
    # pygame.draw.circle(screen, sight_color + (50,), head, self.r * zoom, width)
    # draw the arc of the sensor cone
    range_bbox = AABB.from_center_wh(head, sensor.r * 2 * zoom)
    langle = agent.angle + sensor.angle + sensor.theta
    rangle = agent.angle + sensor.angle - sensor.theta
    pygame.draw.arc(screen, color + (50,), range_bbox.to_rect(), -langle, -rangle, width)

class CustomEvader(AbstractController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        if not self.agent.is_highlighted:
            return
        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])

        # draw lines to closest points on defenders
        for defender in [a for a in self.agent.world.population if a.team == "blue"]:
            # draw line to closest point of agent view cone
            bfovs: BinaryFOVSensor = defender.sensors[1]
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + vec) * zoom + pan
            pygame.draw.line(screen, (200, 100, 0), head, tail, np.clip((3 / mag**2), 1, 5).astype(np.int16))
            # draw agent's view cone
            magnitude = bfovs.r
            head = defender.position * zoom + pan
            proj_tail = (defender.position + self.predict_agent_delta(defender, PROJECTION_DELTA)[0]) * zoom + pan
            pygame.draw.line(screen, (0, 150, 150), head, proj_tail)
            draw_sensor_cone(bfovs, screen, offset)
        
        # draw lines to closest points on predicted locations of defenders
        for defender in [a for a in self.agent.world.population if a.team == "blue"]:
            # draw line to closest point of agent view cone
            bfovs: BinaryFOVSensor = self.project_sensor(defender.sensors[1], PROJECTION_DELTA)
            defender = bfovs.agent
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + vec) * zoom + pan
            pygame.draw.line(screen, (200, 100, 0), head, tail, np.clip((3 / mag**2), 1, 5).astype(np.int16))
            # draw agent's view cone
            draw_sensor_cone(bfovs, screen, offset, (0, 150, 150))

        if hasattr(self, "view_vector"):
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + 2 * self.view_vector / np.linalg.norm(self.view_vector)) * zoom + pan
            pygame.draw.line(screen, (100, 0, 200), head, tail)
    
    def __init__(self, agent=None, parent=None, **kwargs):
        super().__init__(agent, parent)

    def point_normal_to_segment(self, segvec, point):
        orthovec = np.array([segvec[1], -segvec[0]])
        return np.sign(turn(orthovec, segvec - point)) != np.sign(turn(orthovec, -point))

    def predict_agent_delta(self, defender: MazeAgent, delta_steps):
        t = self.agent.world.dt * delta_steps
        v, w = defender.controller.get_actions(defender)
        if v == 0: # no speed
            return np.array([0, 0]), w * t
        v_vec = v * vectorize(defender.angle)
        if w == 0: # no angular velocity
            return v_vec * t, 0
        
        # defender will travel on a circular path, we can use geometry to compute where it will be if v and w hold
        r = abs(v / w)
        theta = t * w
        d = 2 * r * np.sin(theta / 2)
        d_angle = theta / 2

        return d * np.sign(v) * vectorize(defender.angle + d_angle), theta

    def project_sensor(self, sensor: BinaryFOVSensor, delta_steps):
        agent_delta_position, agent_delta_angle = self.predict_agent_delta(sensor.agent, delta_steps)
        fake_sensor = types.SimpleNamespace()
        fake_sensor.r = sensor.r
        fake_sensor.theta = sensor.theta
        fake_sensor.bias = sensor.bias
        fake_sensor.angle = 0
        fake_agent = types.SimpleNamespace()
        fake_agent.position = sensor.agent.position + agent_delta_position
        fake_agent.angle = sensor.agent.angle + agent_delta_angle
        fake_sensor.agent = fake_agent
        fake_sensor.getSectorVectors = lambda : (vectorize(fake_agent.angle + fake_sensor.bias + fake_sensor.theta), vectorize(fake_agent.angle + fake_sensor.bias - fake_sensor.theta))
        return fake_sensor
        
    def get_nearest_point_of_sensor(self, sensor: BinaryFOVSensor, check_inside=False):
        options = []
        pos = self.agent.position
        agent: MazeAgent = sensor.agent
        self_to_sensor_origin = agent.position - pos
        options.append(self_to_sensor_origin) # cone point / sensor origin
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
        
        in_arc = False

        l180 = sensor.theta * 2 < np.pi
        u = pos - agent.position  # vector to agent
        leftTurn = turn(u, left_rel)
        rightTurn = turn(u, right_rel)
        # if fov < 180 use between minor arc, otherwise use not between minor arc
        if rightTurn <= 0 and 0 <= leftTurn if l180 else not (leftTurn < 0 and 0 < rightTurn):
            in_arc = True
            options.append(agent.position - pos + (sensor.r / np.linalg.norm(agent.position - pos)) * (pos - agent.position))
        
        minimum = min(options, key=lambda v : np.sum(np.square(v)))
        if not check_inside: # this happens by default
            return minimum
        else: # this reverses the direction of the return vector if self is inside the predicted sensing cone
            return -minimum if in_arc and np.dot(self_to_sensor_origin, self_to_sensor_origin) < sensor.r**2 else minimum


    def get_actions(self, agent: MazeAgent):
        pos = agent.position
        vector_sum = np.array([0, 0], dtype=np.float64)
        for defender in [a for a in agent.world.population if a.team == "blue"]:
            # repulse sensing cones
            bfovs: BinaryFOVSensor = defender.sensors[1]
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            vector_sum -= (DEFENDER_REPULSION / mag**2) * vec / mag
            # repulse predicted sensing cones
            predicted_sensor: BinaryFOVSensor = self.project_sensor(defender.sensors[1], PROJECTION_DELTA)
            p_vec = self.get_nearest_point_of_sensor(predicted_sensor, check_inside=True)
            p_mag = np.linalg.norm(p_vec)
            vector_sum -= (DEFENDER_REPULSION / p_mag**2) * p_vec / p_mag
        
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
