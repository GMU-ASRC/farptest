from swarmsim.util.collider.AABB import AABB
from swarmsim.world.RectangularWorld import RectangularWorld
from swarmsim.agent.MazeAgent import MazeAgent
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor, vectorize, turn, project, lineCircleIntersect
from swarmsim.agent.control.AbstractController import AbstractController
import pygame
import numpy as np
import types
from Heatmap import Heatmap

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

GOAL_ATTRACTION = 20
CLOSE_GOAL_ATTRACTION = 1
DEFENDER_REPULSION = 1
PROJECTION_DELTA = 20
ATTACK_POINT_DISTANCE = 2.5
DIVE_TRIGGER_DISTANCE = 1
CENTROID_ADJUSTMENT_CUTOFF = 400
KILLZONE_PADDING = 0.1
KILLZONE_NUMBER = 3
KILLZONE_DECAY = 0.7

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

def orthogonal_vector(v):
    return np.array([v[1], -v[0]])

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

        self.heatmap.draw(screen, zoom, pan)

        if self.stage == 1:
            cen = self.get_defender_centroid()
            pygame.draw.circle(screen, (255, 255, 0, 50), cen * zoom + pan, 0.1 * zoom, 4)
            goal_pos = self.agent.world.population[0].position
            head = goal_pos * zoom + pan
            tail = (goal_pos + self.defense_vec) * zoom + pan
            pygame.draw.line(screen, (255, 255, 0, 50), head, tail, 4)
            dist = np.linalg.norm(self.defense_vec)
            if dist > 0:
                attack_point = goal_pos + ATTACK_POINT_DISTANCE * -self.defense_vec / dist
                pygame.draw.circle(screen, (255, 0, 0, 50), attack_point * zoom + pan, 0.1 * zoom, 4)

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
            bfovs = defender.sensors[1]
            self.draw_sensor_path(bfovs, screen, zoom, pan)
            continue
            for kd_steps in self.killzone_delta_steps():
                pos, radius = self.project_killzone(bfovs, kd_steps)
                pygame.draw.circle(screen, (0, 0, 255), (pos) * zoom + pan, radius * zoom, width=2)
            vec, _ = self.vector_away_from_killzone(bfovs, PROJECTION_DELTA)
            mag = np.linalg.norm(vec)
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + vec) * zoom + pan
            pygame.draw.line(screen, (200, 100, 0), head, tail, np.clip((3 / mag**2), 1, 5).astype(np.int16))
            continue
            # draw line to closest point of agent view cone
            bfovs: BinaryFOVSensor = self.project_sensor(bfovs, PROJECTION_DELTA)
            defender = bfovs.agent
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + vec) * zoom + pan
            pygame.draw.line(screen, (200, 100, 0), head, tail, np.clip((3 / mag**2), 1, 5).astype(np.int16))
            # draw projected sensor view cone
            draw_sensor_cone(bfovs, screen, offset, (0, 150, 150))
            

        if hasattr(self, "view_vector"):
            head = self.agent.position * zoom + pan
            tail = (self.agent.position + (self.view_vector / np.linalg.norm(self.view_vector) * np.sqrt(np.linalg.norm(self.view_vector)))) * zoom + pan
            pygame.draw.line(screen, (100, 0, 200), head, tail)
    
    def __init__(self, agent=None, parent=None, **kwargs):
        super().__init__(agent, parent)
        self.stage = 1
        self.defense_vec = np.array([0, 0], dtype=np.float64)
        self.defender_positions = {}
        self.defender_positions_prev = {}
        self.pseudostep = 0
        self.setup_heatmap()
        
    def setup_heatmap(self):
        self.goal = self.agent.world.population[0]
        # self.dbg_center = self.agent.position + np.asarray([0.7, 0.6]) * (self.goal.position - self.agent.position)
        # self.dbg_radius = np.linalg.norm(self.goal.position - self.agent.position) * 0.5
        # self.dbg_radius += self.goal.radius
        # self.dbg_radius *= 1.4
        self.heatmap = Heatmap(center=self.goal.position,
                               radius=self.goal.radius * 3.5,
                               decay_rate=0.8)

    def point_normal_to_segment(self, segvec, point):
        orthovec = orthogonal_vector(segvec)
        return np.sign(turn(orthovec, segvec - point)) != np.sign(turn(orthovec, -point))

    def calculate_defender_v_w(self, defender: MazeAgent):
        step = self.agent.world.total_steps
        if defender.name not in self.defender_positions or self.defender_positions[defender.name][2] < step:
            if defender.name in self.defender_positions:
                self.defender_positions_prev[defender.name] = self.defender_positions[defender.name]
            self.defender_positions[defender.name] = (np.array(defender.position), defender.angle, step)
        
        curr = self.defender_positions[defender.name]
        prev = self.defender_positions_prev[defender.name] if defender.name in self.defender_positions_prev else curr
        
        theta = curr[1] - prev[1]
        w = theta / self.agent.world.dt

        delta_pos = curr[0] - prev[0]
        d = np.linalg.norm(delta_pos)

        if d == 0:
            return 0, w
        
        if theta == 0:
            v = d / self.agent.world.dt
            return v, w

        r = d / (2 * np.sin(theta / 2))
        sproj = np.dot(vectorize(defender.angle), delta_pos) / np.dot(delta_pos, delta_pos)
        v = r * w * np.sign(sproj)
        return v, w
        

    def predict_agent_delta(self, defender: MazeAgent, delta_steps, v_w = None):
        t = self.agent.world.dt * delta_steps
        v, w = v_w if v_w else self.calculate_defender_v_w(defender)
        if v == 0: # no speed
            return np.array([0, 0]), w * t
        v_vec = v * vectorize(defender.angle)
        if w == 0: # no angular velocity
            return v_vec * t, 0
        
        # defender will travel on a circular path, we can use geometry to compute where it will be if v and w hold
        r = abs(v / w)
        theta = t * w
        d_angle = theta / 2
        d = 2 * r * np.sin(d_angle)
        
        return d * np.sign(v) * vectorize(defender.angle + d_angle), theta

    def draw_sensor_path(self, sensor: BinaryFOVSensor, screen, zoom, pan):
        t = self.agent.world.dt
        v, w = self.calculate_defender_v_w(sensor.agent)
        if v == 0: # no speed
            return np.array([0.0, 0.0])
        v_vec = v * vectorize(sensor.agent.angle)
        if w == 0: # no angular velocity
            return np.array([0.0, 0.0])
        
        # defender will travel on a circular path, we can use geometry to compute where it will be if v and w hold
        r = abs(v / w)
        circle_center = sensor.agent.position + orthogonal_vector(vectorize(sensor.agent.angle)) * r * -np.sign(w)
        rel_left, rel_right = sensor.getSectorVectors()
        abs_left, abs_right = sensor.agent.position + sensor.r * rel_left[:2], sensor.agent.position + sensor.r * rel_right[:2]
        radius_left, radius_right = np.linalg.norm(abs_left - circle_center), np.linalg.norm(abs_right - circle_center)
        min_radius = min(r, radius_left, radius_right) - self.agent.radius
        max_radius = max(r, radius_left, radius_right) + self.agent.radius
        
        if min_radius < np.linalg.norm(self.agent.position - circle_center) < max_radius:
            circen_to_defender = sensor.agent.position - circle_center
            circen_to_self = self.agent.position - circle_center
            if screen:
                pygame.draw.circle(screen, (0, 255, 255), circle_center * zoom + pan, min_radius*zoom, 2)
                pygame.draw.circle(screen, (0, 255, 255), circle_center * zoom + pan, max_radius*zoom, 2)
            defender_angle = np.atan2(circen_to_defender[1], circen_to_defender[0])

            middle_radius = (max_radius + min_radius) / 2
            u_cts = circen_to_self / np.linalg.norm(circen_to_self) * (-1 if np.linalg.norm(circen_to_self) < middle_radius else 1)
            if screen:
                range_bbox = AABB.from_center_wh(circle_center * zoom + pan, r * 2 * zoom)
                pygame.draw.line(screen, (0, 255, 255), self.agent.position * zoom + pan, (self.agent.position + u_cts) * zoom + pan)
                pygame.draw.arc(screen, (150, 0, 255), range_bbox.to_rect(), *sorted([-defender_angle, -(defender_angle + w)]), width=10)

            return u_cts
        
        return np.array([0.0, 0.0])

        
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
    
    def project_killzone(self, sensor: BinaryFOVSensor, delta_steps):
        defender = sensor.agent
        v, w = self.calculate_defender_v_w(defender)
        min_r = sensor.r - (np.sin(sensor.theta) * sensor.r * 2) if w == 0 else SPEED_LIMIT / abs(w)
        max_r = sensor.r
        radius = (0 if max_r - min_r < 0 else (max_r - min_r) / 2) + KILLZONE_PADDING
        dpos, dangle = self.predict_agent_delta(defender, delta_steps, (v, w))
        fpos = defender.position + dpos
        fangle = defender.angle + dangle
        return fpos + vectorize(fangle) * (min_r + radius), radius

    def killzone_delta_steps(self):
        return list(range(0, PROJECTION_DELTA+1, PROJECTION_DELTA//(KILLZONE_NUMBER-1)))
    def vector_away_from_killzone(self, sensor: BinaryFOVSensor, delta_steps):
        defender = sensor.agent

        projections = [(*self.project_killzone(defender.sensors[1], delta_steps), kd_steps) for kd_steps in self.killzone_delta_steps()]
        distances = [ppos - self.agent.position for ppos, _, kd_steps in projections]
        dists_msq = [np.dot(d, d) for d in distances]
        midx = np.argmin(dists_msq)
        shorten = lambda v, r : v * max((np.linalg.norm(v) - r) / np.linalg.norm(v), 0.1)
        if midx == 0:
            return shorten(distances[0], projections[0][1]), 0
        prev_delta = projections[midx][0] - projections[midx - 1][0]
        idx = midx -1 if dists_msq[midx - 1] < np.dot(prev_delta, prev_delta) else midx
        return shorten(distances[idx], projections[idx][1]), idx


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
        left_ortho, right_ortho = orthogonal_vector(left_rel), orthogonal_vector(right_rel)
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

    def get_defender_centroid(self):
        defender_positions = np.array([a.position for a in self.agent.world.population if a.team == "blue"], dtype=np.float64)
        return np.mean(defender_positions, axis=0)

    def get_v_w_for_angle(self, angle):
        sad = smallest_angular_difference(self.agent.angle, angle)
            
        if abs(sad) < np.pi / 2:
            return SPEED_LIMIT, -np.sign(sad)
        else:
            return -SPEED_LIMIT, np.sign(sad)

    def get_actions(self, agent: MazeAgent):
        vector_sum = np.array([0, 0], dtype=np.float64)
        pos = agent.position
        goal_agent = agent.world.population[0]
        goal_pos = goal_agent.position
        vec_to_goal = goal_pos - pos

        if self.stage == 1: # stage 1, attract towards attack point
            if self.pseudostep < CENTROID_ADJUSTMENT_CUTOFF:
                self.defense_vec += agent.world.dt * (self.get_defender_centroid() - goal_pos)
                # self.heatmap.update(agent.world, [a for a in agent.world.population if a.team == "blue"], agent.world.dt)
                # self.defense_vec = self.heatmap.goal_heatmap_vector()
            if np.dot(self.defense_vec, self.defense_vec) < 1e-3: # prevent divide by zero
                attack_point = goal_pos
            else:
                attack_point = goal_pos + ATTACK_POINT_DISTANCE * -self.defense_vec / np.linalg.norm(self.defense_vec)
            vec_to_attack_point = attack_point - pos
            vtg_msq = np.dot(vec_to_goal, vec_to_goal)
            vtap_msq = np.dot(vec_to_attack_point, vec_to_attack_point)
            tan_nav = vtg_msq < vtap_msq
            if tan_nav: # if attack point is on the far side of the circle
                vtg_mag = np.sqrt(vtg_msq)
                inside_atkpd = vtg_mag < ATTACK_POINT_DISTANCE
                if inside_atkpd: # move tangentially when inside attack point distance
                    move_vec = np.array([vec_to_goal[1], -vec_to_goal[0]]) * -np.sign(turn(vec_to_goal, vec_to_attack_point))
                else: # when outside attack point distance, move towards tangent point
                    theta = np.arcsin(ATTACK_POINT_DISTANCE / vtg_mag) * np.sign(turn(vec_to_goal, vec_to_attack_point))
                    angle_to_tan_point = np.atan2(vec_to_goal[1], vec_to_goal[0]) + theta
                    move_vec = vectorize(angle_to_tan_point)
            else: # go towards attack point
                move_vec = vec_to_attack_point
            if vtap_msq < DIVE_TRIGGER_DISTANCE**2:
                self.stage = 2
            vector_sum += GOAL_ATTRACTION * (move_vec / np.linalg.norm(move_vec))
        else: # stage 2, attract towards goal
            vector_sum += max(GOAL_ATTRACTION * (vec_to_goal / np.linalg.norm(vec_to_goal)), CLOSE_GOAL_ATTRACTION * (vec_to_goal / np.linalg.norm(vec_to_goal)**3), key=lambda v : np.dot(v,v))
        
        # apf repulse defender
        for defender in [a for a in agent.world.population if a.team == "blue"]:
            # repulse sensing cones
            bfovs: BinaryFOVSensor = defender.sensors[1]
            vec = self.get_nearest_point_of_sensor(bfovs)
            mag = np.linalg.norm(vec)
            vector_sum -= (DEFENDER_REPULSION / mag**2) * vec / mag
            # repulse predicted sensing cones
            # predicted_sensor: BinaryFOVSensor = self.project_sensor(defender.sensors[1], PROJECTION_DELTA)
            # p_vec = self.get_nearest_point_of_sensor(predicted_sensor, check_inside=True)
            # p_mag = np.linalg.norm(p_vec)
            # vector_sum -= (DEFENDER_REPULSION / p_mag**2) * p_vec / p_mag
            # repulse killzones
            # k_vec, idx = self.vector_away_from_killzone(bfovs, PROJECTION_DELTA)
            # k_mag = np.linalg.norm(k_vec)
            # vector_sum -= ((DEFENDER_REPULSION / k_mag**2) * k_vec / k_mag) * KILLZONE_DECAY**idx
            vector_sum += (DEFENDER_REPULSION / mag**2) * self.draw_sensor_path(bfovs, None, None, None)
            
        
        
        self.view_vector = vector_sum

        angle = np.atan2(vector_sum[1], vector_sum[0])
        v, w = self.get_v_w_for_angle(angle)
    
        self.pseudostep += 1
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
