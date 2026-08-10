import pygame
from swarmsim.agent.MazeAgent import MazeAgent
from Sentinel import SentinelController, smallest_angular_difference
from swarmsim.sensors.BinaryFOVSensor import vectorize
import numpy as np


SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 400 # min 100, max 300
HALF_ANGLE = 1.44 # min 0.6, max 2

class HeuristicMillController(SentinelController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        super().draw(screen, offset)
        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
        pseudo_to_world_angle_diff = -(self.agent.angle - self.pseudoangle) # negated becuase the y axis is flipped
        rot_mat = np.array([[np.cos(pseudo_to_world_angle_diff), np.sin(pseudo_to_world_angle_diff)], [np.cos(pseudo_to_world_angle_diff + np.pi/2), np.sin(pseudo_to_world_angle_diff + np.pi/2)]])
        pseudo_to_world_position = lambda ppos : self.original_position + rot_mat @ ppos
        pygame.draw.circle(screen, (0, 150, 255, 50), pseudo_to_world_position(self.pseudoposition) * zoom + pan, 0.1 * zoom, 4)
        

        if not self.agent.is_highlighted:
            return
        if (cen := self.recording_centroid()) is not None:
            pygame.draw.circle(screen, pygame.colordict.THECOLORS["violetred"], pseudo_to_world_position(cen) * zoom + pan, 0.05 * zoom, 4)
            pygame.draw.line(screen, pygame.colordict.THECOLORS["violetred"], pseudo_to_world_position(cen) * zoom + pan, (pseudo_to_world_position(cen) + np.linalg.norm(self.cvec) * vectorize(self.cangle - pseudo_to_world_angle_diff)) * zoom + pan)
            pygame.draw.line(screen, pygame.colordict.THECOLORS["violetred"], pseudo_to_world_position(cen) * zoom + pan, pseudo_to_world_position(self.pseudoposition) * zoom + pan)
        
        for ppos, pangle in zip(self.position_recording, self.angle_recording):
            pos = pseudo_to_world_position(ppos)
            angle = pangle - pseudo_to_world_angle_diff
            vec = vectorize(angle) / 4
            pygame.draw.line(screen, pygame.colordict.THECOLORS["violet"], pos * zoom + pan, (pos + vec) * zoom + pan)

        

    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent, speed_limit=SPEED_LIMIT, turn_limit=TURN_LIMIT)
        self.pseudofirstseen = 0 # track the step at which a defender is seen, this is to stop the overlap prevention system from getting stuck
        self.pseudoposition = np.array([0, 0], dtype=np.float64)
        self.original_position = np.array(agent.position) # NOT used for control, just debug drawing!
        self.position_recording = []
        self.angle_recording = []

    def get_v_w(self, detected):
        if self.stage == 1: # first stage, diffuse and track average direction in which other agents are 
            if detected:
                self.position_recording.append(np.array(self.pseudoposition))
                self.angle_recording.append(self.pseudoangle)
                self.cvec += vectorize(self.pseudoangle) * SPEED_LIMIT * self.agent.world.dt
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)

                v, w = -SPEED_LIMIT, TURN_LIMIT # back away and turn clockwise if other defender detected
            else:
                v, w = SPEED_LIMIT, TURN_LIMIT # go forward and turn clockwise if nothing detected
            
            if self.clock_wait(SECOND_STAGE):
                self.stage = 2
                self.persist = TURN_LIMIT
        else: # second stage, sit and scan away from the other defenders
            sad = smallest_angular_difference(self.pseudoangle, (self.cangle + np.pi) % (2 * np.pi))
            
            if HALF_ANGLE < abs(sad): # start scanning the other way when the edge of the scan arc is reached
                self.persist = TURN_LIMIT * -np.sign(sad)
            else: # overlap prevention system
                if (self.pseudofirstseen == 0 or 100 < self.pseudostep - self.pseudofirstseen) and detected: # if first defender detected in the last 100 steps
                    self.pseudofirstseen = self.pseudostep
                    self.persist *= -1
            if not detected:
                self.pseudofirstseen = 0

            v, w = 0, self.persist

        return v, w
    
    def project_agent_step(self, v, w):
        t = self.agent.world.dt
        if v == 0: # no speed
            return np.array([0, 0])
        v_vec = v * vectorize(self.pseudoangle)
        if w == 0: # no angular velocity
            return v_vec * t
        
        # defender will travel on a circular path, we can use geometry to compute where it will be one step in the future
        r = abs(v / w)
        theta = t * w
        d_angle = theta / 2
        d = 2 * r * np.sin(d_angle)
        
        return d * np.sign(v) * vectorize(self.pseudoangle + d_angle)
    
    def recording_centroid(self):
        return np.mean(np.array(self.position_recording), axis=0) if self.position_recording else None

    def get_actions(self, agent: MazeAgent):
        v, w = super().get_actions(agent)
        self.pseudoposition += self.project_agent_step(v, w)
        return v, w
