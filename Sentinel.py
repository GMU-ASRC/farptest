import pygame
from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np
from swarmsim.util.collider.AABB import AABB


SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

def vec_magnitude(vec):
    return np.sqrt(np.sum(np.array(vec)**2))

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

def setinel_draw(self, screen, offset=((0, 0), 1.0)):
    internal_to_real_angle = lambda t : self.agent.angle - self.pseudoangle + t

    pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
    
    if hasattr(self, "cangle"):        
        pygame.draw.line(screen, pygame.colordict.THECOLORS["violet"], self.agent.pos * zoom + pan, self.agent.pos * zoom + pan + 100*vec_magnitude(self.cvec)*np.array([np.cos(internal_to_real_angle(self.cangle)), np.sin(internal_to_real_angle(self.cangle))]))
    
    self = self.agent.sensors[0]

    sight_color = (255, 0, 0)

    magnitude = self.r

    head = np.asarray(self.agent.getPosition()) * zoom + pan
    e_left, e_right = self.getSectorVectors()
    e_left, e_right = np.asarray(e_left[:2]), np.asarray(e_right[:2])

    tail_l = head + magnitude * e_left * zoom
    tail_r = head + magnitude * e_right * zoom

    pygame.draw.line(screen, sight_color, head, tail_l)
    pygame.draw.line(screen, sight_color, head, tail_r)

    width = max(1, round(0.01 * zoom))
    # pygame.draw.circle(screen, sight_color + (50,), head, self.r * zoom, width)
    # draw the arc of the sensor cone
    range_bbox = AABB.from_center_wh(head, self.r * 2 * zoom)
    langle = self.agent.angle + self.angle + self.theta
    rangle = self.agent.angle + self.angle - self.theta
    pygame.draw.arc(screen, sight_color + (50,), range_bbox.to_rect(), -langle, -rangle, width)

class SentinelController(AbstractController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        setinel_draw(self, screen, offset)
    
    def set_clock(self):
        self.clock = self.pseudostep
    def clock_wait(self, steps):
        return steps <= self.pseudostep - self.clock

    def __init__(self, agent=None, parent=None, speed_limit = SPEED_LIMIT, turn_limit = TURN_LIMIT):
        super().__init__(agent, parent)
        self.speed_limit, self.turn_limit = speed_limit, turn_limit
        self.pseudostep = 0 # track the world step count internally
        self.cvec = np.array([0, 0], dtype=np.float64) # track the average direction in which other agents are, then on toggle fix in the opposite direction
        self.cangle = 0
        self.pseudoangle = 0 # track the agent angle internally
        self.stage = 1
        self.set_clock()

    def get_actions(self, agent):
        self.pseudostep += 1 # increment internal world step count
        detected = agent.sensors[0].current_state

        v, w = self.get_v_w(detected)

        self.pseudoangle += w * self.agent.world.dt # udate internal agent angle
        return np.clip(v, -self.speed_limit, self.speed_limit), np.clip(w, -self.turn_limit, self.turn_limit)  # DO NOT CHANGE THIS LINE
    
    def get_v_w(self, detected):
        if self.stage == 1: # first stage, diffuse and track average direction in which other agents are 
            if detected:
                self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)

                v, w = -SPEED_LIMIT, TURN_LIMIT # back away and turn clockwise if other defender detected
            else:
                v, w = SPEED_LIMIT, TURN_LIMIT # go forward and turn clockwise if nothing detected
            
            if self.clock_wait(200):
                self.stage = 2
                self.persist = TURN_LIMIT
        else: # second stage, sit and scan away from the other defenders
            sad = smallest_angular_difference(self.pseudoangle, (self.cangle + np.pi) % (2 * np.pi))
            
            if 1.4 < abs(sad): # start scanning the other way when the edge of the scan arc is reached
                self.persist = TURN_LIMIT * -np.sign(sad)

            v, w = 0, self.persist
        
        return v, w
