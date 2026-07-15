import pygame
from swarmsim.util.collider.AABB import AABB
from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 200
DISPERSAL_TIME = 50
HALF_ANGLE = 1.4

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi

def full_whisker_override(self, s):
    pass

def setinel_draw(self, screen, offset=((0, 0), 1.0)):
    internal_to_real_angle = lambda t : self.agent.angle - self.pseudoangle + t

    pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
    
    if hasattr(self, "cangle"):        
        pygame.draw.line(screen, pygame.colordict.THECOLORS["violet"], self.agent.pos * zoom + pan, self.agent.pos * zoom + pan + np.sqrt(np.sum(np.array(self.cvec)**2))*np.array([np.cos(internal_to_real_angle(self.cangle)), np.sin(internal_to_real_angle(self.cangle))]))
    
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

class DisperseSentinelController(AbstractController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        setinel_draw(self, screen, offset)

    def set_clock(self):
        self.clock = self.pseudostep
    def clock_wait(self, steps):
        return steps <= self.pseudostep - self.clock
    
    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent)
        self.pseudostep = 0 # track the world step count internally
        self.cvec = (0, 0) # track the average direction in which other agents are, then on toggle fix in the opposite direction
        self.cangle = 0
        self.pseudoangle = 0 # track the agent angle internally
        self.pseudofirstseen = 0 # track the step at which a defender is seen, this is to stop the overlap prevention system from getting stuck
        self.persist = TURN_LIMIT
        self.stage = 1
        self.set_clock()
    
    def get_actions(self, agent):
        self.pseudostep += 1 # increment internal world step count
        detected = agent.sensors[0].current_state

        if self.stage == 1:
            if detected:
                self.cvec = (self.cvec[0] + np.cos(self.pseudoangle), self.cvec[1] + np.sin(self.pseudoangle)) # add unit vector in current direction
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)

            v, w = 0, TURN_LIMIT

            if abs(smallest_angular_difference(self.pseudoangle, self.cangle)) < np.pi/2: 
                v = -SPEED_LIMIT/3
            if abs(smallest_angular_difference(self.pseudoangle, (self.cangle + np.pi) % (2 * np.pi))) < np.pi/2:
                v = SPEED_LIMIT/3

            if self.clock_wait(SECOND_STAGE): # switch to and setup second stage
                self.stage = 2
                self.persist = TURN_LIMIT
                
        else:
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

        self.pseudoangle += w * self.agent.world.dt # udate internal agent angle
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
