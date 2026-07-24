from Sentinel import SentinelController, smallest_angular_difference, vec_magnitude
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 200
DISPERSAL_TIME = 50
HALF_ANGLE = 1.4

class ComplexSentinelController(SentinelController):
    
    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent, speed_limit=SPEED_LIMIT, turn_limit=TURN_LIMIT)
        self.pseudofirstseen = 0 # track the step at which a defender is seen, this is to stop the overlap prevention system from getting stuck
        self.persist = TURN_LIMIT
    
    def get_v_w(self, detected):
        if self.stage == 1:
            if detected:
                self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
            v, w = 0, TURN_LIMIT
            if self.clock_wait(SECOND_STAGE): # switch to and setup second stage
                self.stage = 2
                self.persist = TURN_LIMIT
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
        elif self.stage == 2:
            sad = smallest_angular_difference(self.pseudoangle, self.cangle)
            v, w = 0, TURN_LIMIT * -np.sign(sad)
            if abs(sad) < 0.1: # switch to third stage
                self.stage = 3
                self.cvec = np.array([0, 0], dtype=np.float64)
                self.set_clock()
        elif self.stage == 3:
            v, w = -SPEED_LIMIT, 0
            self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
            if self.clock_wait(DISPERSAL_TIME / 2):
                self.set_clock()
                self.stage = 4
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
        elif self.stage == 4:
            v, w = 0, TURN_LIMIT if self.clock_wait((np.pi / 2) / (TURN_LIMIT * self.agent.world.dt)) else -TURN_LIMIT
            self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
            if self.clock_wait((np.pi * 2) / (TURN_LIMIT * self.agent.world.dt)):
                self.set_clock()
                self.stage = 5
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
        elif self.stage == 5:
            sad = smallest_angular_difference(self.pseudoangle, (self.cangle + np.pi) % (2 * np.pi))
            v, w = 0, TURN_LIMIT * -np.sign(sad)
            if abs(sad) < 0.1: # switch to third stage
                self.stage = 6
                self.set_clock()
        elif self.stage == 6:
            v, w = SPEED_LIMIT, 0
            self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
            if self.clock_wait(DISPERSAL_TIME / 2):
                self.set_clock()
                self.stage = 7
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
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

        return v, w
