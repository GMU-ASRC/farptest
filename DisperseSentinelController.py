from Sentinel import SentinelController, smallest_angular_difference, vec_magnitude
from swarmsim.util.collider.AABB import AABB
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 250
THIRD_STAGE = 0
MAGNITUDE_COUNTERSPEED = 0.2
HALF_ANGLE = 1.4

class DisperseSentinelController(SentinelController):
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
            # if detected:
                # v = -SPEED_LIMIT
            if abs(smallest_angular_difference(self.pseudoangle, self.cangle)) < np.pi/2: 
                v = -MAGNITUDE_COUNTERSPEED/vec_magnitude(self.cvec) * SPEED_LIMIT
            if abs(smallest_angular_difference(self.pseudoangle, (self.cangle + np.pi) % (2 * np.pi))) < np.pi/2:
                v = MAGNITUDE_COUNTERSPEED/vec_magnitude(self.cvec) * SPEED_LIMIT

            
            if self.clock_wait(SECOND_STAGE): # switch to and setup second stage
                self.set_clock()
                self.stage = 2
                self.persist = TURN_LIMIT
        elif self.stage == 2:
            if detected:
                self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
                self.cangle = (np.atan2(self.cvec[1], self.cvec[0])) % (2 * np.pi)
            
            v, w = 0, TURN_LIMIT

            if detected:
                v, w = -SPEED_LIMIT/3, TURN_LIMIT
            else:
                v, w = 0, TURN_LIMIT

            if self.clock_wait(THIRD_STAGE): # switch to and setup second stage
                self.stage = 3
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

        return v, w
