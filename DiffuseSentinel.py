from Sentinel import SentinelController, smallest_angular_difference
import numpy as np


SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 200 # min 100, max 300
HALF_ANGLE = 1.4 # min 0.6, max 2


class DiffuseSentinelController(SentinelController):
    def __init__(self, speed_limit=SPEED_LIMIT, turn_limit=TURN_LIMIT, agent=None, parent=None):
        super().__init__(agent, parent, speed_limit=speed_limit, turn_limit=turn_limit)
        self.pseudoangle = 0 # track the agent angle internally
        self.pseudofirstseen = 0 # track the step at which a defender is seen, this is to stop the overlap prevention system from getting stuck

    def get_v_w(self, detected):
        if self.stage == 1: # first stage, diffuse and track average direction in which other agents are 
            if detected:
                self.cvec += np.array([np.cos(self.pseudoangle), np.sin(self.pseudoangle)]) * SPEED_LIMIT * self.agent.world.dt
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
