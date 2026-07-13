from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
SECOND_STAGE = 200
HALF_ANGLE = 1.4

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi



class DiffuseSentinelController(AbstractController):
    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent)
        self.pseudostep = 0 # track the world step count internally
        self.cvec = (0, 0) # track the average direction in which other agents are, then on toggle fix in the opposite direction
        self.pseudoangle = 0 # track the agent angle internally
        self.pseudofirstseen = 0 # track the step at which a defender is seen, this is to stop the overlap prevention system from getting stuck

    def get_actions(self, agent):
        self.pseudostep += 1 # increment internal world step count
        detected = agent.sensors[0].current_state

        if self.pseudostep < SECOND_STAGE: # first stage, diffuse and track average direction in which other agents are 
            if detected:
                self.cvec = (self.cvec[0] + np.cos(self.pseudoangle), self.cvec[1] + np.sin(self.pseudoangle)) # add unit vector in current direction

                v, w = -SPEED_LIMIT, TURN_LIMIT # back away and turn clockwise if other defender detected
            else:
                v, w = SPEED_LIMIT, TURN_LIMIT # go forward and turn clockwise if nothing detected
        else: # second stage, sit and scan away from the other defenders
            if SECOND_STAGE == self.pseudostep: # setup this stage
                self.persist = TURN_LIMIT
                self.cvec = (np.atan2(self.cvec[1], self.cvec[0]) + np.pi) % (2*np.pi) # set cvec to the opposite angle of the vector

            sad = smallest_angular_difference(self.pseudoangle, self.cvec)
            
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
