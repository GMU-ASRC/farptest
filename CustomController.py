from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6
TOGGLE_TIME = 200
HALF_ANGLE = 1.6

def smallest_angular_difference(a1, a2):
    a = a1 - a2
    return (a + np.pi) % (2*np.pi) - np.pi



class CustomController(AbstractController):
    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent)
        self.pseudostep = 0
        self.cvec = (0, 0)
        self.pseudoangle = 0
        self.pseudofirstseen = 0

    def get_actions(self, agent):
        self.pseudostep += 1
        detected = agent.sensors[0].current_state
        if detected:
            v, w = -SPEED_LIMIT, TURN_LIMIT
        else:
            v, w = SPEED_LIMIT, TURN_LIMIT

        if self.pseudostep < TOGGLE_TIME:
            if detected:
                self.cvec = (self.cvec[0] + np.cos(self.pseudoangle), self.cvec[1] + np.sin(self.pseudoangle))

        if TOGGLE_TIME <= self.pseudostep:
            if TOGGLE_TIME == self.pseudostep:
                self.persist = 0.6
                self.cvec = (np.atan2(self.cvec[1], self.cvec[0]) + np.pi) % (2*np.pi)
            else:
                sad = smallest_angular_difference(self.pseudoangle, self.cvec)
                
                if HALF_ANGLE < abs(sad):
                    self.persist = 0.6 * -np.sign(sad)
                else:
                    if (self.pseudofirstseen == 0 or 100 < self.pseudostep - self.pseudofirstseen) and detected:
                        self.pseudofirstseen = self.pseudostep
                        self.persist *= -1
                if not detected:
                    self.pseudofirstseen = 0

            v, w = 0, self.persist

        self.pseudoangle += w * self.agent.world.dt
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
