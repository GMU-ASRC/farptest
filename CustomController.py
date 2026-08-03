from Sentinel import setinel_draw, SentinelController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6


class CustomController(SentinelController):
    def __init__(self, agent=None, parent=None, ):
        super().__init__(agent=agent, parent=parent)
        self.stage = 1

    def get_v_w(self, detected):
        if self.stage == 1:
            v, w = SPEED_LIMIT, -TURN_LIMIT/2
            if 1 < self.pseudostep and not self.agent.sensors[2].current_state:
                self.stage = 2
        else:
            
            v = 0 if detected else SPEED_LIMIT
            w = TURN_LIMIT / 2 if self.agent.sensors[2].current_state else -TURN_LIMIT
        return v, w
