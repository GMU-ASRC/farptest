from swarmsim.agent.control.AbstractController import AbstractController
from swarmsim.agent.control.MultibitBinaryController import MultibitBinaryController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6


class CustomController(MultibitBinaryController):
    def __init__(self, agent=None, parent=None, ):
        super().__init__({
            0b00: (SPEED_LIMIT, TURN_LIMIT),
            0b01: (SPEED_LIMIT, -TURN_LIMIT),
            0b10: (SPEED_LIMIT, -TURN_LIMIT),
            0b11: (SPEED_LIMIT, -TURN_LIMIT)
        }, agent, parent, sensor_ids=[0, 2])

    def get_actions(self, agent):
        ac = super().get_actions(agent)
        try:
            v, w = ac
        except:
            print(ac, self.outputs)
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
