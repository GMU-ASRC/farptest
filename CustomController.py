from swarmsim.agent.control.AbstractController import AbstractController
from swarmsim.agent.control.MultibitBinaryController import MultibitBinaryController
from Sentinel import setinel_draw
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6


class CustomController(MultibitBinaryController):
    def draw(self, screen, offset=((0, 0), 1.0)):
        setinel_draw(self, screen, offset)

    def __init__(self, agent=None, parent=None, ):
        super().__init__({
            0b00: (SPEED_LIMIT, TURN_LIMIT),
            0b01: (SPEED_LIMIT, -TURN_LIMIT),
            0b10: (SPEED_LIMIT, -TURN_LIMIT),
            0b11: (SPEED_LIMIT, -TURN_LIMIT)
        }, agent=agent, parent=parent, sensor_ids=[0, 2])

    def get_actions(self, agent):
        v, w = super().get_actions(agent)
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
