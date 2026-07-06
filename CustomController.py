from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6


class CustomController(AbstractController):
    def __init__(self, agent=None, parent=None):
        super().__init__(agent, parent)

    def get_actions(self, agent):
        v, w = 0, 0
        return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE
