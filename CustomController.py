from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

VMAX, WMAX = SPEED_LIMIT, TURN_LIMIT
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
INPUT_COUNT = 3


class FarpRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=INPUT_COUNT, hidden_size=4, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x, h0=None):
        rnn_out, hn = self.rnn(x, h0)
        final_out = self.linear(rnn_out)
        return final_out, hn


class CustomController(AbstractController):
    def __init__(self, genome=None, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.sens_hist = [0] * INPUT_COUNT
        self.model = FarpRNN()

        if genome is None:
            g = torch.tensor([-2.0386,  4.2847,  2.0425, -1.2848,  0.7169, -0.5713, -2.1535,  1.1969,
        -2.0398, -2.2568,  3.6647,  1.4011,  3.1853,  5.7340, -0.1510, -0.7145,
         0.7686, -3.5267, -0.2312, -0.5623, -0.8169, -1.1686,  2.3035,  2.3843,
         3.7461, -2.3514, -2.1720, -1.2997,  1.0924, -0.4279, -2.8843,  0.2283,
         3.3923, -2.7025,  1.7441, -2.2811,  0.3741, -2.1146,  1.8763,  4.9663,
         0.4077,  1.6240,  1.8504,  0.5965,  3.2910,  2.1643,  1.2082,  1.1828,
         0.7222, -1.7474, -0.0318, -1.3562,  1.5694,  1.6706, -1.6561,  1.5702])
            # raise NotImplementedError()
        else:
            if isinstance(genome[0], str):
                g = torch.from_numpy(np.fromstring(genome[0], sep=' ')).float()
            else:
                g = torch.asarray(genome).float()

        vector_to_parameters(g, self.model.parameters())

    @torch.inference_mode()
    def get_actions(self, agent): # type: ignore
        detected = float(agent.sensors[self.sensor_id].current_state != 0)
        output, _ = self.model(
            torch.tensor(self.sens_hist[-INPUT_COUNT:]).float().view(1, 1, INPUT_COUNT)
        )
        self.sens_hist.append(detected)

        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }