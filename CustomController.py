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
SENS_WINDOW = 3
INPUT_COUNT = 1 + SENS_WINDOW


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
        self.pseudo_step = 0
        self.sens_hist = [0.0] * SENS_WINDOW
        self.model = FarpRNN()

        if genome is None:
            g = torch.tensor([ 0.9112, -0.6273,  1.3278,  2.1919, -3.2460,  0.5639,  0.0881,  1.1331,
         1.4455, -0.2824,  0.4013, -0.7607,  0.0730, -1.0330, -0.9326, -0.7260,
        -1.9839, -2.8432,  1.5935, -0.5822, -0.8911, -0.2025,  2.8701,  0.1240,
         1.5219, -1.3075, -2.0714, -3.1264,  0.8041,  1.8662,  0.7393,  1.3487,
         2.0567, -2.1651, -0.3991,  0.4793,  0.6906, -0.9383,  1.0436,  0.2185,
         0.7511,  0.8453, -0.5166, -0.8333,  0.2964,  2.7246, -1.1402, -1.1473,
         1.8944, -0.6893])
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
            torch.tensor([*self.sens_hist[-SENS_WINDOW:], self.pseudo_step]).float().view(1, 1, INPUT_COUNT)
        )
        self.sens_hist.append(detected)
        self.pseudo_step += 1

        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }