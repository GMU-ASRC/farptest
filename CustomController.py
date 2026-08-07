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
            # g = torch.tensor([
            #      0.9843,  -0.0630,  -0.8113,   6.0496,   6.2267,   1.1070,  -0.0827,
            #     -0.0703,   0.2808,   6.0814,  -2.7982,  -4.3426,  -1.8812,   1.9555,
            #      3.8920,  -0.1906,   2.3481,  -2.3552,   3.3442,  -8.4039,   1.5242,
            #     -2.5292,  -4.9531,  -1.6374,  -4.0107,  -2.3715,   2.1547,  -1.4259,
            #     -3.2585,  -4.7903, -10.0326,  -2.2905,  -2.3988,   1.3420,   5.2107,
            #     -0.2031,   3.9850,  -3.8695,   0.1669,  -2.9956,   2.0178,   0.1494,
            #      2.6730,  -2.4851,   3.7443,  -3.8775
            # ])
            raise NotImplementedError()
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