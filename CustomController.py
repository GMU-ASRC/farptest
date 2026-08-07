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
            g = torch.tensor([  3.6984,  10.4825,  -6.1933,  -4.0966,  -0.6312,  -7.0232, -14.4090,
         -1.6235,   2.2066,  -0.7314,   2.7778,   0.6396, -10.4244,   2.8760,
          5.3073,  -0.8869,  -4.8985,   5.0579,  -1.5696,  -1.0510,  -3.6404,
          1.6608,   5.8362,  -3.5595,  -1.3666,  -0.0415,   1.9167,  -3.1388,
          9.8562,   1.2687,  14.7881,   3.2982,   8.8745,  -7.4220,  -2.0406,
          0.5301,  -3.0271,   1.4108,   5.7679,  -5.4854,  -8.6641,  -2.8596,
         -5.0980,  -9.9118,   7.6231,   3.1581])
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