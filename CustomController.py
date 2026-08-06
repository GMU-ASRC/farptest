# from swarmsim.agent.control.AbstractController import AbstractController
# import numpy as np
# import torch
# import torch.nn as nn


# SPEED_LIMIT = 0.3
# TURN_LIMIT = 0.6

# class FARPNN(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

# class CustomController(AbstractController):
#     def __init__(self, agent=None, parent=None):
#         super().__init__(agent, parent)

#     def get_actions(self, agent):
#         v, w = 0, 0
#         return np.clip(v, -SPEED_LIMIT, SPEED_LIMIT), np.clip(w, -TURN_LIMIT, TURN_LIMIT)  # DO NOT CHANGE THIS LINE

from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

VMAX, WMAX = SPEED_LIMIT, TURN_LIMIT
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
INPUT_COUNT = 1


class FarpRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=4, batch_first=True)
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x, h0=None):
        rnn_out, hn = self.rnn(x, h0)
        final_out = self.linear(rnn_out)
        return final_out, hn


class CustomController(AbstractController):
    def __init__(self, genome=None, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.model = FarpRNN()

        if genome is None:
            raise NotImplementedError()
        else:
            if isinstance(genome[0], str):
                g = torch.from_numpy(np.fromstring(genome[0], sep=' ')).float()
            else:
                g = torch.asarray(genome).float()

        vector_to_parameters(g, self.model.parameters())

    @torch.inference_mode()
    def get_actions(self, agent):
        detected = float(agent.sensors[self.sensor_id].current_state != 0)
        output, _ = self.model(
            torch.tensor(detected).float().view(1, 1, INPUT_COUNT)
        )

        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }