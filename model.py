from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn
import torch.nn.functional as F


class QModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 full_obs_space=None, hidden_size=32):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if full_obs_space is not None:
            obs_space = full_obs_space

        obs_shape = obs_space["obs"].shape
        info_shape = obs_space["info"].shape

        self.info_encoder = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(info_shape)), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.convs = nn.Sequential(
            nn.Conv2d(obs_shape[2] + hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )


        self.policy_layers = nn.Sequential(
            nn.Linear(obs_shape[0] * obs_shape[1] * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        self.value_layers = nn.Sequential(
            nn.Linear(obs_shape[0] * obs_shape[1] * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.value = None

    def forward(self, input_dict, state, seq_lens):

        grid_size = input_dict["obs"]["obs"].shape[1]

        info = input_dict["obs"]["info"]  # B, O, M, 2
        info = torch.flatten(info, start_dim=1)  # B, O*M*2
        info = self.info_encoder(info)  # B, H
        info = torch.tile(info, (grid_size, grid_size, 1, 1))  # S, S, B, H
        info = torch.permute(info, (2, 0, 1, 3))

        obs = input_dict["obs"]["obs"]  # B, S, S, O+2
        obs = torch.cat((obs, info), dim=-1)  # B, S, S, O+2+H
        obs = torch.permute(obs, (0, 3, 1, 2))  # B, O+2+H, S, S
        obs = self.convs(obs)  # B, H, S, S
        obs = torch.flatten(obs, start_dim=1)  # B, S*S*H

        policy = self.policy_layers(obs)  # B, A
        self.value = self.value_layers(obs)  # B, 1

        return policy, state

    def value_function(self):
        return self.value.squeeze(1)
