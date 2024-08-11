import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc_b = nn.Linear(width, width, bias=False)
        self.fc_f = nn.Linear(width, width, bias=False)
        self.g_b = nn.Linear(width, input_dim)
        self.g_f = nn.Linear(width, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        theta_b = self.fc_b(x)
        theta_f = self.fc_f(x)

        backcast = self.g_b(theta_b)
        forecast = self.g_f(theta_f)

        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(self, n_blocks, input_dim, output_dim, width):
        super().__init__()
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = NBeatsBlock(input_dim, output_dim, width)
            self.blocks.append(block)

    def forward(self, x):
        stack_forecast = []
        for i in range(self.n_blocks):
            backcast, forecast = self.blocks[i](x)
            x = x - backcast
            stack_forecast.append(forecast)
        stack_forecast = torch.stack(stack_forecast, axis=-1)
        stack_forecast = torch.sum(stack_forecast, axis=-1)
        stack_residual = x
        return stack_residual, stack_forecast


class NBeats(nn.Module):
    def __init__(self, n_blocks=2, n_stacks=4, input_dim=590, output_dim=1, width=1024):
        super().__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = NBeatsStack(n_blocks, input_dim, output_dim, width)
            self.stacks.append(stack)

    def forward(self, x):
        x = x.view(len(x), 590)
        global_forecast = []
        for i in range(self.n_stacks):
            stack_residual, stack_forecast = self.stacks[i](x)
            x = stack_residual
            global_forecast.append(stack_forecast)
        global_forecast = torch.stack(global_forecast, axis=-1)

        global_forecast = global_forecast[0:280]

        global_forecast = torch.sum(global_forecast, axis=-1)
        return global_forecast
