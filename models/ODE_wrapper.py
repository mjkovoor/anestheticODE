import torch
import torch.nn as nn

class ODEWrapper(nn.Module):
    def __init__(self, ode_func, c_tensor, t_vector):
        super().__init__()
        self.ode_func = ode_func  # e.g., ODEFunc()
        self.c_tensor = c_tensor  # shape [B, T, 1]
        self.t_vector = t_vector  # shape [T]
        self.t_max = t_vector[-1].item()
        self.T = t_vector.shape[0]

    def forward(self, t_scalar, y_batch):
        t_val = t_scalar.item()
        t_idx = min(int((t_val / self.t_max) * (self.T - 1)), self.T - 1)
        c_t = self.c_tensor[:, t_idx, :]  # shape [B, 1]
        return self.ode_func(t_scalar, y_batch, c_t)