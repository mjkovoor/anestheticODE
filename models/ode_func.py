import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        # self.net = nn.Sequential(
        #     nn.Linear(2, hidden_dim),
        #     nn.Tanh(),  # I keep alternating between tanh and ReLU here, still figuring out what adapts best
        #     nn.Linear(hidden_dim, 1)
        # )

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)

    def forward(self, t, y, c_t):
        # t: scalar, y: [B, 1], c_t: [B, 1]
        t_input = torch.ones_like(c_t) * t  # broadcast t across batch
        inp = torch.cat([c_t, t_input], dim=1)  # shape [B, 2]
        dy_dt = self.net(inp)
        return dy_dt
