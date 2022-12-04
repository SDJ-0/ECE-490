import torch
from torch import nn


class SVDRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k1=1, k2=1, device='cuda'):
        super(SVDRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k1 = k1
        self.k2 = k2
        self.device = device
        self.in2hidden = nn.Linear(input_size, hidden_size).to(device)
        self.hidden2out = nn.Linear(hidden_size, output_size).to(device)
        self.us = nn.ParameterList([torch.nn.Parameter(torch.ones(i, dtype=torch.float32, device=device)) for i in
                                    range(k1, hidden_size + 1)])
        self.sigmas = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32, device=device))
        self.vs = nn.ParameterList([torch.nn.Parameter(torch.ones(i, dtype=torch.float32, device=device)) for i in
                                    range(k2, hidden_size + 1)])

    def W_SVD(self):
        u_hats = torch.zeros((self.hidden_size - self.k1 + 1, self.hidden_size), dtype=torch.float32,
                             device=self.device)
        indices = [torch.arange(0, i) for i in range(self.k1, self.hidden_size + 1)]
        for i in range(len(indices)):
            u_hats[i, indices[i]] = self.us[i]
        u_hats = torch.flip(u_hats, dims=[1])
        U = torch.eye(self.hidden_size, dtype=torch.float32, device=self.device)
        for u_hat in u_hats:
            U = (torch.eye(self.hidden_size, dtype=torch.float32, device=self.device) -
                 2 / torch.inner(u_hat, u_hat) * torch.outer(u_hat, u_hat)) @ U

        v_hats = torch.zeros((self.hidden_size - self.k2 + 1, self.hidden_size), dtype=torch.float32,
                             device=self.device)
        indices = [torch.arange(0, i) for i in range(self.k2, self.hidden_size + 1)]
        for i in range(len(indices)):
            v_hats[i, indices[i]] = self.vs[i]
        v_hats = torch.flip(v_hats, dims=[0, 1])
        V = torch.eye(self.hidden_size, dtype=torch.float32, device=self.device)
        for v_hat in v_hats:
            V = V @ (torch.eye(self.hidden_size, dtype=torch.float32, device=self.device)
                     - 2 / torch.inner(v_hat, v_hat) * torch.outer(v_hat, v_hat))

        return U @ torch.diag(self.sigmas) @ V

    def forward(self, x):
        batch_size, time = x.shape[:2]
        hidden_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=self.device)
        output = torch.empty((batch_size, time, self.output_size), device=self.device)
        for t in range(time):
            hidden_state = torch.sigmoid(self.in2hidden(x[:, t]) + hidden_state @ self.W_SVD().T)
            output[:, t] = self.hidden2out(hidden_state)
        return output

    def control_sigma(self, r=0.01, sigma_star=1):
        with torch.no_grad():
            self.sigmas.copy_(2 * r * (torch.sigmoid(self.sigmas) - 0.5) + sigma_star)
