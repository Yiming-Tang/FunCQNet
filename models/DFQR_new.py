import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.neurons = configs.neurons
        self.dropout = configs.dropout
        self.activation_name = configs.activation
        self.p = configs.p
        self.q = configs.q

        activation_dict = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            'identity': nn.Identity()  
        }
        self.act = activation_dict[self.activation_name.lower()]

        self.beta = nn.Parameter(torch.randn(1, 1))


        self.layers = nn.ModuleList()
        input_dim = self.q + 1  
        for hidden_dim in self.neurons:
            self.layers.append(
                self._make_residual_block(input_dim, hidden_dim)
            )
            input_dim = hidden_dim  
            
        self.out = nn.Linear(self.neurons[-1], self.p)

    def _make_residual_block(self, in_dim, out_dim):
        block = nn.ModuleDict({
            "linear": nn.Linear(in_dim, out_dim),
            "activation": self.act,
            "dropout": nn.Dropout(self.dropout),
            "proj": nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        })
        return block

    def forward_residual_block(self, block, x):
        identity = x
        out = block["linear"](x)
        out = block["activation"](out)
        out = block["dropout"](out)

        if block["proj"] is not None:
            identity = block["proj"](identity)

        return out + identity

    def forward(self, I, X, t_values, Z):
        B, m, _ = X.shape
        beta0 = I @ self.beta

        inp_alpha = torch.cat(
            [t_values.unsqueeze(-1), Z.unsqueeze(1).repeat(1, m, 1)],
            dim=-1
        ).view(-1, self.q + 1)

        x = inp_alpha
        for block in self.layers:
            x = self.forward_residual_block(block, x)

        alpha = self.out(x).view(B, m, self.p)

        integral = torch.trapz(X * alpha, t_values.unsqueeze(-1), dim=1).sum(dim=1, keepdim=True)
        y_hat = (beta0 + integral).unsqueeze(-1)

        return y_hat, self.beta, alpha