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
            'leakyrelu': nn.LeakyReLU()
        }
        self.act = activation_dict[self.activation_name.lower()]
    
        self.beta = nn.Parameter(torch.randn(1, 1))  
    
        self.input_proj = nn.Linear(1, self.neurons[0])
        layers = []
        for i in range(len(self.neurons) - 1):
            layers.append(nn.Linear(self.neurons[i], self.neurons[i+1]))
            layers.append(self.act)
            layers.append(nn.Dropout(self.dropout))
        self.resblock = nn.Sequential(*layers)
        self.out = nn.Linear(self.neurons[-1], self.p)

    def forward(self, I, X, t_values, Z):
        B, m, _ = X.shape
        q = Z.shape[1]
        beta0 = I @ self.beta
    

        t_exp = t_values
        inp_alpha = t_exp.unsqueeze(-1).view(-1, 1)
    
        x = self.input_proj(inp_alpha)
        x = self.act(x)
        x = self.resblock(x) + x
        alpha = self.out(x).view(B, m, self.p)
        integral = torch.trapz(X * alpha, t_values.unsqueeze(-1), dim=1).sum(dim=1, keepdim=True)
        y_hat = (beta0 + integral).unsqueeze(-1)
        return y_hat, self.beta, alpha