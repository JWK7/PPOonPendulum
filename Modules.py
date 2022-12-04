import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class PendulumNN(nn.Module):
#     def __init__(self,input,output,activation=nn.Tanh):
#         super(PendulumNN,self).__init__()
#         self.fc1=nn.linear(input,30)
#         self.fc2=nn.linear(30,30)
#         self.fc3=nn.linear(30,30)
#         self.fc4=nn.linear(30,output)
#         log_std = -0.5 * np.ones(output, dtype=np.float32)
#         self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#         self.act1 = activation

#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         vout = torch.exp(self.log_std)
#         return x,vout

class PastPendulumNN(nn.Module):
    def __init__(self,input,output,activation=nn.Tanh):
        super(PastPendulumNN,self).__init__()
        self.fc1=nn.Linear(input,30)
        self.fc2=nn.Linear(30,30)
        self.fc3=nn.Linear(30,30)
        self.fc4=nn.Linear(30,output)
        self.act1 = activation
        log_std = -0.5 * np.ones(output, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        vout = torch.exp(self.log_std)
        return x, vout


class NormalModule(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act1 = activation

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        return mout, vout


