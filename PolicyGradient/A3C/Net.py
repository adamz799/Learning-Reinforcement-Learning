import torch
from torch import nn, optim
import torch.nn.init as init
import torch.distributions as dist

# class Reshape(nn.Module):
#     def __init__(self):
#         super(Reshape, self).__init__()
        
#     def forward(self, inputs, shape = -1):
#         return inputs.view(shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential( #(batch_size,1,80,80) 
            nn.Conv2d(1, 16, (8,8), stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (4,4), stride=2),
            nn.ReLU(inplace=True)
        ) # 32x8x8
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 256),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(nn.Linear(256, 1))
        self.prob = nn.Sequential(
            nn.Linear(256,2),
            nn.Softmax()
        )
        for layers in self.children():
            for layer in layers.children():
                if isinstance(layer, nn.Conv2d):
                    init.normal_(layer.weight, std=0.01)
                    init.normal_(layer.bias, std = 0.01)
                elif isinstance(layer, nn.Linear):
                    init.normal_(layer.weight,std=0.01)
                    init.normal_(layer.bias, std = 0.01)
                    
    def forward(self, inputs):
        self.temp = self.fc(self.conv(inputs).view(-1))
        self.p = self.prob(self.temp)
        return dist.Categorical(self.p), self.value(self.temp)

