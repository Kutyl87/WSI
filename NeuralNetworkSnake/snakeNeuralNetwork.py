import torch.nn as nn


class SnakeNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 16)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(16, 4)
        self.act_output = nn.Softmax()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        # x = self.act3(self.hidden3(x))
        # x = self.act4(self.hidden4(x))
        x = self.act_output(self.output(x))
        return x
