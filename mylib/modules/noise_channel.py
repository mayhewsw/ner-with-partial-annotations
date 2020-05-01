from torch import nn
import math
from torch.nn import Parameter
import torch


class Channel(nn.Module):
    """
    Taken from:
    https://discuss.pytorch.org/t/pyotrch-equivalent-of-noise-adaptation-layer-keras-code/19338
    """
    def __init__(self, input_dim, bias=False, *argv):
        super(Channel, self).__init__()
        self.input_dim = input_dim
        self.activation2 = nn.Softmax(dim=2)
        self.activation1 = nn.Softmax(dim=1)

        if len(argv) == 0:
            # construct the proper layer as it is not initialized
            # from some previously learned models
            self.weight = Parameter(torch.Tensor(input_dim, input_dim))
            if bias:
                self.bias = Parameter(torch.Tensor(input_dim))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        else:
            # use the pre-initialized weights
            self.weight = Parameter(argv[0])
            if bias:
                self.bias = Parameter(torch.zeros(input_dim))
            else:
                self.register_parameter('bias', None)

    def forward(self, x):
        # ensure that the weights are probabilities
        channel_matrix = torch.softmax(self.weight, dim=1)
        #channel_matrix = self.weight
        # ensure that the inputs are probabilities
        # x shape is: (batch_size, max_seq_len, num_classes)
        prob_x = torch.nn.functional.softmax(x, dim=-1)

        #channel_matrix = torch.eye(self.input_dim, self.input_dim).cuda()


        # I don't want the shape to change.
        #
        # print()
        # print(x.shape)
        #print(channel_matrix)
        #
        # instance = x[0]
        # print(instance)

        #torch.matmul(prob_x, channel_matrix)

        # multiply together
        return torch.matmul(prob_x, channel_matrix)



    def reset_parameters(self):
        n = self.input_dim
        stdv = 1. / math.sqrt(n)
        #self.weight.data.uniform_(-stdv, stdv)
        self.weight.data = torch.ones(self.input_dim, self.input_dim)
        self.weight.data[:, 0] = 0

        #self.weight.data[:, 1] = 1

        #if self.bias is not None:
        #    self.bias.data.uniform_(-stdv, stdv)