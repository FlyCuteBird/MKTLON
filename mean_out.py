import torch
import torch.nn as nn
from collections import OrderedDict
class MeanOut(nn.Module):
    def __init__(self, input_size, output_size, num_linear):
        super(MeanOut, self).__init__()
        # initialize variable
        self.input_size = input_size
        self.num_linear = num_linear
        self.output = output_size
        self.mean = nn.ModuleList([nn.Linear(input_size, output_size) for i in range(num_linear)])
        self.init_parameters()

    def init_parameters(self):
        for i in range(self.num_linear):
            # weight norm
            nn.utils.weight_norm(self.mean[i])
            # # Xavier initialization
            # r = np.sqrt(6.)/np.sqrt(self.mean[i].in_features + self.mean[i].out_features)
            # self.mean[i].weight.data.uniform_(-r, r)
            # self.mean[i].bias.data.fill_(0)
    def forward(self, input):
        meanList= [self.mean[i](input) for i in range(self.num_linear)]
        meanout = torch.cat([i.unsqueeze(0) for i in meanList], dim=0)
        result = torch.sum(meanout, dim=0)
        return result / self.num_linear

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(MeanOut, self).load_state_dict(new_state)

# test = torch.FloatTensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
# print(test.shape)
#
# model = MeanOut(4, 5, 2)
#
# result = model(test)
#
# print(result.shape)
# print(result)






