import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_layers, output):
        super(MLP, self).__init__()
        # Parameters
        input_features = [in_channels] + hidden_layers
        output_features = hidden_layers + [output]

        # Model Structure
        self.seq_layers = nn.Sequential()
        for idx, (i, o) in enumerate(zip(input_features, output_features)):
            self.seq_layers.add_module(f'linear_{idx}', nn.Linear(i, o))
            if idx != len(hidden_layers):
                self.seq_layers.add_module(f'activation_{idx}', nn.ReLU(inplace=True))

    def forward(self, x):
        return F.log_softmax(self.seq_layers(x), dim=1)
