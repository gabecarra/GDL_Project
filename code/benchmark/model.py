from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import GCNDecoder
import torch


class TimeThenSpaceModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_layers,
                 gcn_layers,
                 horizon):
        super(TimeThenSpaceModel, self).__init__()

        self.input_encoder = torch.nn.Linear(input_size, hidden_size)

        self.encoder = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers)

        self.decoder = GCNDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            horizon=horizon,
            n_layers=gcn_layers
        )

    def forward(self, x, edge_index, edge_weight):
        # x: [batches steps nodes channels]
        x = self.input_encoder(x)

        x = self.encoder(x, return_last_state=True)

        return self.decoder(x, edge_index, edge_weight)