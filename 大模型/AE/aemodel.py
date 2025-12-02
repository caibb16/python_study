import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class AutoEncoderModel(nn.Module):
    def __init__(self, dim, layer_num):
        super(AutoEncoderModel, self).__init__()
        self.encoder = Encoder(dim, layer_num=layer_num)
        self.decoder = Decoder(dim, layer_num=layer_num)

    def forward(self, inputs):
        latent = self.encoder(inputs)
        reconstruct_img = self.decoder(latent)
        return reconstruct_img