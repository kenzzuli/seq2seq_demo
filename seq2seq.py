"""
把encoder和decoder合并，得到seq2seq模型
"""
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, input_length, target):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        outputs = self.decoder(encoder_hidden, target)
        return outputs

    def evaluate(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        outputs = self.decoder.evaluation(encoder_hidden)
        return outputs
