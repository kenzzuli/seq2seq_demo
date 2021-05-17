"""
编码器
"""
import torch.nn as nn
import config


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=len(config.sen2seq),
                                  embedding_dim=config.embedding_dim,
                                  padding_idx=config.padding_index)
        self.gru = nn.GRU(input_size=config.embedding_dim, hidden_size=config.hidden_size,
                          num_layers=config.num_layers, batch_first=config.batch_first,
                          dropout=config.drop_out, bidirectional=config.bidirectional)

    def forward(self, input, input_length):
        """
        :param input: [batch_size, seq_len]
        :param input_length: batch_size
        :return hidden: [num_layers*num_directions, batch_size, hidden_size]
        :return output: [seq_len, batch_size, hidden_size*num_directions]
        """
        embed = self.embed(input)  # (batch_size, seq_len, embedding_dim)
        embed = embed.permute(1, 0, 2)  # （seq_len, batch_size, embedding_dim)
        # 打包
        embed = nn.utils.rnn.pack_padded_sequence(embed, lengths=input_length, batch_first=config.batch_first)
        output, hidden = self.gru(embed)
        # 解包 其实最后没有用到output，解包也毫无意义，只用了hidden
        output, output_length = nn.utils.rnn.pad_packed_sequence(output, batch_first=config.batch_first,
                                                                 padding_value=config.padding_index,
                                                                 total_length=config.seq_len)
        return output, hidden


if __name__ == '__main__':
    from dataset import get_dataloader

    train_dataloader = get_dataloader()
    encoder = Encoder()
    print(encoder)
    # Encoder(
    #   (embed): Embedding(14, 100, padding_idx=11)
    #   (gru): GRU(100, 128, num_layers=3, dropout=0.3)
    # )
    for feature, target, feature_length, target_length in train_dataloader:
        output, hidden = encoder(feature, feature_length)
        print(output.size())  # 这里的10是填充后的seq_len
        # torch.Size([10, 128, 128])
        print(hidden.size())
        # torch.Size([3, 128, 128])
        break
