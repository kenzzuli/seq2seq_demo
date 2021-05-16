"""
解码器
"""
import torch.nn as nn
import torch
import config
import torch.nn.functional as F
import random


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.sen2seq),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.padding_index)
        self.gru = nn.GRU(input_size=config.embedding_dim, hidden_size=config.hidden_size,
                          num_layers=config.num_layers, batch_first=config.batch_first,
                          dropout=config.drop_out, bidirectional=config.bidirectional)
        # 经过全连接层，将[batch_size, hidden_size*num_directions] 转成 [batch_size, vocab_size]
        self.fc = nn.Linear(in_features=config.hidden_size * config.num_directions, out_features=len(config.sen2seq))

    def forward(self, encoder_hidden, target):
        """
        :param encoder_hidden: [num_layers*num_directions, batch_size, hidden_size]
        :param target: [batch_size, seq_len+1] 构造数据集时指定长度为seq_len+1, 做teacher forcing
        :return: outputs: [seq_len, batch_size, vocab_size]
        """

        # 1. 接收encoder的hidden_state作为decoder第一个时间步的hidden_state
        decoder_hidden = encoder_hidden
        # 2. 构造第一个时间步的输入 形状为[batch_size, 1]，全为SOS
        batch_size = encoder_hidden.size(1)
        # decoder_input = torch.LongTensor(torch.ones([batch_size, 1]) * config.sos_index)
        decoder_input = torch.LongTensor([[config.sos_index]] * batch_size).to(config.device)

        # 保存所有的结果，需要用outputs和target计算损失
        outputs = []  # outputs最后的形状是 [seq_len, batch_size, vocab_size]
        for i in range(config.seq_len + 1):
            # 3. 获取第一个时间步的decoder_output，形状为[batch_size, vocab_size]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            # 4. 计算第一个decoder_output，得到最后的输出结果 形状为[batch_size, 1]
            # 在训练中，是否使用teacher forcing教师纠偏
            use_teacher_forcing = random.random() < config.teacher_forcing_ratio
            if use_teacher_forcing:
                # 下一次的输入使用真实值
                decoder_input = target[:, i].unsqueeze(1)  # [batch_size,1]
            else:
                # 获取最后一个维度中最大值所在的位置，即确定是哪一个字符，以此作为下一个时间步的输入
                # 模型最开始也不知道哪个位置对应哪个字符，通过训练，慢慢调整，才知道的
                decoder_input = torch.argmax(decoder_output, dim=-1, keepdim=True)  # [batch_size, 1]
            # 5. 把前一次的hidden_state作为当前时间步的hidden_state，把前一次的输出作为当前时间步的输入
            # 6. 循环4-5
        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size, vocab_size]
        return outputs

    def forward_step(self, decoder_input, decoder_hidden):
        """
        计算每个时间步的结果
        :param decoder_input [batch_size,1]
        :param decoder_hidden [num_layers*num_directions, batch_size, hidden_size]
        :return output [batch_size, vocab_size]
        :return decoder_hidden 形状同上面的decoder_hidden
        """
        decoder_input_embed = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim]
        decoder_input_embed = decoder_input_embed.permute(1, 0, 2)  # [1, batch_size, embedding_dim]
        output, decoder_hidden = self.gru(decoder_input_embed, decoder_hidden)
        # Output: [1, batch_size, hidden_size*num_directions]
        # decoder_hidden: [num_layers*num_directions, batch_size, hidden_size]
        # 将output的第0个维度去掉
        output = output.squeeze(0)  # [batch_size, hidden_size*num_directions]
        output = self.fc(output)  # [batch_size, vocab_size]
        output = F.log_softmax(output, dim=-1)  # 取概率
        return output, decoder_hidden

    def evaluation(self, encoder_hidden):
        """
        模型评估时调用
        :param encoder_hidden: [num_direction*num_layers, batch_size, hidden_size]
        :return outputs: [seq_len, batch_size, vocab_size]
        """
        decoder_hidden = encoder_hidden  # 获取encoder_hidden作为初始的decoder_hidden
        batch_size = encoder_hidden.size(1)  # 获取batch_size,构造初始的decoder_input
        decoder_input = torch.LongTensor([[config.sos_index]] * batch_size).to(config.device)
        outputs = []
        for i in range(config.seq_len + 1):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=-1, keepdim=True)
        outputs = torch.stack(outputs, dim=0)
        return outputs
