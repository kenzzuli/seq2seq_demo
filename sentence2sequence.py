"""
句子转序列
"""


class Sen2Seq(object):
    UNK_TAG = "UNK"  # 未知
    PAD_TAG = "PAD"  # 填充
    SOS_TAG = "SOS"  # 句子开始
    EOS_TAG = "EOS"  # 句子结束

    def __init__(self):
        # 构造str->index的字典
        self.dict = {}
        vocab = list(str(i) for i in range(10))  # 所有的数字
        vocab += [self.UNK_TAG, self.PAD_TAG, self.SOS_TAG, self.EOS_TAG]  # 四个原始字符
        for token in vocab:
            self.dict[token] = len(self.dict)
        # 构造index->str的字典
        # self.inverse_dict = {value: key for key, value in self.dict.items()} # 两种方法都行
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def __len__(self):
        return len(self.dict)

    def sen2seq(self, sentence: list, seq_len: int, add_eos=False) -> list:
        """
        将句子转成序列
        训练时，特征值中不需要eos，目标值中需要eos
        如果不需要eos，返回的结果长度为seq_len
        如果需要eos，返回的结果长度为seq_len+1
        结构为 数字字符 + (EOS) + PAD
        :param sentence: 单词组成的列表，即句子
        :param seq_len: 指定序列长度，可能要对句子进行删减或填充
        :param add_eos: 是否添加eos标记
        """
        res_list = []
        if len(sentence) > seq_len:  # 如果句子长度大于seq_len，裁剪
            res_list += sentence[:seq_len]
            if add_eos:
                res_list += [self.EOS_TAG]
        else:
            res_list += sentence
            if add_eos:
                res_list += [self.EOS_TAG]
                res_list += [self.PAD_TAG] * (seq_len + 1 - len(res_list))  # 如果句子长度小于seq_len，填充
            else:
                res_list += [self.PAD_TAG] * (seq_len - len(res_list))
        # 如果词表中不存在某个token，将该token对应的索引换成UNK_TAG对应的索引
        res_list = [self.dict.get(token, self.dict[self.UNK_TAG]) for token in res_list]
        return res_list

    def seq2sen(self, sequence: list) -> list:
        """
        将序列转成句子
        :param sequence: 序列
        :return: 句子
        """
        # 如果词表中不存在某个索引，将该索引对应的token替换为UNK_TAG
        return [self.inverse_dict.get(index, self.UNK_TAG) for index in sequence]


if __name__ == '__main__':
    n2s = Sen2Seq()
    print(n2s.dict)
    print(n2s.inverse_dict)
    str_list = ['2', '8', '7', '8', '1', '0', '7']
    index_list = n2s.sen2seq(str_list, seq_len=5, add_eos=False)
    print(index_list)
    new_str_list = n2s.seq2sen(index_list)
    print(new_str_list)
