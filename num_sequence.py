"""
str类型的数字转成词表上的索引
即 文本转索引
"""


class Num2Seq(object):
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    SOS_TAG = "SOS"  # 开始
    EOS_TAG = "EOS"  # 结束

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

    def str2index(self, str_list, seq_len, add_eos=False):
        """
        训练时，特征值中不需要eos，目标值中需要eos
        如果不需要eos，返回的结果长度为seq_len
        如果需要eos，返回的结果长度为seq_len+1
        结构为 数字字符 + (EOS) + PAD
        """
        res_list = []
        if len(str_list) > seq_len:
            res_list += str_list[:seq_len]
            if add_eos:
                res_list += [self.EOS_TAG]
        else:
            res_list += str_list
            if add_eos:
                res_list += [self.EOS_TAG]
                res_list += [self.PAD_TAG] * (seq_len + 1 - len(res_list))
            else:
                res_list += [self.PAD_TAG] * (seq_len - len(res_list))
        # 如果词表中不存在某个token，将该token对应的索引换成UNK_TAG对应的索引
        res_list = [self.dict.get(token, self.dict[self.UNK_TAG]) for token in res_list]
        return res_list

    def index2str(self, index_list):
        # 如果词表中不存在某个索引，将该索引对应的token替换为UNK_TAG
        return [self.inverse_dict.get(index, self.UNK_TAG) for index in index_list]


if __name__ == '__main__':
    n2s = Num2Seq()
    print(n2s.dict)
    print(n2s.inverse_dict)
    str_list = ['2', '8', '7', '8', '1', '0', '7']
    index_list = n2s.str2index(str_list, seq_len=5, add_eos=False)
    print(index_list)
    new_str_list = n2s.index2str(index_list)
    print(new_str_list)
