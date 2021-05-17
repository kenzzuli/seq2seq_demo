"""
准备dataset dataloader
"""
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config
import torch


class NumDataset(Dataset):
    def __init__(self, seed=6):
        # 使用numpy随机造一堆数字
        np.random.seed(seed)
        # 随机数的范围是[0, 1e8) 数量为50000个
        self.data = np.random.randint(0, 1e8, [50000])

    def __getitem__(self, index):
        # 将int转成str，并使用列表存放
        # 40887177 --> ['4', '0', '8', '8', '7', '1', '7', '7']
        feature = list(str(self.data[index]))
        target = feature + ["0"]
        feature_length = len(feature)
        target_length = len(target)
        # 将feature和target转成序列
        feature = config.sen2seq.sen2seq(feature, seq_len=config.seq_len)
        target = config.sen2seq.sen2seq(target, seq_len=config.seq_len, add_eos=True)
        return feature, target, feature_length, target_length

    def __len__(self):
        return len(self.data)


def get_dataloader(train=True):
    train_dataset = NumDataset(6)
    test_dataset = NumDataset(10)
    if train:
        return DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, collate_fn=collate_fn,
                          shuffle=True, drop_last=config.drop_last)
    else:
        return DataLoader(dataset=test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn,
                          shuffle=True, drop_last=config.drop_last)


def collate_fn(batch):
    """
    定义整理函数
    :param batch: [(feature, target, feature_length, target_length), (feature, target, feature_length, target_length), ...]
    :return:
    """
    # 先对batch依据target_length从大到小排序
    batch = sorted(batch, key=lambda x: x[3], reverse=True)
    # 先对batch拆包，变成多个元组对象，然后从每个元组中取第一个元素组成元组作为zip的第一个元素，相当于矩阵的转置
    feature, target, feature_length, target_length = zip(*batch)
    feature = torch.LongTensor(feature)
    target = torch.LongTensor(target)
    feature_length = torch.LongTensor(feature_length)
    target_length = torch.LongTensor(target_length)
    return feature, target, feature_length, target_length


if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    for feature, target, feature_length, target_length in train_dataloader:
        print(feature)
        # for i in feature:
        #     print(config.sen2seq.index2str(i))
        print(target.size())
        # for i in target:
        #     print(config.sen2seq.index2str(i))
        print(feature_length)
        print(target_length)
        break
