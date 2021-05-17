import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
from dataset import get_dataloader
import config
from seq2seq import Seq2Seq

# 使用gpu训练
s2s = Seq2Seq().to(config.device)
optimizer = torch.optim.Adam(s2s.parameters())

# 模型加载
if os.path.exists(config.model_path) and os.path.exists(config.optimizer_path):
    s2s.load_state_dict(torch.load(config.model_path, map_location=config.device))
    optimizer.load_state_dict(torch.load(config.optimizer_path, map_location=config.device))


def train():
    train_dataloader = get_dataloader(train=True)
    for i in range(config.EPOCHS):
        with tqdm(total=len(train_dataloader)) as t:
            for index, (feature, target, feature_length, target_length) in enumerate(train_dataloader):
                t.set_description("Epoch %i/%i" % (i + 1, config.EPOCHS))  # 设置描述
                optimizer.zero_grad()  # 梯度置为0
                feature = feature.to(config.device)
                target = target.to(config.device)
                feature_length = feature_length.to(config.device)
                target_length = target_length.to(config.device)
                y_predict = s2s(feature, feature_length, target)  # [seq_len, batch_size, vocab_size]
                y_predict = y_predict.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
                # 计算损失前，要将 3阶和2阶 变成 2阶和1阶
                y_predict = y_predict.reshape(y_predict.size(0) * y_predict.size(1),
                                              y_predict.size(2))  # [batch_size*seq_len, vocab_size]
                # print(y_predict.size(), y_predict.type())
                target = target.reshape(target.size(0) * target.size(1))  # [batch_size*seq_len]
                # print(target.size(), target.type())
                loss = F.nll_loss(y_predict, target)
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())  # 设置后缀
                t.update(1)  # 手动更新进度条
                if index % 10 == 0:  # 每10个batch保存一次模型
                    torch.save(s2s.state_dict(), config.model_path)
                    torch.save(optimizer.state_dict(), config.optimizer_path)


def eval():
    s2s.eval()  # 进入评估模式
    test_dataloader = get_dataloader(train=False)
    loss = 0
    correct = 0
    with tqdm(total=len(test_dataloader)) as t:
        for index, (feature, target, feature_length, target_length) in enumerate(test_dataloader):
            t.set_description("Evaluation")  # 设置描述
            with torch.no_grad():
                feature = feature.to(config.device)
                target = target.to(config.device)
                feature_length = feature_length.to(config.device)
                target_length = target_length.to(config.device)
                y_predict = s2s.evaluate(feature, feature_length)  # [seq_len, batch_size, vocab_size]
                y_predict = y_predict.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
                # 准确率
                pred = y_predict.argmax(dim=-1)  # [batch_size, seq_len]
                # eq会生成[batch_size, seq_len]的布尔矩阵 all是只有一行全为true时，才会返回true，形状为[batch_size]
                correct += pred.eq(target).all(dim=-1).sum()
                # 损失
                # 计算损失前，要将 3阶和2阶 变成 2阶和1阶
                y_predict = y_predict.reshape(y_predict.size(0) * y_predict.size(1),
                                              y_predict.size(2))  # [batch_size*seq_len, vocab_size]
                target = target.reshape(target.size(0) * target.size(1))  # [batch_size*seq_len]
                loss += F.nll_loss(y_predict, target, reduction="sum")
                t.update(1)  # 手动更新进度条
        loss /= len(test_dataloader.dataset)
        acc = 100.0 * correct / len(test_dataloader.dataset)
        print("Avg Loss:{}\tAccuracy:{}%".format(loss, acc))


if __name__ == '__main__':
    train()
    eval()
