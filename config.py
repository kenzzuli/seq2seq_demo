from sentence2sequence import Sen2Seq
import torch

# 实例化一个sen2seq
sen2seq = Sen2Seq()

# dataset相关参数
seq_len = 10

# dataloader相关参数
train_batch_size = 512
test_batch_size = 512
drop_last = False

# 定义编码器相关参数
embedding_dim = 100
padding_index = sen2seq.dict[sen2seq.PAD_TAG]
sos_index = sen2seq.dict[sen2seq.SOS_TAG]
eos_index = sen2seq.dict[sen2seq.EOS_TAG]
pad_index = sen2seq.dict[sen2seq.PAD_TAG]
hidden_size = 128
num_layers = 3
batch_first = False
drop_out = 0.3
bidirectional = False
num_directions = 2 if bidirectional else 1

# 解码器相关参数
teacher_forcing_ratio = 0.5  # 范围（0,1)，0是不使用，1是完全使用

# 训练
EPOCHS = 1
model_path = "./s2s_model/s2s.pkl"
optimizer_path = "./s2s_model/optim.pkl"

# gpu运行
# 实例化device  实验室有两块gpu，第一块cuda:0正在用，所以指定gpu为cuda:1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
