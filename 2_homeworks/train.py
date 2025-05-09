import json
from torch.utils.data import DataLoader, TensorDataset
from dataset import collate_fnn, collate_seq
import torch.nn as nn
import torch
from models.FNNLM import FNNLM
from models.RNNLM import RNNLM
from models.TransformersLM import TransformerLM
from train_utils import train_model
from configs.params import TrainConfig
from functools import partial

with open('tokenizer.json', 'r') as f:
    token2idx = json.load(f)
with open('data/datasets/fnn_train_pairs.json', 'r') as f:
    fnn_train_pairs = json.load(f)
with open('data/datasets/fnn_val_pairs.json', 'r') as f:
    fnn_val_pairs = json.load(f)
with open('data/datasets/train_sequences.json', 'r') as f:
    train_sequences = json.load(f)
with open('data/datasets/val_sequences.json', 'r') as f:
    val_sequences = json.load(f)

config = TrainConfig()

PAD_TOKEN = "<PAD>"
PAD_ID = token2idx[PAD_TOKEN]

vocab_size = len(token2idx)

# DataLoader for RNN/Transformer
train_loader_seq = DataLoader(train_sequences, batch_size=config.seq_batch_size, shuffle=True, collate_fn=partial(collate_seq, pad_id=PAD_ID))
val_loader_seq = DataLoader(val_sequences, batch_size=config.seq_batch_size, shuffle=False, collate_fn=partial(collate_seq, pad_id=PAD_ID))

train_loader_fnn = DataLoader(fnn_train_pairs, batch_size=config.fnn_batch_size, shuffle=True, collate_fn=collate_fnn)
val_loader_fnn = DataLoader(fnn_val_pairs, batch_size=config.fnn_batch_size, shuffle=False, collate_fn=collate_fnn)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)


# 设置设备为GPU (如果可用)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # macOS MPS backend
    device = torch.device("mps")
else:
    # CPU
    device = torch.device("cpu")
print("Using device:", device)


# 初始化模型实例
model_fnn = FNNLM(vocab_size=vocab_size, embed_size=config.embed_size, hidden_size=config.hidden_size, context_size=config.context_size)
model_rnn = RNNLM(vocab_size=vocab_size, embed_size=config.embed_size, hidden_size=config.hidden_size, num_layers=config.rnn_num_layers)
model_trans = TransformerLM(vocab_size=vocab_size, d_model=config.embed_size, nhead=config.nhead, dim_feedforward=config.dim_feedforward, num_layers=config.transformer_num_layers)



print("训练 FNN 模型...")
model_fnn = train_model(model_fnn, train_loader_fnn, val_loader_fnn, config, device, vocab_size, criterion, PAD_ID)
torch.save(model_fnn.state_dict(), 'checkpoints/fnn_model.pth')
print("FNN 模型训练完成，保存模型参数 fnn_model.pth")

print("\n训练 RNN 模型...")
model_rnn = train_model(model_rnn, train_loader_seq, val_loader_seq, config, device, vocab_size, criterion, PAD_ID)
torch.save(model_rnn.state_dict(), 'checkpoints/rnn_model.pth')
print("RNN 模型训练完成，保存模型参数 rnn_model.pth")

print("\n训练 Transformer 模型...")
model_trans = train_model(model_trans, train_loader_seq, val_loader_seq, config, device, vocab_size, criterion, PAD_ID)
torch.save(model_trans.state_dict(), 'checkpoints/transformer_model.pth')
print("Transformer 模型训练完成，保存模型参数 transformer_model.pth")
print("所有模型训练完成！")




