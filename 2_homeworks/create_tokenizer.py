import io
import math
import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset
import json

# 1. 读取数据集文件
train_lines = []
val_lines = []
data_path = "data/raw_data/news.2017.zh.shuffled.deduped"  # 假设已将语料下载解压为此文本文件
with io.open(data_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]
# 将最后1000句作为验证集，其余作为训练集
if len(lines) > 1000:
    train_lines = lines[:-1000]
    val_lines = lines[-1000:]
else:
    # 如果总行数不足1000，则取最后10%作为验证集（备用策略）
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
print(f"总句子数: {len(lines)}, 训练集: {len(train_lines)}, 验证集: {len(val_lines)}")

# 2. 使用 jieba 对每个句子分词
#    对于中文，jieba.lcut 将句子切分为词语列表；对于英文词或数字会原样保留
train_tokens = [jieba.lcut(line) for line in train_lines]
val_tokens = [jieba.lcut(line) for line in val_lines]

# 3. 构建词汇表（Vocabulary）
#    统计训练集中的词频，建立词到索引的映射
from collections import Counter
token_counter = Counter()
for tokens in train_tokens:
    token_counter.update(tokens)

# 特殊标记
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# 初始化词表，确保特殊符号有固定索引
vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]  # 索引0,1,2,3分别留给PAD, BOS, EOS, UNK
# 将训练集中出现的token按频率加入词表
for token, freq in token_counter.items():
    # 跳过特殊标记（如果语料中有可能出现这些字符串，可以在此过滤）
    if token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        continue
    vocab.append(token)
# 建立token到索引的字典
token2idx = {token: idx for idx, token in enumerate(vocab)}
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

# 4. 将训练和验证语料转换为ID序列
#    在每个句子开头加<BOS>，结尾加<EOS>，并用<UNK>替换不在词表中的词
bos_id = token2idx[BOS_TOKEN]
eos_id = token2idx[EOS_TOKEN]
unk_id = token2idx[UNK_TOKEN]

train_sequences = []
for tokens in train_tokens:
    seq = [bos_id]  # 起始标记
    for token in tokens:
        seq.append(token2idx.get(token, unk_id))
    seq.append(eos_id)  # 结束标记
    train_sequences.append(seq)

val_sequences = []
for tokens in val_tokens:
    seq = [bos_id]
    for token in tokens:
        # 验证集中的词如果不在词表中，记为UNK
        seq.append(token2idx.get(token, unk_id))
    seq.append(eos_id)
    val_sequences.append(seq)

# 5. 为 FNN 模型生成固定窗口大小的训练样本 (context, target) 对
context_size = 5  # 定义前文窗口长度N（使用N个token预测下一个token）
fnn_train_pairs = []   # 每项为(tuple: (context_token_ids列表, target_token_id))
fnn_val_pairs = []

# 生成训练集 (context, target) 对
for seq in train_sequences:
    # seq 已包含<BOS>和<EOS>
    # 遍历每个可作为目标的位置（跳过索引0，因为那是<BOS>起始标记，不能作为目标预测）
    for i in range(1, len(seq)):
        target_id = seq[i]  # 当前位置的token作为目标
        # 取前 context_size 个token 作为输入上下文，不足则用<BOS>填充
        context_ids = []
        for j in range(i - context_size, i):
            if j < 0:
                # 序列超出左边界，用<BOS>填充
                context_ids.append(bos_id)
            else:
                context_ids.append(seq[j])
        # 断言 context_ids 长度等于 context_size
        if len(context_ids) != context_size:
            # 理论上不会进入此分支，仅用于安全检查
            context_ids = ([bos_id] * (context_size - len(context_ids))) + context_ids
            context_ids = context_ids[-context_size:]
        fnn_train_pairs.append((context_ids, target_id))

# 生成验证集 (context, target) 对（用于计算FNN的困惑度）
for seq in val_sequences:
    for i in range(1, len(seq)):
        target_id = seq[i]
        # 构建长度为 context_size 的上下文
        context_ids = []
        for j in range(i - context_size, i):
            if j < 0:
                context_ids.append(bos_id)
            else:
                context_ids.append(seq[j])
        if len(context_ids) != context_size:
            context_ids = ([bos_id] * (context_size - len(context_ids))) + context_ids
            context_ids = context_ids[-context_size:]
        fnn_val_pairs.append((context_ids, target_id))

print(f"FNN训练样本数: {len(fnn_train_pairs)}, 验证样本数: {len(fnn_val_pairs)}")

print('save train and val data')
data_dir = 'data/datasets/'
with open('tokenizer.json', 'w') as f:
    json.dump(token2idx, f)

with open(data_dir + 'fnn_train_pairs.json', 'w') as f:
    json.dump(fnn_train_pairs, f)

with open(data_dir + 'fnn_val_pairs.json', 'w') as f:
    json.dump(fnn_val_pairs, f)

with open(data_dir + 'train_sequences.json', 'w') as f:
    json.dump(train_sequences, f)

with open(data_dir + 'val_sequences.json', 'w') as f:
    json.dump(val_sequences, f)

print('save done')