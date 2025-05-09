import torch.nn as nn
import torch
import math



class PositionalEncoding(nn.Module):
    """
    基于正弦和余弦函数的位置编码 [oai_citation:6‡h-huang.github.io](https://h-huang.github.io/tutorials/beginner/transformer_tutorial.html#:~:text=pe%20%3D%20torch,register_buffer%28%27pe%27%2C%20pe)。
    给定 embedding 维度 d_model 和最大序列长度 max_len，预计算位置编码矩阵。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 形状 (max_len, 1)
        # 按照公式计算 sin, cos 编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维使用 sin，奇数维使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度以便与输入相加: 最终 pe 形状 (max_len, 1, d_model)
        pe = pe.unsqueeze(1)
        # 将 pe 注册为buffer，使其在模型保存时保存，但不作为参数训练
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        :param x: 形状 (seq_len, batch_size, d_model)
        :return: 加上位置编码后的张量 (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        # 将对应长度的位置编码加到输入上
        x = x + self.pe[:seq_len]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, dim_feedforward=512, num_layers=2, dropout=0.1, pad_id=0):
        """
        Transformer 自注意力语言模型 (基于TransformerEncoder)
        :param vocab_size: 词汇大小
        :param d_model: 模型隐藏维度(也是词嵌入维度)
        :param nhead: 多头注意力头数
        :param dim_feedforward: 前馈网络隐藏维度
        :param num_layers: TransformerEncoderLayer 层数
        :param dropout: dropout概率
        """
        super(TransformerLM, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        # 嵌入层和位置编码层
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        # TransformerEncoder：由 num_layers 个 TransformerEncoderLayer 堆叠
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出解码层：将最终隐藏状态映射到词表
        self.fc = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id  # 用于填充的token ID
    
    def _generate_square_subsequent_mask(self, seq_len, device):
        """
        生成长度为 seq_len 的序列的上三角掩码张量，使模型不能看到当前位置之后的词 [oai_citation:7‡h-huang.github.io](https://h-huang.github.io/tutorials/beginner/transformer_tutorial.html#:~:text=layers%20of%20nn,Softmax%20function)。
        Mask 张量形状 (seq_len, seq_len)，在 [i,j] (0-index)位置为 -inf 表示屏蔽 j>i 的位置。
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device))  # 上三角矩阵(含主对角线)元素为1
        mask = mask.transpose(0, 1)  # 转置使得下三角(包括主对角线)为1，上三角为1的转置结果
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # 注意：PyTorch Transformer 要求 mask[i,j] 为 -inf 时表示 i 不能看 j
        return mask
    
    def forward(self, input_ids):
        """
        :param input_ids: 张量形状 (batch_size, seq_len)，表示输入序列批次
        :return: 输出 logits 张量，形状 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # 1. 获取词嵌入并乘以 sqrt(d_model) (缩放)，然后转置为 (seq_len, batch_size, d_model)
        embeds = self.embed(input_ids) * math.sqrt(self.d_model)
        embeds = embeds.transpose(0, 1)  # 转置为 (seq_len, batch, d_model) 以供 Transformer 使用
        # 2. 加入位置编码
        embeds = self.pos_encoder(embeds)  # (seq_len, batch, d_model)
        # 3. 生成自回归的注意力 mask，防止模型关注未来的token
        seq_mask = self._generate_square_subsequent_mask(seq_len, device)  # (seq_len, seq_len)
        # 4. 构造填充 mask：大小 (batch_size, seq_len)，在padding位置为 True
        #    作用是在自注意力中屏蔽对<PAD>的关注，以及在输出层忽略这些位置
        pad_mask = (input_ids == self.pad_id)  # PAD_ID 在上文已定义
        # 5. 通过 TransformerEncoder 得到编码表示
        #    注意：TransformerEncoderLayer 默认实现中，src_mask 是应用在自注意力的 [tgt x tgt] 注意力矩阵上，
        #          src_key_padding_mask 用于在注意力中屏蔽填充位置。
        encoder_output = self.transformer_encoder(embeds, mask=seq_mask, src_key_padding_mask=pad_mask)
        # encoder_output 形状: (seq_len, batch, d_model)
        # 6. 将维度转回 (batch, seq_len, d_model)
        encoder_output = encoder_output.transpose(0, 1)  # (batch, seq_len, d_model)
        # 7. 输出层映射到词汇表空间
        output_logits = self.fc(encoder_output)  # (batch, seq_len, vocab_size)
        return output_logits