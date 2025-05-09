import torch.nn as nn
import torch.nn.functional as F

class FNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_size):
        """
        前馈神经网络语言模型
        :param vocab_size: 词汇大小
        :param embed_size: 词向量维度
        :param hidden_size: 隐藏层大小
        :param context_size: 上下文窗口大小 N (使用前N个词预测下一个词)
        """
        super(FNNLM, self).__init__()
        self.model_type = "FNN"
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 将 context_size 个词的嵌入拼接后输入隐藏层，全连接层维度: context_size*embed_size -> hidden_size
        self.fc1 = nn.Linear(context_size * embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        # 可以加入 dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, context_ids):
        """
        :param context_ids: 张量形状 (batch_size, context_size)，表示每个样本的上下文token序列
        :return: 输出张量 (batch_size, vocab_size)，表示每个样本预测下一个词的logits
        """
        # 1. 获取上下文的词嵌入，形状: (batch_size, context_size, embed_size)
        embeds = self.embed(context_ids)
        # 2. 将 context_size 个嵌入向量展平成单一向量 (在维度拼接)
        batch_size = embeds.size(0)
        embeds_flat = embeds.view(batch_size, -1)  # (batch_size, context_size*embed_size)
        # 3. 前馈网络：隐藏层激活
        hidden = F.relu(self.fc1(embeds_flat))
        hidden = self.dropout(hidden)
        # 4. 输出层 (未经过softmax的logits)
        output_logits = self.fc2(hidden)
        return output_logits