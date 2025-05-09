import torch.nn as nn
import torch




class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        """
        循环神经网络语言模型 (使用LSTM)
        :param vocab_size: 词汇大小
        :param embed_size: 词向量维度
        :param hidden_size: LSTM隐藏状态维度
        :param num_layers: LSTM层数
        """
        super(RNNLM, self).__init__()
        self.model_type = "RNN"
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM层，batch_first=True 使输入输出形状为 (batch, seq_len, feature)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, 
                             batch_first=True)
        # 输出线性层：将隐藏状态映射到词表大小
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, hidden_state=None):
        """
        :param input_ids: 张量形状 (batch_size, seq_len)，表示输入序列（已包含<BOS>但不含序列终止<EOS>）
        :param hidden_state: 初始隐藏状态 (h0, c0)，如果为 None 则使用全零初始化。
        :return: 输出 logits 张量，形状 (batch_size, seq_len, vocab_size)
        """
        # 1. 嵌入层: (batch, seq_len) -> (batch, seq_len, embed_size)
        embeds = self.embed(input_ids)
        # 2. 初始化隐藏状态如果未提供
        if hidden_state is None:
            # LSTM有两个隐状态 (h, c)，初始化为零
            # 形状: (num_layers, batch, hidden_size)
            h0 = torch.zeros(self.num_layers, input_ids.size(0), self.hidden_size, device=input_ids.device)
            c0 = torch.zeros(self.num_layers, input_ids.size(0), self.hidden_size, device=input_ids.device)
            hidden_state = (h0, c0)
        # 3. 通过 LSTM 得到所有时间步的输出
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        # lstm_out 形状: (batch, seq_len, hidden_size)
        # 4. 将 LSTM 的输出映射到词表空间
        output_logits = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        return output_logits