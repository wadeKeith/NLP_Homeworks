import torch

def collate_seq(batch, pad_id):
    # 计算本 batch 中序列的最大长度
    max_len = max(len(seq) for seq in batch)
    # 我们将序列长度统一到 max_len
    max_input_len = max_len - 1  # 输入序列长度（去掉EOS）
    batch_inputs = []
    batch_targets = []
    for seq in batch:
        # 输入为去掉最后一个元素(EOS)的序列
        inp_seq = seq[:-1]
        # 目标为去掉第一个元素(BOS)的序列
        tgt_seq = seq[1:]
        # 断言 len(inp_seq) == len(tgt_seq)
        # 用 PAD 填充到 max_input_len 长度
        inp_seq_pad = inp_seq + [pad_id] * (max_input_len - len(inp_seq))
        tgt_seq_pad = tgt_seq + [pad_id] * (max_input_len - len(tgt_seq))
        batch_inputs.append(inp_seq_pad)
        batch_targets.append(tgt_seq_pad)
    # 转换为张量
    batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)
    batch_targets = torch.tensor(batch_targets, dtype=torch.long)
    return batch_inputs, batch_targets


def collate_fnn(batch):
    # batch是一个列表，内部每个元素为(tuple: (context_ids, target_id))
    contexts = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return contexts_tensor, targets_tensor



