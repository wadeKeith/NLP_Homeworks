import torch
import math
from torch.nn import functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from configs.params import TrainConfig
import torch.nn as nn
from torch.utils.data import DataLoader
import gc


def compute_perplexity(model, data_loader, device, vocab_size, PAD_ID=0):
    """
    计算模型在指定数据集上的困惑度 (Perplexity)。
    :param model: 已训练的语言模型
    :param data_loader: 验证或测试集的 DataLoader (返回 (inputs, targets) 张量对)
    :return: perplexity 困惑度值
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # 禁用梯度计算
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 前向计算得到 logits
            logits = model(inputs)
            # 将 logits 和 targets 展平成二维，用于计算损失
            if logits.dim() == 3:
                # 形状 (batch, seq_len, vocab) 展平为 ((batch*seq_len), vocab)
                logits_flat = logits.view(-1, vocab_size)
            else:
                # 形状 (batch, vocab) 直接视为 (batch, vocab)
                logits_flat = logits
            targets_flat = targets.view(-1)
            # 计算当前 batch 的总损失（使用 sum reduction）
            loss = F.cross_entropy(
                logits_flat, targets_flat, ignore_index=PAD_ID, reduction="sum"
            )
            total_loss += loss.item()
            # 统计有效token数量（非PAD的目标）
            if PAD_ID is not None:
                total_tokens += (targets_flat != PAD_ID).sum().item()
            else:
                total_tokens += targets_flat.numel()
    # 计算平均损失（总损失除以有效token数），然后取指数得到困惑度
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def train_model(
    model: nn.Module,
    train_loader:DataLoader,
    val_loader:DataLoader,
    config: TrainConfig,
    device,
    vocab_size,
    loss_fn,
    pad_id,
):
    model.to(device)
    total_steps = len(train_loader) * config.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init)
    warmup_steps = int(total_steps * 0.01)  # 1% 的训练步数用于 warmup
    # 学习率调度器：使用余弦退火调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    global_step = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        # tqdm 进度条：leave=False 在每个 epoch 结束后自动清空
        pbar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            unit="batch",
            leave=False,
        )

        for step, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            logits_flat = logits.view(-1, vocab_size) if logits.dim() == 3 else logits
            targets_flat = targets.view(-1)
            loss = loss_fn(logits_flat, targets_flat)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            running_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            # 更新 tqdm 后缀
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running_loss / step:.4f}",
                lr=f"{current_lr:.6f}",
                gs=global_step,
            )

            del inputs, targets, logits, logits_flat, targets_flat, loss
            torch.cuda.empty_cache()

        # 一个 epoch 完成后计算验证集困惑度
        val_ppl = compute_perplexity(model, val_loader, device, vocab_size, pad_id)
        print(
            f"[Epoch {epoch}] 训练集平均 loss = {running_loss / len(train_loader):.4f} | "
            f"验证集 PPL = {val_ppl:.2f}"
        )
        # ---------- 清理 epoch 结束后的显存 ----------
        torch.cuda.empty_cache()   # 清理缓存显存
        gc.collect()               # 垃圾回收
    return model  # 返回训练后的模型
