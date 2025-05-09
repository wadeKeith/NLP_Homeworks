from dataclasses import dataclass
from pathlib import Path




@dataclass
class TrainConfig:
    lr_init: float = 1e-3
    num_epochs: int = 5
    fnn_batch_size: int = 10240
    seq_batch_size: int = 16
    context_size: int = 5
    embed_size: int = 128
    hidden_size: int = 256
    rnn_num_layers: int = 1
    transformer_num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 512
