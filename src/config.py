import os
from typing import NamedTuple

dirname, _ = os.path.split(os.path.dirname(__file__))

class ConveRTModelConfig(NamedTuple):

    num_embed_hidden: int = 512
    feed_forward1_hidden: int = 2048
    feed_forward2_hidden: int = 1024
    num_attention_project: int = 64
    vocab_size: int = 25000
    num_encoder_layers: int = 6
    dropout_rate: float = 0.0
    n: int = 121
    relative_attns: list = [3, 5, 48, 48, 48, 48]
    num_attention_heads: int = 2
    token_sequence_truncation: int = 60


class ConveRTTrainConfig(NamedTuple):

    sp_model_path: str = os.path.join(dirname, "data/en.wiki.bpe.vs25000.model")
    dataset_path: str = os.path.join(dirname, "data/sample-dataset.json")
    test_dataset_path: str = "data/sample-dataset.json"

    model_save_dir: str = "lightning_logs/checkpoints/"
    log_dir: str = "lightning_logs"
    device: str = "cpu"
    use_data_paraller: bool = True

    is_reddit: bool = True

    train_batch_size: int = 64
    test_batch_size: int = 256

    split_size: int = 8
    learning_rate: float = 1e-3  # final learning rate ie 'lr annealed to'
    lr_warmup_start: float = 0.1  # start of lr before initial linear warmup section
    lr_warmup_end: float = 1.0  # end of linear warmup section , annealing begin
    warmup_batch: float = 10000  # how many batches linear warm up for
    final_batch: float = 1e8  # final batch of training when want learning rate
    learning_rate_end: float = 0.0001
    epochs: int = 10
    grad_norm_clip: float = 1.0
    smoothing: float = 0.2
    l2_weight_decay: float = 1e-5  # note: different from L2 reg, as working with Adam. L2 regularization
    # (or any lagrange m on loss) not wise


