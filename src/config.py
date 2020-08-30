from typing import NamedTuple

import os

dirname, _ = os.path.split(os.path.dirname(__file__))
# Todo environment variable


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
    device: str = 'cuda'


class ConveRTTrainConfig(NamedTuple):
    # base_path: str =
    sp_model_path: str = os.path.join(dirname, "data/en.wiki.bpe.vs25000.model")
    dataset_path: str = os.path.join(dirname, "data/sample-dataset.json")
    test_dataset_path: str = "data/sample-dataset.json"
    device: str = 'cuda'
    model_save_dir: str = "logs/models/"
    log_dir: str = "logs"
    device: str = "cuda:0"
    use_data_paraller: bool = True

    is_reddit: bool = True

    train_batch_size: int = 64
    test_batch_size: int = 256

    split_size: int = 8
    learning_rate: float = 0.0001
    epochs: int = 10
