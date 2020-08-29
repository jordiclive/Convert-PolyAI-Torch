import json
from dataclasses import dataclass
from typing import List, NamedTuple

import pytorch_lightning as pl
import torch
from sentencepiece import SentencePieceProcessor
from torch.nn.functional import pad
from torch.utils.data import DataLoader



# Todo implement BPE from scratch with unk tokens hashed (although may achieve worse results on downstream tasks) as
#  perhaps not as general as bpemb's 25000.model


@dataclass
class EncoderInputFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    input_lengths: torch.Tensor

    def pad_sequence(self, seq_len: int):
        self.input_ids = pad(
            self.input_ids, [0, seq_len - self.input_ids.size(0)], "constant", 0
        )
        self.attention_mask = pad(
            self.attention_mask,
            [0, seq_len - self.attention_mask.size(0)],
            "constant",
            0,
        )
        self.position_ids = pad(
            self.position_ids, [0, seq_len - self.position_ids.size(0)], "constant", 0
        )


@dataclass
class EmbeddingPair:
    context: EncoderInputFeature
    reply: EncoderInputFeature


class DataModule(pl.LightningDataModule):
    # using pytorch lightning, as will save a lot of time downstream, when using multi-gpu, distributed method
    # and for managing 16 precision.
    def __init__(self):
        super().__init__()
        self.input_attributes = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "input_lengths",
        ]

    def batching_input_features(
            self, encoder_inputs: List[EncoderInputFeature]
    ) -> EncoderInputFeature:
        max_seq_len = max(
            [
                int(encoder_input.input_lengths.item())
                for encoder_input in encoder_inputs
            ]
        )
        for encoder_input in encoder_inputs:
            encoder_input.pad_sequence(max_seq_len)

        batch_features = {
            feature_name: torch.stack(
                [
                    getattr(encoder_input, feature_name)
                    for encoder_input in encoder_inputs
                ],
                dim = 0,
            )
            for feature_name in self.input_attributes
        }
        return EncoderInputFeature(**batch_features)

    def convert_collate_fn(self, features: List[EmbeddingPair]) -> EmbeddingPair:
        return EmbeddingPair(
            context = self.batching_input_features(
                [feature.context for feature in features]
            ),
            reply = self.batching_input_features([feature.reply for feature in features]),
        )

    def train_dataloader(self, train_dataset):
        return DataLoader(
            train_dataset,
            config.train_batch_size,
            collate_fn = self.convert_collate_fn,
            drop_last = True,     # drop last incomplete batch

        )


    def val_dataloader(self):
        # Todo
        pass
        # return DataLoader()

    def test_dataloader(self):
        # Todo
        pass
        # return DataLoader()


class DatasetInstance(NamedTuple):
    context: List[str]
    response: str


def load_instances_from_reddit_json(dataset_path: str) -> List[DatasetInstance]:
    instances: List[DatasetInstance] = []
    with open(dataset_path) as f:
        for line in f:
            x = json.loads(line)
            context_keys = sorted([key for key in x.keys() if "context" in key])
            instance = DatasetInstance(
                context = [x[key] for key in context_keys], response = x["response"],
            )
            instances.append(instance)
    return instances


class RedditData(torch.utils.data.Dataset):
    def __init__(
            self,
            instances: List[DatasetInstance],
            sp_processor: SentencePieceProcessor,
            truncation_length: int,
    ):
        self.sp_processor = sp_processor
        self.instances = instances
        self.truncation_length = truncation_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        context_str = self.instances[item].context[0]
        context_embedding = self._convert_instance_to_embedding(context_str)
        reply_embedding = self._convert_instance_to_embedding(
            self.instances[item].response
        )
        return EmbeddingPair(context = context_embedding, reply = reply_embedding)

    def _convert_instance_to_embedding(self, input_str: str) -> EncoderInputFeature:
        input_ids = self.sp_processor.EncodeAsIds(input_str)
        if self.truncation_length:
            input_ids = input_ids[: self.truncation_length]
        attention_mask = [1 for _ in range(len(input_ids))]  # [1]*len(input_ids)
        position_ids = [i for i in range(len(input_ids))]  # list(range(len(input_ids))

        return EncoderInputFeature(
            input_ids = torch.tensor(input_ids),
            attention_mask = torch.tensor(attention_mask),
            position_ids = torch.tensor(position_ids),
            input_lengths = torch.tensor(len(input_ids)),
        )
