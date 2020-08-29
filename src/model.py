import logging
import random
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch

from config import ConveRTModelConfig, ConveRTTrainConfig
from criterion import LossFunction

from model_components import FeedForward2, TransformerLayers

import argparse
from sentencepiece import SentencePieceProcessor
from dataset import DataModule, RedditData, load_instances_from_reddit_json

from src.lr_decay import LearningRateDecayCallback

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_subword_params(model):
    """Long winded helper fn to return Subword Embedding Params for clipping, as they are the only parameters that
    are gradient clipped in the paper, only calculated once after model instantiation, but before training"""
    embeds = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if mn.startswith("transformer_layers.subword_embedding"):
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                embeds.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    return [param_dict[pn] for pn in sorted(list(embeds))], embeds


# todo  need to write own
# lightning optimizer step to include torch.nn.utils.clip_grad_norm_(find_subword_params(model), config.grad_norm_clip),


class SingleContextConvert(pl.LightningModule):
    def __init__(
            self, model_config: ConveRTModelConfig, train_config: ConveRTTrainConfig
    ):
        super().__init__()

        self.model_config = model_config
        self.train_config = train_config
        self.transformer_layers = TransformerLayers(model_config)
        self.ff2_context = FeedForward2(model_config)
        self.ff2_reply = FeedForward2(model_config)
        self.loss_function = LossFunction()

        self.weight_decay = train_config.l2_weight_decay

        self.hparams = self.train_config._field_defaults
        self.hparams.update(self.model_config._field_defaults)
        self.subword_params = None

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    def register_subword_params(self):
        self.subword_params = find_subword_params(self)[0]

    def forward(self, x):
        return self.transformer_layers(x)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.subword_params, self.train_config.grad_norm_clip)


    def configure_optimizers(self):
        """
        here I did not implement weight decay on bias and Layernorm layers as is typical in modern  NLP papers.
        I do not think the paper specified params to avoid weight decay on
        :return:
        :rtype:
        """
        # create the optimizer, here I did not implement weight decay on bias and weight as is customary in modern
        # NLP papers.
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.l2_weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr = self.hparams.learning_rate
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_context = batch.context
        batch_reply = batch.reply
        rx = self(batch_context)
        ry = self(batch_reply)
        hx = self.ff2_context(rx, batch_context.attention_mask)
        hy = self.ff2_reply(ry, batch_reply.attention_mask)

        loss = self.loss_function(hx, hy)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        # result = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        # result.log("train_loss", loss)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.training_step(batch, batch_idx)
        val_output = {"val_loss": output["loss"]}
        return val_output


def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type = int, default = 1)
    #parser.add_argument("--precision", type = int, default = 16)
    parser.add_argument("--progress_bar_refresh_rate", type = int, default = 1)
    parser.add_argument("--row_log_interval", type = int, default = 1)

    args = parser.parse_args()

    return args


def main():
    train_config = ConveRTTrainConfig()
    model_config = ConveRTModelConfig()
    tokenizer = SentencePieceProcessor()
    args = _parse_args()
    tokenizer.Load(train_config.sp_model_path)
    train_instances = load_instances_from_reddit_json(train_config.dataset_path)
    RD = RedditData(train_instances, tokenizer, 60)
    dm = DataModule()
    train_loader = dm.train_dataloader(RD)
    model = SingleContextConvert(model_config, train_config)
    lr_decay = LearningRateDecayCallback(train_config)
    model.register_subword_params()

    trainer = (
        pl.Trainer.from_argparse_args(args, callbacks = [lr_decay])
    )  # ,checkpoint_callback = checkpoint_callback)  # ,resume_from_checkpoint=)
    trainer.fit(model, train_dataloader = train_loader, val_dataloaders = train_loader)


if __name__ == "__main__":
    main()
