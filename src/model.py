import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.nn.modules.normalization import LayerNorm
import os

# fixme Just for jupyter
dirname = os.path.dirname(__file__)
os.chdir(dirname)

from config import ConveRTModelConfig, ConveRTTrainConfig
from dataset import EncoderInputFeature

# start from importing some stuff
from model_components import FeedForward1, FeedForward2, TransformerLayers
from criterion import LossFunction
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
import random

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# class LearningRateDecayCallback(pl.Callback):
#
#     def __init__(self, learning_rate, config,linear_part=1e4, end_part=10e7,min_step lr_decay=True):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.tokens = 0
#         self.final_tokens = final_tokens
#         self.lr_decay = lr_decay
#         self.warmup_tokens = linear_part*config.train_batch_size
#         self.final_tokens = end_part*config.train_batch_size
#         self.min_step = min_step
#
#     def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#         optimizer = trainer.optimizers[0]
#         _, y = batch
#
#         if self.lr_decay:
#             self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
#             if self.tokens < self.warmup_tokens:
#                 # linear warmup
#                 lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
#             else:
#                 # cosine learning rate decay
#                 progress = float(self.tokens - self.warmup_tokens) / float(
#                     max(1, self.final_tokens - self.warmup_tokens))
#                 lr_mult = max(0.1,  0.5  * (1.0 + math.cos(math.pi * progress)))
#             lr = self.learning_rate * lr_mult
#             if lr < self.min_step
#                 self.lr_decay = False
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr




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

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        return self.transformer_layers(x)

    def configure_optimizers(self):
        #todo annealing? param decay, not sure how want if they have restarts? how important. do mention linear part
        optimizer = torch.optim.AdamW(self.parameters(),lr=train_config.learning_rate) # min 0.0001
        return optimizer


    def training_step(self, batch, batch_idx):
        batch_context = batch.context
        batch_reply = batch.reply
        rx = self(batch_context)
        ry = self(batch_reply)
        hx = self.ff2_context(rx, batch_context.attention_mask)
        hy = self.ff2_reply(ry, batch_reply.attention_mask)

        loss = self.loss_function(hx, hy)

        # tqdm_dict = {"train_loss": loss}
        # output = OrderedDict(
        #     {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        # )
        result = pl.TrainResult(minimize = loss, checkpoint_on = loss)
        result.log('train_loss', loss)
        return result


    def validation_step(self, batch, batch_idx):
        output = self.training_step(batch, batch_idx)
        val_output = {"val_loss": output["loss"]}
        return val_output




if __name__ == "__main__":
    from sentencepiece import SentencePieceProcessor
    from dataset import DataModule, RedditData, load_instances_from_reddit_json

    # from pytorch_lightning.callbacks import ModelCheckpoint
    # checkpoint_callback = ModelCheckpoint(filepath = 'CHECKPOINTS', verbose = True, monitor = 'val_loss', mode = 'min')
    train_config = ConveRTTrainConfig()
    model_config = ConveRTModelConfig()
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(train_config.sp_model_path)
    train_instances = load_instances_from_reddit_json(train_config.dataset_path)
    RD = RedditData(train_instances, tokenizer,60)
    dm = DataModule()
    train_loader = dm.train_dataloader(RD)

    model = SingleContextConvert(model_config, train_config)
    #print([p for p in model.named_parameters()])
    Total = sum(p.numel() for p in model.parameters())

    print(
        f"{Total:,}"
    )  # 27,478,744 (16M + 13M) so less paramters, makes sense maybe because  embedding was different.
    print(
        f"{Total-12829692:,}" # maybe only including shared part.
    )

    print(
        f"{Total-29000000:,}"
    )
    print("grad_total", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optimizer = torch.optim.AdamW(model.parameters(), lr = train_config.learning_rate)
    # for batch in train_loader:
    #     batch_context = batch.context
    #     batch_reply = batch.reply
    #     rx = model(batch_context)
    #     ry = model(batch_reply)
    #     hx = model.ff2_context(rx, batch_context.attention_mask)
    #     hy = model.ff2_reply(ry, batch_reply.attention_mask)
    #     loss = model.loss_function(hx, hy)
    #     loss.backward()
    #     break
    # bias = model.transformer_layers.transformer_layers[0].self_attention.bias
    # print(bias.grad) # does change.

    # # only saves best
    # trainer = (
    #     pl.Trainer()
    # )  # ,checkpoint_callback = checkpoint_callback)  # ,resume_from_checkpoint=)
    # trainer.fit(model, train_dataloader=train_loader, val_dataloaders=train_loader)
