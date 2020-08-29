"""
Callback function to be passed to the lightning trainer
–Implements the linear warm up schedule followed by cosine annealing, demarcated by current batch idx
"""

import math
import pytorch_lightning as pl

# Really not clear from paper, paper starts talking about cosine annealing when discussing
# the cosine similarity measure. Needs clarification
# I assume 0.1 to 1 linear warm up over first 10000 batches  then annealed to 0.001


class LearningRateDecayCallback(pl.Callback):
    def __init__(
        self,
        config,
        lr_decay=True,
    ):
        super().__init__()
        self.lr_warmup_end = config.lr_warmup_end
        self.lr_warmup_start = config.lr_warmup_start
        self.learning_rate = config.learning_rate
        self.warmup_batch = config.warmup_batch
        self.final_batch = config.final_batch

        self.lr_decay = lr_decay


    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """

        :param trainer:
        :type trainer:
        :param pl_module:
        :type pl_module:
        :param batch:
        :type batch:
        :param batch_idx:
        :type batch_idx:
        :param dataloader_idx:
        :type dataloader_idx:
        """
        optimizer = trainer.optimizers[0]

        if self.lr_decay:
            if batch_idx < self.warmup_batch:
                # linear warmup, in paper: start from 0.1 to 1 over 10000 batches
                lr_mult = float(batch_idx) / float(max(1, self.warmup_batch))
                lr = self.lr_warmup_start + lr_mult * (
                    self.lr_warmup_end - self.lr_warmup_start
                )

            else:
                # Cosine learning rate decay
                progress = float(batch_idx - self.warmup_batch) / float(
                    max(1, self.final_batch - self.warmup_batch)
                )

                lr = max(
                    self.learning_rate
                    + 0.5
                    * (1.0 + math.cos(math.pi * progress))
                    * (self.lr_warmup_end - self.learning_rate),
                    self.learning_rate,
                )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr



