# , computed as cosine similarity
# with annealing between the encodings hx and
# hy. It starts at 1 and ends at
# p
# d,

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def CosineSimilarityMatrix(
        context_embed: torch.Tensor, reply_embed: torch.Tensor
    ) -> torch.Tensor:

        assert context_embed.size(0) == reply_embed.size(0)
        cosine_similarity = torch.matmul(context_embed, reply_embed.T)# both normalized.

        return cosine_similarity


    def forward(self, context_embed: torch.Tensor, reply_embed: torch.Tensor
    ) -> Tuple[int]:
        cosine_similarity = self.CosineSimilarityMatrix(context_embed, reply_embed)
        J = -torch.sum(torch.diagonal(cosine_similarity))
        cosine_similarity.diagonal().copy_(torch.zeros(cosine_similarity.size(0)))
        # torch.logsumexp(input, dim, keepdim=False, out=None)
        #Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. The computation is numerically stabilized.
        # done in C
        J += torch.sum(torch.logsumexp(cosine_similarity,dim=0))
        return J # negative J as loss fn

if __name__ == "__main__":
    context = fnn.normalize(torch.rand(64, 1024), dim = 1, p = 2)
    reply = fnn.normalize(torch.rand(64, 1024), dim = 1, p = 2)
    L = LossFunction()
    s = L.CosineSimilarityMatrix(context, context)
    print(s)
    print(L(context, reply))

    from src.config import *
    from src.dataset import *
    from src.model_components import *
    args = ConveRTTrainConfig()
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(args.sp_model_path)
    u = load_instances_from_reddit_json(args.dataset_path)
    RD = RedditData(u, tokenizer, 60)
    SE = SubwordEmbedding(ConveRTModelConfig())
    dm = DataModule()
    train_loader = dm.train_dataloader(RD)
    iterat = iter(train_loader)
    batch = next(iterat)
    batch_context = batch.context
    batch_reply = batch.reply
    model_config = ConveRTModelConfig()
    tls = TransformerLayers(model_config)
    ff2_context = FeedForward2(model_config)
    ff2_reply = FeedForward2(model_config)
    rx = tls(batch_context)
    ry = tls(batch_reply)
    hx = ff2_context(rx, batch_context.attention_mask)
    hy = ff2_reply(ry, batch_reply.attention_mask)
    print(L(hx,hx))
