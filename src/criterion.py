import torch
import torch.nn as nn


class LossFunction(nn.Module):
    @staticmethod
    def cosine_similarity_matrix(
        context_embed: torch.Tensor, reply_embed: torch.Tensor
    ) -> torch.Tensor:
        assert context_embed.size(0) == reply_embed.size(0)
        cosine_similarity = torch.matmul(
            context_embed, reply_embed.T
        )  # both normalized already from last layer. So cosine similarity for batch can be
        # efficiently calculated as  simply similarity matrix
        return cosine_similarity

    def forward(
        self, context_embed: torch.Tensor, reply_embed: torch.Tensor
    ) -> torch.Tensor:
        cosine_similarity = self.cosine_similarity_matrix(context_embed, reply_embed)
        j = -torch.sum(torch.diagonal(cosine_similarity))

        cosine_similarity.diagonal().copy_(torch.zeros(cosine_similarity.size(0)))
        # The abel smoothing implemented is not clear from the paper. As not CE loss, have negative sampling.
        # I assumed a lessening of how penalized the model is when assigns non zero probs
        # to wrong class by increasing the negative component of loss fn by label smoothing mass indicated in paper

        j = 0.8 * j + (
            0.2 / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))
        ) * torch.sum(cosine_similarity)


        j += torch.sum(torch.logsumexp(cosine_similarity, dim=0))

        # torch.logsumexp(input, dim, keepdim=False, out=None)

        # Returns the log of summed exponentials of each row of the input tensor in the
        # given dimension dim. Very important The computation is numerically stable with logs/exp in loss.
        # This torch implementation is done in C.
        return j  # negative of objective fn in paper as want loss fn


