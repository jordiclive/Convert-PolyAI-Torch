import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from src.config import ConveRTModelConfig
from src.dataset import EncoderInputFeature




def circulant_mask(n: int, window: int) -> torch.Tensor:
    """ Calculate the relative attention mask, calculated once when model instatiated, as a subset of this matrix
        will be used for a input length less than max.
        i,j represent relative token positions in this matrix and in the attention scores matrix,
         this mask enables attention scores to be set to 0 if further than the specified window length

            :param n: a fixed parameter set to be larger than largest max sequence length across batches
            :param window: [window length],
            :return relative attention mask
    """
    circulant_t = torch.zeros(n, n)
    # [0, 1, 2, ..., window, -1, -2, ..., window]
    offsets = [0] + [i for i in range(window + 1)] + [-i for i in range(window + 1)]
    if window >= n:
        return torch.ones(n, n)
    for offset in offsets:
        # size of the 1-tensor depends on the length of the diagonal
        circulant_t.diagonal(offset=offset).copy_(torch.ones(n - abs(offset)))
    return circulant_t


class SubwordEmbedding(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        """ init embedding model

        :param config: model.config
        :type config: ConveRTModelConfig
        """
        super().__init__()
        self.subword_embed = nn.Embedding(
            config.vocab_size, config.num_embed_hidden
        )  #eg. 25000 x 512
        self.m1_positional_embed = nn.Embedding(47, config.num_embed_hidden)
        self.m2_positional_embed = nn.Embedding(11, config.num_embed_hidden)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Subword Embedding and Positional encoding, takes in sequence of sub words, calculates
        subword embeddings and adds positional encodings

        m1_positional_embed is calculated with m1_embed_weight(mod(position_ids, 47))
        m2_positional_embed is calculated with m1_embed_weight(mod(position_ids, 11))

        :param input_ids: raw token ids
        :type input_ids: torch.LongTensor
        :param position_ids: [description], defaults to None
        :type position_ids: torch.LongTensor, optional
        :return: return embedding sum (position{m1, m2} + sub-word)
        :rtype: torch.Tensor

        """
        subword_embed = self.subword_embed.forward(
            input_ids
        )  # B x T x d_emb eg. 64 x 47 x 512
        m1_positional_embed = self.m1_positional_embed.forward(
            torch.fmod(position_ids, 47)
        )
        m2_positional_embed = self.m2_positional_embed.forward(
            torch.fmod(position_ids, 11)
        )  # B x T x d_emb
        embedding = subword_embed + m1_positional_embed + m2_positional_embed
        return embedding


class SelfAttention(
    nn.Module
):
    """normal query, key, value based self attention but with relative attention functionality
     and a learnable bias encoding relative token position which is added to the attention scores before the softmax"""

    def __init__(self, config: ConveRTModelConfig, relative_attention: int):
        """init self attention weight of each key, query, value and output projection layer.

        :param config: model config
        :type config: ConveRTModelConfig
        """
        super().__init__()

        self.config = config
        self.query = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.key = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.value = nn.Linear(config.num_embed_hidden, config.num_attention_project)

        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(
            config.num_attention_project, config.num_embed_hidden
        )
        self.bias = torch.nn.Parameter(torch.randn(config.n), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.bias.data.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        self.relative_attention = relative_attention
        self.n = self.config.n
        self.half_n = self.n // 2
        self.register_buffer(
            "relative_mask",
            circulant_mask(config.token_sequence_truncation, self.relative_attention),
        )

    def forward(
        self, attn_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """ calculate self-attention of query, key and weighted to value at the end.
        self-attention input is projected by linear layer at the first time.
        applying attention mask for ignore pad index attention weight. Relative attention mask
        applied and a learnable bias added to the attention scores.
        return value after apply output projection layer to value * attention

        :param attn_input: [description]
        :type attn_input: [type]
        :param attention_mask: [description], defaults to None
        :type attention_mask: [type], optional
        :return: [description]
        :rtype: [type]
        """
        self.T = attn_input.size()[1]
        # input is B x max seq len x n_emb
        _query = self.query.forward(attn_input)
        _key = self.key.forward(attn_input)
        _value = self.value.forward(attn_input)

        # scaled dot product
        attention_scores = torch.matmul(_query, _key.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(
            self.config.num_attention_project
        )

        # Relative attention

        # extended_attention_mask = attention_mask.to(attention_scores.device)  # fp16 compatibility
        extended_attention_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        attention_scores = attention_scores + extended_attention_mask


        # fix circulant_matrix to matrix of size 60 x60 (max token truncation_length,
        # register as buffer, so not keep creating masks of different sizes.

        attention_scores = attention_scores.masked_fill(
            self.relative_mask.unsqueeze(0)[:, : self.T, : self.T] == 0, float("-inf")
        )

        # Learnable bias vector is used of max size,for each i, different subsets of it are added to the scores, where the permutations
        # depend on the relative position (i-j). this way cleverly allows no loops. bias vector is 2*max truncation length+1
        # so has a learnable parameter for each eg. (i-j) /in {-60,...60} .

        ii, jj = torch.meshgrid(torch.arange(self.T), torch.arange(self.T))
        B_matrix = self.bias[self.n // 2 - ii + jj]

        attention_scores = attention_scores + B_matrix.unsqueeze(0)

        attention_scores = self.softmax(attention_scores)
        output = torch.matmul(attention_scores, _value)

        output = self.output_projection(output)

        return output  # B x T x num embed hidden 64 x eg. 47 x 512


class FeedForward1(nn.Module):
    """ feed-forward 1 is the
        standard FFN layer also used by Vaswani et al. (2017),"""

    def __init__(
        self, input_hidden: int, intermediate_hidden: int, dropout_rate: float = 0.0
    ):
        #          512         2048
        """
        :param input_hidden: first-hidden layer input embed-dim
        :type input_hidden: int
        :param intermediate_hidden: layer-(hidden)-layer middle point weight
        :type intermediate_hidden: int
        :param dropout_rate: dropout rate, defaults to None
        :type dropout_rate: float, optional
        """
        super().__init__()

        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, input_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward through fully-connected 2-layer

        :param x: F input
        :type x: torch.Tensor
        :return: return F output
        :rtype: torch.Tensor
        """
        x = F.gelu(self.linear_1(x))
        return self.linear_2(self.dropout(x))


class SharedInnerBlock(nn.Module):
    """ Inner 'Transformer' block, this block is repeated six times in the original paper with respective relative attentions
    [3, 5, 48, 48, 48, 48]

    """

    def __init__(self, config: ConveRTModelConfig, relative_attn: int):
        super().__init__()
        """
        :param config: model config
        :type config: ConveRTModelConfig
        :param config: relative attention
        :type config: int
         
        """
        self.config = config
        self.self_attention = SelfAttention(config, relative_attn)
        self.norm1 = LayerNorm(config.num_embed_hidden)  # 512
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ff1 = FeedForward1(
            config.num_embed_hidden, config.feed_forward1_hidden, config.dropout_rate
        )
        self.norm2 = LayerNorm(config.num_embed_hidden)

    def forward(self, x: torch.Tensor, attention_mask: int) -> torch.Tensor:
        """calculating single Transformer block

        1. single-self attention (EMBED_DIM -> ATTEN_PROJ -> EMBED_DIM)
        2. first  residual connection -> layer norm
        3. feed-forward-1 layer (EMBED_DIM -> FFD-1-DIM -> EMBED_DIM)
        4. second  residual connection -> layer norm

        :param x: embed_output: sub-word embedding + positional encoding
        :type x: embed_output: torch.Tensor
        :param attention_mask: 1.0 for token position, 0.0 for padding position, defaults to None
        :type attention_mask: Optional[torch.Tensor], optional
        :return: Transformer block forward output
        :rtype: torch.Tensor

        """
        # think better practice to relabel same var , although is more confusing to read.
        x = x + self.self_attention(x, attention_mask=attention_mask)
        x = self.norm2(x)
        x = x + self.ff1(x)
        return self.norm2(x)




# pretty basic, just single head. but done many times, stack to have another dimension (4 with batches).# so get stacks of B x H of attention scores T x T..
# then matrix multiply these extra stacks with the v
# (B xnh)x T xT . (Bx nh xTx hs) gives (B Nh) T x hs stacks. now  hs is set to be final dimension/ number of heads, so reorder the stacks (concatenating them)
# can have optional extra projection layer, but doing that later


class MultiheadAttention(nn.Module):
    """Standard non causal MHA, Half Hugging Face/Half Andrej Karpathy implementation,
     no need to mask as after previous layers"""

    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_attn_proj = config.num_embed_hidden * config.num_attention_heads
        self.attention_head_size = int(self.num_attn_proj / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.key = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.value = nn.Linear(config.num_embed_hidden, self.num_attn_proj)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

            attention_scores = attention_scores + attention_mask

        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self.dropout(attention_scores)

        y = attention_scores @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_attn_proj)

        return y


class TransformerLayers(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.config = config

        self.subword_embedding = SubwordEmbedding(config)
        self.transformer_layers = nn.ModuleList(
            [SharedInnerBlock(config, window) for window in config.relative_attns]
        )
        self.MHA = MultiheadAttention(config)

    def forward(self, encoder_input: EncoderInputFeature) -> torch.Tensor:
        input_ids = encoder_input.input_ids
        position_ids = encoder_input.position_ids
        attention_mask = encoder_input.attention_mask
        output = self.subword_embedding(input_ids, position_ids)
        for l in self.transformer_layers:
            output = l(output, attention_mask)
        output = self.MHA(output)
        return output


class FeedForward2(
    nn.Module
):  # params are not shared for context and reply. so need two sets of weights
    """Fully-Connected 3-layer Linear Model"""

    def __init__(self, config):
        """
        :param input_hidden: first-hidden layer input embed-dim
        :type input_hidden: int
        :param intermediate_hidden: layer-(hidden)-layer middle point weight
        :type intermediate_hidden: int
        :param dropout_rate: dropout rate, defaults to None
        :type dropout_rate: float, optional
        """
        # paper specifies,skip connections,layer normalization, and orthogonal initialization

        super().__init__()
        # 3,679,744 x2 params
        self.linear_1 = nn.Linear(
            config.feed_forward2_hidden, config.feed_forward2_hidden
        )
        self.linear_2 = nn.Linear(
            config.feed_forward2_hidden, config.feed_forward2_hidden
        )
        # self.linear_3 = nn.Linear(
        #     config.feed_forward2_hidden, config.feed_forward2_hidden
        # )
        self.norm1 = LayerNorm(config.feed_forward2_hidden)
        self.norm2 = LayerNorm(config.feed_forward2_hidden)
        # self.norm3 = LayerNorm(config.feed_forward2_hidden)
        self.final = nn.Linear(config.feed_forward2_hidden, config.num_embed_hidden)
        self.orthogonal_initialization() # torch implementation works perfectly out the box,

    def orthogonal_initialization(self):
        for l in [
            self.linear_1,
            self.linear_2,
        ]:  # self.linear_3]:
            torch.nn.init.orthogonal_(l.weight)

    def forward(self, x: torch.Tensor, attn_msk: torch.Tensor) -> torch.Tensor:
        sentence_lengths = attn_msk.sum(1)

        # adding square root reduction projection separately as not a shared.
        # part of the diagram torch.Size([64, 50, 1024])

        # x has dims B x T x 2*d_emb
        norms = 1 / torch.sqrt(sentence_lengths.double()).float()  # 64
        x = norms.unsqueeze(1) * torch.sum(x, dim=1)  # 64 x1024

        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))
        # x = x + F.gelu(self.linear_3(self.norm3(x)))

        return F.normalize(self.final(x), dim=1, p=2)  # 64 512



