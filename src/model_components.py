import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.nn.modules.normalization import LayerNorm
import os

# #fixme Just for jupyter
# dirname= os.path.dirname(__file__)
# os.chdir(dirname)

from src.config import ConveRTModelConfig
from src.dataset import EncoderInputFeature

# start from importing some stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




def circulant_mask(n: int, window: int) -> torch.Tensor:
    """ Calculate the relative attention mask
        i,j represent relative token positions in this matrix and in the attention scores matrix,
         this mask enables attention scores to be set to 0 if further than the specified window length

            :param n: a fixed parameter set to be larger than largest max sequence length across batches
            :param window: [window length],
            :return relative attention mask
    """
    #todo sort out GPU for this tensor, do not register as different every batch unless pad all inputs to be of size 60
    circulant_t = torch.zeros(n, n)
    # [0, 1, 2, ..., window, -1, -2, ..., window]
    offsets = [0] + [i for i in range(window + 1)] + [-i for i in range(window + 1)]
    if window >= n:
        return torch.ones(n, n)
    for offset in offsets:
        # size of the 1-tensor depends on the length of the diagonal
        circulant_t.diagonal(offset = offset).copy_(torch.ones(n - abs(offset)))
        circulant_t.to('cuda')
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
        )  # 10000 x512, bpe is 10000
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
        # believe correct implementation. embedding when called. takes the row corresponding to the
        # number of element and stacks it in matrix.
        # #fixme pids= tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  0,  0,  0..]
        # this is one example, each position will have an embedding representation of that token.
        # each position will produce a different positional encoding to add to the embedding for the token.
        # torch.fmod(pids,11)  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  1,  2,  0,  0,  0,
        # will do for the whole batch
        subword_embed = self.subword_embed.forward(
            input_ids
        )  # 64 x 47 x 512 (B x T x 512
        m1_positional_embed = self.m1_positional_embed.forward(
            torch.fmod(position_ids, 47)
        )
        m2_positional_embed = self.m2_positional_embed.forward(
            torch.fmod(position_ids, 11)
        )  # 64 x 47 x 512
        embedding = subword_embed + m1_positional_embed + m2_positional_embed
        return embedding


class SelfAttention(nn.Module): # bulk of params in these 6 layers. maybe with relative attention, can get rid of params sparse query key value matrices
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

        self.softmax = nn.Softmax(dim = -1)
        self.output_projection = nn.Linear(
            config.num_attention_project, config.num_embed_hidden
        )
        self.bias = torch.nn.Parameter(torch.randn(config.n), requires_grad = True)
        stdv = 1.0 / math.sqrt(self.bias.data.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        self.relative_attention = relative_attention
        self.n = self.config.n
        self.half_n = self.n // 2

    def forward(
            self, attn_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """ calculate self-attention of query, key and weighted to value at the end.
        self-attention input is projected by linear layer at the first time.
        applying attention mask for ignore pad index attention weight. Relative attention mask
        appied and a Learnable bias added to the attention scores.
        return value after apply output projection layer to value * attention

        :param attn_input: [description]
        :type attn_input: [type]
        :param attention_mask: [description], defaults to None
        :type attention_mask: [type], optional
        :return: [description]
        :rtype: [type]
        """
        self.T = attn_input.size()[1]
        # input is B x max seq len x 512
        _query = self.query.forward(attn_input)
        _key = self.key.forward(attn_input)
        _value = self.value.forward(attn_input)

        # scaled dot product (https://www.aclweb.org/anthology/N18-2074.pdf Fig.2)
        attention_scores = torch.matmul(_query, _key.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(
            self.config.num_attention_project
        )

        # Relative attention

        # extended_attention_mask = attention_mask.to(attention_scores.device)  # fp16 compatibility
        extended_attention_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        attention_scores = attention_scores + extended_attention_mask

        # relative line
        attention_scores = attention_scores + (
                (1.0 - circulant_mask(self.T, self.relative_attention)) * -10000.0
        ).unsqueeze(0).to('cuda')

        # attention scores is B x max_seq_len x max_seq_len
        # adding bias score

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

        :param x: fnn input
        :type x: torch.Tensor
        :return: return fnn output
        :rtype: torch.Tensor
        """
        x = fnn.gelu(self.linear_1(x))
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
        x = x + self.self_attention(x,attention_mask=attention_mask)
        x = self.norm2(x)
        x = x + self.ff1(x)
        return self.norm2(x)

        # self_attn_output = self.self_attention(
        #     embed_output, attention_mask = attention_mask
        # )
        #
        # norm1_output = self.norm1(self_attn_output + embed_output)
        #
        # ff1_output = self.ff1(norm1_output)
        # norm2_output = self.norm2(ff1_output + norm1_output)
        # return norm2_output

# pretty basic, just single head. but done many times, stack to have another dimension (4 with batches).# so get stacks of B x H of attention scores T x T..
# then matrix multiply these extra stacks with the v
# (B xnh)x T xT . (Bx nh xTx hs) gives (B Nh) T x hs stacks. now  hs is set to be final dimension/ number of heads, so reorder the stacks (concatenating them)
 #original paper can have optional extra projection layer, but doing that anyway later

class MultiheadAttention(nn.Module):
    """Standard non causal MHA, Half Hugging Face/Half Karpathy implementation, no need to mask as after previous layers"""

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
        k = self.key(hidden_states).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                             2)  # (B, nh, T, hs)
        q = self.query(hidden_states).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                               2)  # (B, nh, T, hs)
        v = self.value(hidden_states).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                               2)  # (B, nh, T, hs)

        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

            attention_scores = attention_scores + attention_mask

        attention_scores = F.softmax(attention_scores, dim = -1)

        attention_scores = self.dropout(attention_scores)

        y = attention_scores @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_attn_proj)

        return y


class TransformerLayers(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.config = config

        self.subword_embedding = SubwordEmbedding(config) #12,829,692 params
        self.transformer_layers = nn.ModuleList(
            [SharedInnerBlock(config, window) for window in config.relative_attns]
        ) # 13,401,942
        self.MHA = MultiheadAttention(config)  #1,575,936

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
        #todo orthogonal initialization
        #with skip connections,
        #layer
        #normalization, and orthogonal
        #initialization
        # not much details , so presume no activation function at end. as want it to represent something. maybe no layernorm is warranted at beginning
        # not sure if want bias, this torch works out of box, random orthogonal matrix, square as all layers are same neurons
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
        #self.norm3 = LayerNorm(config.feed_forward2_hidden)
        self.final = nn.Linear(config.feed_forward2_hidden, config.num_embed_hidden)
        self.orthogonal_initialization()

    def orthogonal_initialization(self):
            for l in [self.linear_1,self.linear_2,]:#self.linear_3]:
                torch.nn.init.orthogonal_(l.weight)

    def forward(self, x: torch.Tensor, attn_msk) -> torch.Tensor:
        sentence_lengths = attn_msk.sum(1)
        # adding square root reduction projection separately as not shared. torch.Size([64, 50, 1024])

         #x is 64 47 1024
        norms = 1 / torch.sqrt(sentence_lengths.double()).float() # 64
        x = norms.unsqueeze(1) * torch.sum(x, dim = 1) # 64 x1024

        x = x + fnn.gelu(self.linear_1(self.norm1(x)))
        x = x + fnn.gelu(self.linear_2(self.norm2(x)))
        #x = x + fnn.gelu(self.linear_3(self.norm3(x)))


        return fnn.normalize(self.final(x), dim = 1, p = 2) # 64 512


if __name__ == '__main__':

    from src.config import *
    from src.dataset import *

    args = ConveRTTrainConfig()
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(args.sp_model_path)
    u = load_instances_from_reddit_json(args.dataset_path)
    RD = RedditData(u, tokenizer, 60)
    SE = SubwordEmbedding(ConveRTModelConfig())
    Total = sum(p.numel() for p in SE.parameters())
    model_config = ConveRTModelConfig()
    print(
        f"{Total:,}"
    )  # 12,829,696
    Tls = TransformerLayers(model_config)
    Total = sum(p.numel() for p in Tls.parameters())
    print(
        f"{Total:,}"
    )
    ff2 = FeedForward2(model_config)
    Total = sum(p.numel() for p in ff2.parameters())
    print(
        f"{Total:,}"
    )


    # 12,
    # dm = DataModule()
    # train_loader = dm.train_dataloader(RD)
    # iterat = iter(train_loader)
    # batch = next(iterat)
    # batch_context = batch.context
    # batch_reply = batch.reply
    #
    # tls  = TransformerLayers(model_config)
    # ff2_context = FeedForward2(model_config)
    # ff2_reply = FeedForward2(model_config)
    # rx = tls(batch_context)
    # ry = tls(batch_reply)
    # hx = ff2_context(rx, batch_context.attention_mask)
    # hy = ff2_reply(ry, batch_reply.attention_mask)
    # print(torch.norm(hx[3]))


