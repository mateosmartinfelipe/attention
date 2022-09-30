from os import device_encoding
from turtle import forward
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super(SelfAttention, self).__init__()
        self.emdeb_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert self.heads_dim * heads == embed_size, "Embed_size has to be div by heads"
        # Wv # the dimension of on W ij , j could be anything not
        # just the same as i.
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Wk
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Wq
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Linear layer concatenate heads to embed dimensions
        # last layer of the multihead transformer.
        self.fully_connected_out = nn.Linear(self.heads * self.heads_dim, embed_size)

    # we are sending the matrxi values , ... becouse the dimension
    # changes depending whether they are going to be used in
    # the encoder or the decoder
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        # depending were we are using the Attention ( Encoder ( source sentence)
        # or Decoder target sentence
        # the length is going to change ( of the sentence )
        value_length, key_length, query_length = (
            values.shape[1],
            keys.shape[1],
            query.shape[1],
        )
        # split embedings in self.head pieces
        # befire we just had embed no heads,head_dims
        # original (N,value_legth,embed)
        values = values.reshape(N, value_length, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_length, self.heads, self.heads_dim)
        queries = queries.reshape(N, query_length, self.heads, self.heads_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Q*Kt :
        # Q ( N , query_lenth, heads , head_dim )
        # K ( N , key_length , heads , head_dim)
        # E ( N , heads , query_length , key_length)
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # dim=3 ( key_dim) , we normalized on the quey dimension
        # showing to what key to pay attention
        attention = torch.softmax(energy / (self.emdeb_size ** (1 / 2)), dim=3)
        # attention * values
        # attention ( N, heads, query_length, key_length)
        # values ( N , values_length,heads , heads_dim)
        # out (N , query_length , heads,heads_dim)
        # THE KEY_LENGTH AND VALUE LENGTH ARE ALWAYS THE SAME
        # THE ONLY ONE CHANGING IS QUERY , BETWEEN THE ENCODER AND DECODER
        out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(
            N, query_length, self.heads * self.heads_dim
        )
        # then flatten to send
        out = self.fully_connected_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embeded_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size=embeded_size, heads=heads)
        self.norm1 = nn.LayerNorm(embeded_size)
        self.norm2 = nn.LayerNorm(embeded_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embeded_size, forward_expansion * embeded_size),
            nn.ReLU(),
            nn.Linear(embeded_size * forward_expansion, embeded_size),
        )
        self.drop_out = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # we are not using the mask atttibute
        attention = self.attention(value, key, query)
        x = self.drop_out(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.drop_out(self.norm2(forward + x))
        return torch.sigmoid(out)


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        devide,
        forward_expansion,
        dropout,
        max_length,  # we will be using it to calculate the possitional embedding
    ) -> None:
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = devide
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # this positional encoding has cahnge from simple numeric
        # to sine/cosine embedding ( rewrite )
        self.positional_embeding = nn.Embedding(max_length, embed_size)
        # why to use here the module list when we have just 1 Transformer
        # in here we need to define the number of layers
        list_layers = []
        for _ in range(0, num_layers):
            list_layers.append(
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            )
        self.layers = nn.ModuleList(list_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.positional_embeding(positions))
        query = out
        key = out
        value = out
        for layer in self.layers:
            out = layer(query, key, value, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, emde_size, heads, forward_expansion, dropout, device) -> None:
        super(DecoderBlock, self).__init__()
        self.mask_attention = SelfAttention(embed_size=emde_size, heads=heads)
        self.norm = nn.LayerNorm(emde_size)
        self.transform_block = TransformerBlock(
            embeded_size=emde_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, value, key, src_mask, target_mask):
        value_mask_attention = x
        query_mask_attention = x
        key_mask_attention = x

        attention = self.mask_attention(
            query_mask_attention, key_mask_attention, value_mask_attention, target_mask
        )
        query = self.dropout(self.norm(attention + x))
        out = self.transform_block(query, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ) -> None:
        super(Decoder, self).__init__(),
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        # needs to change
        self.position_embedding = nn.Embedding(max_length, embed_size)
        list_layers = []
        for _ in range(0, num_layers):
            list_layers.append(
                DecoderBlock(
                    emde_size=embed_size,
                    heads=heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(list_layers)
        self.feed_forward = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_key_out, enco_values_out, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_key_out, enco_values_out, src_mask, target_mask)
        out = self.feed_forward(x)
        return out


class Tranformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        targer_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansions=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=10,
    ) -> None:
        super(Tranformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            devide=device,
            forward_expansion=forward_expansions,
            dropout=dropout,
            max_length=max_length,
        )
        self.decoder = Decoder(
            target_vocab_size=target_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansions,
            dropout=dropout,
            max_length=max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.targer_pad_idx = targer_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # masking soem words to train the model
        # when we want to train the model as ssl
        # those mask tokensa re the ones it needs to predict
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N,1,1,src_length)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones(target_len, target_len)).expand(
            N, 1, target_len, target_len
        )
        # (N,1,1,src_length)
        return target_mask.to(self.device)

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, enc_src, src_mask, target_mask)
        return out
