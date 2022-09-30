from functools import partial
from operator import indexOf
from attention_is_all_you_need import TransformerBlock
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable, Optional
import regex as re
from utils import TEXT, MAX_LENGTH, PERCENT_MASKING, D
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset


# TODO : TO TRAIN BERT WE NNED TO  IMPLEMENT THE NEXT SENTENCE PART
# BOTH PREDICTIONS ARE THEN PASSED TO THE LOSS FUNCTION
# ALSO WE NEED TO IMPLEMENT THE MASKED AS IN THE PAPER 15% -> 80% , 10% RANDOM ,10% LEAVE IT THE SAME

"""
 “The cat is walking. The dog is barking at the tree”

then with padding, it will look like this: 

“[CLS] The cat is walking [PAD] [PAD] [PAD] [SEP] The dog is barking at the tree [SEP]” 

"""

special_tokens = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}

# were text is a sentence , might have to have some preprocessing
sentences = TEXT.split("\n")


def format_text(text: str, reg: List[str] = ["[,!?\\-]"]) -> str:
    s = text.lower()
    for r in reg:
        s = re.sub(r, "", s)
    return s


format_text_fn = partial(format_text, reg=["[,!?\\-]"])


def get_unique_words(txt: List[str]) -> List[str]:
    out = []
    for s in txt:
        out.extend(format_text(s).split(" "))
    return list(set(out))


def get_voc(
    text: List[str], d: int, special_tokens: Dict[str, int] = special_tokens
) -> Dict[str, int]:
    uniqe_words = get_unique_words(text)
    voc = special_tokens.copy()
    i = len(special_tokens) - 1
    for word in uniqe_words:
        i += 1
        voc[word] = i
    return voc


voc = get_voc(sentences, D, special_tokens)


class Tokenizer(nn.Module):
    def __init__(self, voc, format_text: Optional[Callable[[str], str]]) -> None:
        super(Tokenizer, self).__init__()
        self.voc = voc
        self.format_text = format_text

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        if self.format_text:
            text = self.format_text(text)
        for word in text.split(" "):
            tokens.append(voc[word])
        return tokens

    def mask(
        self, sentence_1: List[int], sentence_2: List[int], percent: float
    ) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
        """return the mask tokens and their position and their index"""
        aux_sentence = sentence_1.append(sentence_2)
        # Number of tokens to mask
        n = np.floor(len(aux_sentence) * percent)
        n_amended = 1 if n == 0 else int(n)
        # Random selection
        mask_index = sorted(random.sample(range(0, len(aux_sentence)), n_amended))
        max_index_sentence_1 = [i for i in mask_index if i <= len(sentence_1)]
        max_index_sentence_2 = [i for i in mask_index if i > len(sentence_1)]
        mask_tokens = [aux_sentence[i] for i in mask_index]
        # TODO ,to the 15% that is assing to be mask , take the 10% and assing a random token
        # Substitute original token for masked one
        mask_embedings_sentence_1 = [
            self.voc["[MASK]"] if i in max_index_sentence_1 else v
            for i, v in enumerate(sentence_1)
        ]
        mask_embedings_sentence_2 = [
            self.voc["[MASK]"] if i in max_index_sentence_2 else v
            for i, v in enumerate(aux_sentence)
        ][len(sentence_1) :]
        # Return the mask vector , the masked tokens, the index of the mask tokens in the sentece

        return (
            mask_embedings_sentence_1,
            mask_embedings_sentence_2,
            max_index_sentence_1,
            max_index_sentence_2,
            mask_tokens,
        )

    def add_special_tokens(
        self,
        sentence1: List[str],
        max_index_sentence_1: List[int],
        sentence2: Optional[List[str]],
        max_index_sentence_2: List[int],
        max_length: int,
    ) -> Tuple[List[str], List[int], Optional[List[str]], Optional[List[int]]]:
        if sentence2:
            padding_to = max(len(sentence1), len(sentence2))
            "[CLS]".append(sentence1)
            max_index_sentence_1 = max_index_sentence_1 + 1
            while len(sentence1) <= padding_to:
                sentence1.append("[PAD]")
            "[SEP]".append(sentence1)
            max_index_sentence_2 = max_index_sentence_2 + 1
            while len(sentence2) <= padding_to:
                sentence2.append("[PAD]")
            "[SEP]".append(sentence2)
            return (
                sentence1,
                sentence2,
                max_index_sentence_1.extend(max_index_sentence_2),
            )
        else:
            "[CLS]".append(sentence1)
            max_index_sentence_1 = max_index_sentence_1 + 1
            while len(sentence1) <= max_length:
                sentence1.append("[PAD]")
            "[SEP]".append(sentence1)
        return sentence1, [], max_index_sentence_1

    def forward(
        self, sentence_1: str, sentence_2: str, max_length: Optional[int]
    ) -> Tuple[List[int], List[int], List[str]]:
        sentence_1 = self.tokenize(self.format_text(sentence_1))
        sentence_2 = self.tokenize(self.format_text(sentence_2))
        (
            mask_embedings_sentence_1,
            mask_embedings_sentence_2,
            max_index_sentence_1,
            max_index_sentence_2,
            mask_tokens,
        ) = self.mask(sentence_1, sentence_2, 0.80)
        tokens_1, tokens_2, masked_index = self.add_special_tokens(
            mask_embedings_sentence_1,
            max_index_sentence_1,
            mask_embedings_sentence_2,
            max_index_sentence_2,
            max_length,
        )
        tokens_idx = [self.voc[i] for i in tokens_1.extend(tokens_2)]
        return tokens_idx, masked_index, mask_tokens


def position_embedding_fn(length: int, n=10_000, d=512) -> torch.Tensor:
    """
    where k = position in the seq
          n = constant
          d = diemnsion of the encoding space ( the same as the tokens ( aprox 512 , depending of embeding use))
    sin(k / n**(2*i/d))
    cos(k / n**(2*i/d))
    return  (length x d) matrix with the ecode for each position
    """
    assert d % 2 == 0, "d must be divisable by 2"
    P = np.zeros((length, d))
    for k in range(length):
        for i in range(int(d / 2)):
            numerator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / numerator)
            P[k, 2 * i + 1] = np.cos(k / numerator)
    return torch.tensor(P).reshape(length, d)


class Embbeding(nn.Module):
    def __init__(
        self,
        positional_embedding: torch.Tensor,
        voc: Dict[str, int],
        model_length: int,
        n_segments: int,
    ) -> None:
        super(Embbeding, self).__init__()
        self.model_length = model_length
        self.positional_embedding = positional_embedding
        self.voc_embedding = nn.Embedding(len(voc), model_length)
        self.segment_embedding = nn.Embedding(n_segments, model_length)
        self.norm = nn.LayerNorm(model_length)

    def forward(self, tokens_idx, seg):
        # where tokens_idx is the idx in the tokenizer ( including pading , sep , mask , etc ....)
        return self.norm(
            self.voc_embedding(tokens_idx)
            + self.segment_embedding(seg)
            + self.positional_embedding
        )


# BEST WAY TO USE FUNCTIONS FROM HUGGING FACE ALREADY IMPLEMENTING TOKENOIZERS:
# FOR BERT https://huggingface.co/docs/transformers/model_doc/bert : class transformers.BertTokenizer
# implements WordPiece algorithm
# hugging face recomends to use : class transformers.ByteLevelBPETokenizer as in:
# https://huggingface.co/blog/how-to-train
def token_embedding_fn(
    sentence: str, max_length: int, voc: Dict[str, int]
) -> torch.Tensor:
    """Add the special tokens and them map word to the corresponding vocabulary index"""
    token_embedding: List[int] = [voc["[CLS]"]]
    for word in sentence.split(" "):
        token_embedding.append(voc.get(word, voc["[UNK]"]))
    sentence_length = len(token_embedding) - 1
    while sentence_length < max_length:
        token_embedding.append(voc["[PAD]"])
        sentence_length += 1
    token_embedding.append(voc["[SEP]"])
    return token_embedding


"""
def get_embedings_fn(
    text: List[str],
    max_length: int,
    d: int,
    tokenize_fn: Callable[..., List[int]],
    formater_fn: Callable[..., str],
    mask_fn: Callable[..., List[int]],
    special_tokens: Dict[str, int] = special_tokens,
    percent: float = 0.15,
) -> Tuple[
    Callable[[str], Tuple[torch.Tensor, List[int], List[int], int]], Dict[str, int]
]:
    uniqe_words = get_unique_words(text)
    voc = get_voc_and_embedings(uniqe_words, d, special_tokens)
    # the + 2 for the cls and sep tokens in each sentence
    pos_embedings = position_embedding_fn(max_length + 2, d=d)

    def compute(sentence: str) -> Tuple[torch.Tensor, List[int], List[int], int]:
        clean_sentence = formater_fn(sentence)
        tokens = tokenize_fn(clean_sentence, max_length, voc)
        tokenized_and_masked, mask_tokens, mask_index = mask_fn(
            tokens, special_tokens, percent
        )
        token_embedding = token_embeding_fn(tokenized_and_masked)
        embedings = token_embedding + pos_embedings
        return [torch.tensor(embedings), mask_tokens, mask_index, len(voc)]

    return compute, voc


emededder, voc = get_embedings_fn(
    text=sentences,
    max_length=MAX_LENGTH,
    d=D,
    tokenize_fn=token_embedding_fn,
    formater_fn=format_text,
    mask_fn=mask,
    special_tokens=special_tokens,
    percent=PERCENT_MASKING,
)


def batch_process(
    batch: List[str],
    build_embeddings: Callable[..., Tuple[torch.Tensor, List[int], List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    embeddings = []
    mask_elemns = []
    mask_index = []
    for sentence in batch:
        emmbed, mask, idx, voc_size = build_embeddings(sentence=sentence)
        embeddings.append(torch.unsqueeze(emmbed, 0))
        mask_elemns.append(mask)
        mask_index.append(idx)
    labels = build_labels_and_weights(mask_elemns, voc_size)
    return (
        torch.cat(embeddings, 0),
        labels,
        mask_index,
    )


def build_labels_and_weights(
    mask_tokens: List[List[int]], voc_size: int
) -> torch.Tensor:
    batch_num = []
    voc_ecode = []
    for i, tokens in enumerate(mask_tokens):
        for token in tokens:
            batch_num.append(i)
            voc_ecode.append(token)
    index = [batch_num, voc_ecode]
    value = [1] * len(batch_num)
    return torch.sparse_coo_tensor(index, value, size=(len(mask_tokens), voc_size))


a, b, c = batch_process(sentences, emededder)
print("")


class MlmBertDatset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)


# Not too clear what it does :)
class MlmBertDatset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()

    def __iter__(self, index) -> T_co:
        return super().__getitem__(index)

"""


class Encoder(nn.Module):
    def __init__(
        self,
        layers,
        vocabulary_size,
        embeded_size,
        heads,
        dropout,
        forward_expansion,
        positional_encoding: torch.Tensor,
    ) -> None:
        super(Encoder, self).__init__()
        self.positional_encoding = positional_encoding
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(embeded_size, heads, dropout, forward_expansion)
                for _ in range(layers)
            ]
        )
        self.dense = nn.Linear(embeded_size, vocabulary_size)
        # Not needed as it is part or CrossEntropy loss fucntion
        # self.softmax = nn.Softmax(2)

    def forward(self, query, key, value):
        x = self.transformer(query, key, value, False)
        x = self.dense(x)
        return x


a = Encoder(
    layers=5,
    vocabulary_size=len(voc),
    embeded_size=D,
    heads=4,
    dropout=0.2,
    forward_expansion=4,
)

# we nned the lost function
