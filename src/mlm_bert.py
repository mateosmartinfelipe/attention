from collections import defaultdict
from re import I
from attention_is_all_you_need import TransformerBlock
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable
import regex as re
from utils import TEXT, MAX_LENGTH, PERCENT_MASKING, D
import random
import numpy as np
import torch.nn as nn

"""
 “The cat is walking. The dog is barking at the tree”

then with padding, it will look like this: 

“[CLS] The cat is walking [PAD] [PAD] [PAD]. [CLS] The dog is barking at the tree.” 

"""

special_tokens = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}

# were text is a sentence , might have to have some preprocessing
sentences = TEXT.split("\n")


def format_text(text: str, reg: List[str] = ["[,!?\\-]"]) -> str:
    s = text.lower()
    for r in reg:
        s = re.sub(r, "", s)
    return s


def get_unique_words(txt: List[str]) -> List[str]:
    out = []
    for s in txt:
        out.extend(format_text(s).split(" "))
    return list(set(out))


def get_voc_and_embedings(
    words: List[str], d: int, special_tokens: Dict[str, int] = special_tokens
) -> Tuple[Dict[str, int], nn.Embedding]:
    voc = special_tokens.copy()
    i = len(special_tokens) - 1
    for word in words:
        i += 1
        voc[word] = i
    return voc, nn.Embedding(len(voc), d)


def mask(
    token_embedings: List[int], special_tokens: Dict[str, int], percent: float
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """return the mask tokens and their position"""
    try:
        start = token_embedings.index(special_tokens["[CLS]"])
    except:
        start = 0

    try:
        end = token_embedings.index(special_tokens["[PAD]"])
    except:
        end = len(token_embedings)

    n = np.floor((end - 1 - start) * percent)
    n_amended = 1 if n == 0 else int(n)
    mask_index = sorted(random.sample(range(start, end), n_amended))
    mask_tokens = [token_embedings[i] for i in mask_index]
    mask_embedings = [
        special_tokens["[MASK]"] if i in mask_index else v
        for i, v in enumerate(token_embedings)
    ]
    return torch.tensor(mask_embedings), mask_tokens, mask_index


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


def segment_embedding(max_length: int) -> torch.Tensor:
    """A position embedding gives position to each embedding in a sequence.
    +1 is for the token SEP
    """
    return torch.tensor([0] * (max_length + 1) + [1] * (max_length + 1))


def position_embedding(length: int, n=10_000, d=512) -> torch.Tensor:
    """
    where k = position in the seq
          n = constant
          d = diemnsion of the encoding space ( the same as the tokens ( aprox 512 , depending of embeding use))
    sin(k / n**(2*i/d))
    cos(k / n**(2*i/d))

    """
    assert d % 2 == 0, "d must be divisable by 2"
    P = np.zeros((length, d))
    for k in range(length):
        for i in range(int(d / 2)):
            numerator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / numerator)
            P[k, 2 * i + 1] = np.cos(k / numerator)
    return torch.tensor(P).reshape(length, d)


def get_embedings_fn(
    text: List[str],
    max_length: int,
    d: int,
    tokenize_fn: Callable[..., List[int]],
    formater_fn: Callable[..., str],
    mask_fn: Callable[..., List[int]],
    special_tokens: Dict[str, int] = special_tokens,
    percent: float = 0.15,
):
    """TODO"""
    uniqe_words = get_unique_words(text)
    voc, token_embeding_fn = get_voc_and_embedings(uniqe_words, d, special_tokens)
    # the + 2 for the cls and sep tokens in each sentence
    pos_embedings = position_embedding(max_length + 2, d=d)

    def compute(sentence: str) -> torch.tensor:
        clean_sentence = formater_fn(sentence)
        tokens = tokenize_fn(clean_sentence, max_length, voc)
        tokenized_and_masked, mask_tokens, mask_index = mask_fn(
            tokens, special_tokens, percent
        )
        token_embedding = token_embeding_fn(tokenized_and_masked)
        embedings = token_embedding + pos_embedings
        return [torch.tensor(embedings), mask_tokens, mask_index]

    return compute


emededder = get_embedings_fn(
    text=sentences,
    max_length=MAX_LENGTH,
    d=D,
    tokenize_fn=token_embedding_fn,
    formater_fn=format_text,
    mask_fn=mask,
    special_tokens=special_tokens,
    percent=PERCENT_MASKING,
)

a = emededder(sentence=sentences[0])


def batch_process(batch: List[str], special_tokens: Dict[str, int]) -> List[List[int]]:
    ...


class MLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()


tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
print(tokens_a_index)
print(tokens_b_index)
