# Reference colab: https://colab.research.google.com/drive/1Zl3zSdli_epSfaoQ_HeBCuE6dkGWTowd#scrollTo=OlU317NRm4R9

# %%
# setup 
import os, sys
CHAPTER = r"chapter1_transformers"
chapter_dir = r"./" if CHAPTER in os.listdir() else os.getcwd().split(CHAPTER)[0]
sys.path.append(chapter_dir + f"{CHAPTER}/exercises")

import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

# %%

# looking at how BPE works over time:

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
# %%

# tokenized arithmetic:
print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))
# %%

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %%
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)

# %%
probs = logits.softmax(dim=-1)
print(probs.shape)

# %%
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))
# %%

next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))