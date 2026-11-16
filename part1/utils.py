import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import re
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


### Our transformation: "careless reviewer" = synonym swaps + keyboard typos
# Each eligible word (alphabetic, >= 3 letters, not a negation/stopword) is, independently:
#   - replaced by a WordNet synonym with probability SYNONYM_PROB, otherwise
#   - given one keyboard typo with probability TYPO_PROB.
# A typo is one of: hit a neighbouring QWERTY key, swap two adjacent letters,
# drop a letter, or double a letter. Capitalisation is preserved.
# IMDB's "<br />" line breaks are kept intact.

SYNONYM_PROB = 0.25
TYPO_PROB = 0.20

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

# Words we never touch: changing these could flip or blur the sentiment / grammar
PROTECTED_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere",
    "but", "the", "and", "for", "are", "was", "were", "this", "that", "with", "have",
    "has", "had", "you", "his", "her", "its", "they", "them", "she", "him", "who",
    "what", "which", "than", "then", "there", "their", "from", "into", "very", "too",
}

_rng = random.Random(0)


def _match_case(new_word, original):
    if original.isupper():
        return new_word.upper()
    if original[0].isupper():
        return new_word[0].upper() + new_word[1:]
    return new_word


def _synonym(word):
    candidates = set()
    # only the 2 most common senses, so we don't pick obscure meanings (e.g. "subtle" -> "pernicious")
    for syn in wordnet.synsets(word.lower())[:2]:
        for lemma in syn.lemmas():
            name = lemma.name()
            # single plain words only (skip multi-word lemmas like "motion_picture")
            if "_" in name or "-" in name or not name.isalpha():
                continue
            if name.lower() != word.lower():
                candidates.add(name.lower())
    if not candidates:
        return None
    return _match_case(_rng.choice(sorted(candidates)), word)


def _typo(word):
    chars = list(word)
    kind = _rng.choice(["neighbor", "swap", "drop", "double"])
    # avoid the first letter so the word stays recognisable
    i = _rng.randrange(1, len(chars))
    if kind == "neighbor" and chars[i].lower() in QWERTY_NEIGHBORS:
        new = _rng.choice(QWERTY_NEIGHBORS[chars[i].lower()])
        chars[i] = new.upper() if chars[i].isupper() else new
    elif kind == "swap" and i < len(chars) - 1:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif kind == "drop":
        del chars[i]
    else:  # double (also the fallback when the chosen edit isn't possible)
        chars.insert(i, chars[i])
    return "".join(chars)


def _transform_segment(text):
    tokens = word_tokenize(text)
    out = []
    for tok in tokens:
        if tok.isalpha() and len(tok) >= 3 and tok.lower() not in PROTECTED_WORDS:
            r = _rng.random()
            if r < SYNONYM_PROB:
                syn = _synonym(tok)
                if syn is not None:
                    out.append(syn)
                    continue
            elif r < SYNONYM_PROB + TYPO_PROB:
                out.append(_typo(tok))
                continue
        out.append(tok)
    detok = TreebankWordDetokenizer().detokenize(out)
    # the detokenizer leaves a space before sentence-internal punctuation ("ended ." -> "ended.")
    return re.sub(r" ([.,!?;:])", r"\1", detok)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Transform the text between <br /> tags separately so the tags survive tokenization
    segments = example["text"].split("<br />")
    example["text"] = "<br />".join(_transform_segment(seg) if seg.strip() else seg for seg in segments)

    ##### YOUR CODE ENDS HERE ######

    return example
