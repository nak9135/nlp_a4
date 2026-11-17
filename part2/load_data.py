import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

T5_CHECKPOINT = 'google-t5/t5-small'
TASK_PREFIX = 'translate English to SQL: '
MAX_SOURCE_LEN = 128   # longest NL question is 60 T5 tokens
MAX_TARGET_LEN = 512   # longest SQL query is 511 T5 tokens (incl. </s>)

# The T5 SentencePiece vocab has no '<' character (it becomes <unk>), but SQL uses it
# ("arrival_time < 1200", "<="). We swap it for a placeholder word that never occurs in the data
# and that T5 can tokenize, then swap it back after decoding. Verified: all train/dev queries
# survive the round trip exactly.
LT_PLACEHOLDER = 'lessthan'

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5TokenizerFast.from_pretrained(T5_CHECKPOINT)
    return _TOKENIZER


def encode_sql(sql):
    return sql.replace('<', LT_PLACEHOLDER)


# ---------------------------------------------------------------------------------------------
# SQL-side tokenization. Two modes:
#   't5'   (default, used for fine-tuning): the T5 SentencePiece tokenizer for the SQL too.
#   'word' (extra credit, training from scratch): every whitespace-separated SQL word seen in
#          train.sql gets its own new token id (32128, 32129, ...). This shrinks the average target
#          from ~217 T5 tokens to ~61. Words never seen in training (e.g. a flight number that only
#          appears in the question) fall back to ordinary T5 SentencePiece pieces, so any SQL can
#          still be written. The English questions always use the normal T5 tokenizer.
# ---------------------------------------------------------------------------------------------
SQL_TOKENIZER = 't5'
USE_PREFIX = True
_WORD_VOCAB = None      # list of SQL words; word i has token id WORD_OFFSET + i
_WORD_TO_ID = None
WORD_OFFSET = 32128     # size of the original t5-small embedding table
EOS_ID = 1


def configure(sql_tokenizer='t5', use_prefix=True):
    global SQL_TOKENIZER, USE_PREFIX
    assert sql_tokenizer in ('t5', 'word')
    SQL_TOKENIZER, USE_PREFIX = sql_tokenizer, use_prefix


def get_word_vocab(data_folder='data'):
    """Sorted list of all whitespace-separated words in the training SQL (built the same way every
    run, so token ids are stable between training and a later --eval_only)."""
    global _WORD_VOCAB, _WORD_TO_ID
    if _WORD_VOCAB is None:
        words = set()
        for q in load_lines(os.path.join(data_folder, 'train.sql')):
            words.update(q.split())
        _WORD_VOCAB = sorted(words)
        _WORD_TO_ID = {w: WORD_OFFSET + i for i, w in enumerate(_WORD_VOCAB)}
    return _WORD_VOCAB


def target_vocab_size():
    """Embedding/output size the model needs for the current SQL tokenizer."""
    if SQL_TOKENIZER == 'word':
        return WORD_OFFSET + len(get_word_vocab())
    return WORD_OFFSET


def encode_targets(sql_lines, tokenizer):
    """SQL strings -> list of token-id lists, each ending with </s>."""
    if SQL_TOKENIZER == 't5':
        return tokenizer([encode_sql(q) for q in sql_lines], max_length=MAX_TARGET_LEN,
                         truncation=True)['input_ids']
    get_word_vocab()
    out = []
    for q in sql_lines:
        ids = []
        for w in q.split():
            if w in _WORD_TO_ID:
                ids.append(_WORD_TO_ID[w])
            else:  # unseen word: spell it with T5 pieces (each starts with the word-boundary mark)
                ids.extend(tokenizer.encode(encode_sql(w), add_special_tokens=False))
        out.append(ids[:MAX_TARGET_LEN - 1] + [EOS_ID])
    return out


def _decode_word_ids(ids, tokenizer):
    parts, run = [], []

    def flush():
        if run:
            parts.append(tokenizer.decode(run, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False))
            run.clear()

    for t in ids:
        t = int(t)
        if t == EOS_ID:
            break
        if t >= WORD_OFFSET:
            flush()
            parts.append(_WORD_VOCAB[t - WORD_OFFSET])
        else:
            run.append(t)   # pad/start (0) and other special ids are dropped by decode()
    flush()
    return ' '.join(' '.join(parts).split()).replace(LT_PLACEHOLDER, '<')


def decode_sql(token_ids_batch):
    """Turn generated token ids back into SQL strings (inverse of the target preprocessing)."""
    tokenizer = get_tokenizer()
    if SQL_TOKENIZER == 'word':
        get_word_vocab()
        return [_decode_word_ids(ids, tokenizer) for ids in token_ids_batch]
    texts = tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    return [t.replace(LT_PLACEHOLDER, '<').strip() for t in texts]


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        tokenizer = get_tokenizer()
        # Decoder start token. We use T5's own decoder_start_token_id (= pad id 0), which is what
        # the pretrained model saw during pretraining and what model.generate() uses by default.
        self.bos_id = tokenizer.pad_token_id
        self.encoder_ids, self.decoder_targets = self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        sources = [(TASK_PREFIX if USE_PREFIX else '') + line for line in nl_lines]
        encoder_ids = tokenizer(sources, max_length=MAX_SOURCE_LEN, truncation=True)['input_ids']
        encoder_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoder_ids]

        # The test split has no SQL labels
        if split == 'test':
            return encoder_ids, None

        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        assert len(sql_lines) == len(nl_lines)
        # Targets end with </s> (the tokenizer appends it), so the model learns when to stop
        targets = encode_targets(sql_lines, tokenizer)
        targets = [torch.tensor(ids, dtype=torch.long) for ids in targets]
        return encoder_ids, targets

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.decoder_targets is None:
            return self.encoder_ids[idx], self.bos_id
        return self.encoder_ids[idx], self.decoder_targets[idx], self.bos_id


def _pad_encoder(encoder_list):
    encoder_ids = pad_sequence(encoder_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    return encoder_ids, encoder_mask


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_list, target_list, bos_list = zip(*batch)
    encoder_ids, encoder_mask = _pad_encoder(encoder_list)

    # Teacher forcing: decoder sees [BOS, y1, ..., y_{n-1}] and must predict [y1, ..., y_n(=</s>)]
    decoder_input_list = [torch.cat([torch.tensor([bos]), tgt[:-1]]) for tgt, bos in zip(target_list, bos_list)]
    decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(target_list, batch_first=True, padding_value=PAD_IDX)

    initial_decoder_inputs = torch.tensor(bos_list, dtype=torch.long).unsqueeze(1)  # B x 1
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_list, bos_list = zip(*batch)
    encoder_ids, encoder_mask = _pad_encoder(encoder_list)
    initial_decoder_inputs = torch.tensor(bos_list, dtype=torch.long).unsqueeze(1)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
