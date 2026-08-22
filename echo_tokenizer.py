"""SentencePiece tokenizer used by domain-scale Echo pretraining."""

import json
import os

import sentencepiece as spm


class EchoTokenizer:
    """Small persistence wrapper around a SentencePiece model."""

    def __init__(self, model_file):
        self.model_file = model_file
        self.processor = spm.SentencePieceProcessor(model_file=model_file)

    @property
    def vocab_size(self):
        return self.processor.get_piece_size()

    @property
    def eos_id(self):
        return int(self.processor.eos_id())

    @property
    def bos_id(self):
        return int(self.processor.bos_id())

    def encode(self, text):
        return self.processor.encode(text, out_type=int)

    def decode(self, token_ids):
        return self.processor.decode(list(token_ids))

    def save_metadata(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"model_file": os.path.basename(self.model_file),
                       "vocab_size": self.vocab_size}, handle, indent=2)


def train_tokenizer(input_files, output_prefix, vocab_size=16384):
    """Train a subword vocabulary directly from newline-delimited text files."""
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        split_digits=True,
        byte_fallback=True,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
           train_extremely_large_corpus=True,
           hard_vocab_limit=False,
    )
    return EchoTokenizer(output_prefix + ".model")
