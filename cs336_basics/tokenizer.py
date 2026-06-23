import pickle
import token
from typing import Iterable
import regex as re
from tqdm import tqdm
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_to_idx = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.vocab_size = len(vocab)
        
        self.special_tokens_sorted = []
        if special_tokens is not None:
            for i, token in enumerate(special_tokens):
                if token.encode('utf-8') not in self.vocab_to_idx:
                    self.vocab[self.vocab_size + i] = token.encode('utf-8')
                    self.vocab_to_idx[token.encode('utf-8')] = self.vocab_size + i
            self.special_tokens_sorted = sorted(special_tokens, key = len, reverse = True)

        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pair_to_idx = {pair: i for i, pair in enumerate(merges)}
        self.words_to_token = {}

        
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pretokenize the input text
        if len(self.special_tokens) > 0:
            pattern = '(' + '|'.join(re.escape(tok) for tok in self.special_tokens_sorted) + ')'
            text = re.split(pattern, text)
            text = [chunk for chunk in text if chunk != '']
        else:
            text = [text]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []
        for chunk in text:
            if chunk.encode('utf-8') in self.vocab_to_idx:
                tokens.append(self.vocab_to_idx[chunk.encode('utf-8')])
                continue
            pretok = re.finditer(PAT, chunk)
            for match in pretok:
                if match.group(0) in self.words_to_token:
                    tokens.extend(self.words_to_token[match.group(0)])
                    continue
                word = match.group(0)
                word_bytes = [bytes([b]) for b in word.encode('utf-8')]
                while True:
                    min_merge_idx = self.vocab_size
                    to_merge = -1
                    for i in range(len(word_bytes) - 1):
                        pair = word_bytes[i:i+2]
                        if tuple(pair) in self.pair_to_idx:
                            if self.pair_to_idx[tuple(pair)] < min_merge_idx:
                                to_merge = i
                                min_merge_idx = self.pair_to_idx[tuple(pair)]
                    if min_merge_idx == self.vocab_size:
                        break
                    word_bytes = word_bytes[0:to_merge] + [word_bytes[to_merge] + word_bytes[to_merge+1]] + word_bytes[to_merge+2:]
                self.words_to_token[match.group(0)] = [self.vocab_to_idx[byte] for byte in word_bytes]
                tokens.extend([self.vocab_to_idx[byte] for byte in word_bytes])

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # pretokenize the input 
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return text_bytes.decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    vocab_file = '/Users/ziangs/Desktop/CS336/assignment1-basics/tokenization/vocab.pkl'
    merges_file = '/Users/ziangs/Desktop/CS336/assignment1-basics/tokenization/merges.pkl'
    tokenizer_owt = Tokenizer.from_files(vocab_file, merges_file, special_tokens=None)
    input_from = '/Users/ziangs/Desktop/CS336/assignment1-basics/data/owt_train.txt'
    with open(input_from, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # with open(input_from, "r", encoding="utf-8") as f:
    #     all_ids = list(tokenizer_owt.encode_iterable(
    #         tqdm(f, total=total_lines, desc="Encoding lines")
    #     ))

    import numpy as np

    output_to = "/Users/ziangs/Desktop/CS336/assignment1-basics/data/owt_train_token_ids.bin"

    chunk = []
    chunk_size = 1_000_000

    with open(input_from, "r", encoding="utf-8") as f_in, open(output_to, "wb") as f_out:

        for token_id in tokenizer_owt.encode_iterable(tqdm(f_in, total=total_lines, desc="Encoding tokens")):
            chunk.append(token_id)

            if len(chunk) >= chunk_size:
                arr = np.array(chunk, dtype=np.uint16)
                arr.tofile(f_out)
                chunk.clear()

        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            arr.tofile(f_out)