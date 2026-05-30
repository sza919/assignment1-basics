import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
):
    frequency_table = {}
    #special_tokens = [token.encode('utf-8') for token in special_tokens]
    """Process a chunk of the file from start to end byte offsets."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode('utf-8', errors='ignore')
    pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    chunk_data_list = re.split(pattern, chunk_data)
    for stories in chunk_data_list:
        pretok = re.finditer(PAT, stories)
        for match in pretok:
            if match.group(0) not in frequency_table:
                frequency_table[match.group(0)] = 1
            else:
                frequency_table[match.group(0)] += 1
    return frequency_table

def get_frequency(pair: tuple[bytes, bytes], frequency_table: dict[str, int]) -> int:
    pair_str = (pair[0] + pair[1]).decode('utf-8', errors='ignore')
    total = 0
    for word, count in frequency_table.items():
        total += count * word.count(pair_str)
    return total
    
def merge_tokens(vocab: list[bytes], token_frequency_table: dict[tuple[int,...], int], pair_freq_table: dict[tuple[int, int], int]):
    # Find the most frequent pair
    to_merge = max(pair_freq_table, key=lambda pair: (pair_freq_table[pair], (vocab[pair[0]], vocab[pair[1]])))
    new_token_frequency_table = {}
    need_change = False
    for word, count in token_frequency_table.items():
        new_word = []
        w = len(word)
        if w == 1:
            new_token_frequency_table[word] = count
            continue
        i = 0
        while i < w - 1:
            if word[i] == to_merge[0] and word[i+1] == to_merge[1]:
                need_change = True
                new_word.append(len(vocab))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i == w - 1:
            new_word.append(word[-1])
        new_token_frequency_table[tuple(new_word)] = count
        if need_change:
            for i in range(len(word)-1):
                pair_freq_table[(word[i], word[i+1])] -= count
                if pair_freq_table[(word[i], word[i+1])] == 0:
                    del pair_freq_table[(word[i], word[i+1])]
            for i in range(len(new_word) - 1):
                if (new_word[i], new_word[i+1]) not in pair_freq_table:
                    pair_freq_table[(new_word[i], new_word[i+1])] = 0
                pair_freq_table[(new_word[i], new_word[i+1])] += count
    return new_token_frequency_table, pair_freq_table, to_merge

def initialize_pair(token_frequency_table: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pair_freq_table = {}
    for word, count in token_frequency_table.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            if pair not in pair_freq_table:
                pair_freq_table[pair] = count
            else:
                pair_freq_table[pair] += count
    return pair_freq_table

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Pretokenization to get frequency counts.
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenize_chunk, 
                           [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
    print("Finished pre-tokenization.")
    #results = [pre_tokenize_chunk(input_path, boundaries[0], boundaries[-1], special_tokens)] # process the last chunk in the main process
    frequency_table = {}
    for result in results:
        for token, count in result.items():
            if token not in frequency_table: 
                frequency_table[token] = count
            else:
                frequency_table[token] += count
    token_frequency_table = {tuple(word.encode('utf-8')): count for word, count in frequency_table.items()} # list of int -> int
    pair_freq_table = initialize_pair(token_frequency_table) # tuple of int, int -> int
    #print(list(pair_freq_table.items())[:10])
    # Initialize vocab
    merges = []
    vocab = [bytes([i]) for i in range(256)]

    for t in tqdm(range(vocab_size - 256 - len(special_tokens)), desc="Training BPE"):
        #token_frequency_table, pair_freq_table, new_merge = merge_tokens(vocab, token_frequency_table, pair_freq_table)
        
        # avoid call merge_tokens
        new_merge = max(pair_freq_table, key=lambda pair: (pair_freq_table[pair], (vocab[pair[0]], vocab[pair[1]])))
        new_token_frequency_table = {}
        for word, count in token_frequency_table.items():
            
            w = len(word)
            v = len(vocab)
            need_change = False
            if w == 1:
                new_token_frequency_table[word] = count
                continue
            i = 0
            for i in range(w - 1):
                if word[i] == new_merge[0] and word[i+1] == new_merge[1]:
                    need_change = True
                    break
            if not need_change:
                new_token_frequency_table[word] = count
                continue
            
            if need_change:
                i = 0
                new_word = []
                while i < w - 1:
                    if word[i] == new_merge[0] and word[i+1] == new_merge[1]:
                        new_word.append(v)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i == w - 1:
                    new_word.append(word[-1])
                new_token_frequency_table[tuple(new_word)] = count

                for i in range(w-1):
                    pair_freq_table[(word[i], word[i+1])] -= count
                    if pair_freq_table[(word[i], word[i+1])] == 0:
                        del pair_freq_table[(word[i], word[i+1])]
                for i in range(len(new_word) - 1):
                    if (new_word[i], new_word[i+1]) not in pair_freq_table:
                        pair_freq_table[(new_word[i], new_word[i+1])] = 0
                    pair_freq_table[(new_word[i], new_word[i+1])] += count
        token_frequency_table = new_token_frequency_table

        new_merge = (vocab[new_merge[0]], vocab[new_merge[1]])
        merges.append(new_merge)
        vocab.append(new_merge[0] + new_merge[1])
        #print(new_merge, vocab[-1])
    
    vocab = vocab + [token.encode('utf-8') for token in special_tokens]
    vocab = {i: token for i, token in enumerate(vocab)}
    return vocab, merges

import pickle

if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    os.makedirs("tokenization", exist_ok=True)

    with open("tokenization/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tokenization/merges.pkl", "wb") as f:
        pickle.dump(merges, f)