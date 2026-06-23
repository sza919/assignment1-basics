import argparse
import torch
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, ScaledDotProductAttention, \
MultiHeadAttention, TransformerBlock, TransformerLM
from cs336_basics.training import cross_entropy, AdamW, learning_rate_from_cosine_annealing, gradient_clipping, \
get_batch, save_checkpoint, load_checkpoint

vocab_file = 'tokenization/vocab.pkl'
merges_file = 'tokenization/merges.pkl'
tokenizer_owt = Tokenizer.from_files(vocab_file, merges_file, special_tokens=None)
end_token = tokenizer_owt.encode("<|endoftext|>")[0]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--dtype', type=str, default='float32')

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/checkpoint.pt",
    )

    return parser.parse_args()


print(end_token)
def softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def top_p_sample(probs: torch.Tensor, top_p: float):
    """
    probs: shape (vocab_size,)
    top_p: e.g. 0.9
    returns one sampled token id
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # keep tokens until cumulative probability exceeds top_p
    keep_mask = cumulative_probs <= top_p

    # Always keep at least the first token that crosses top_p
    keep_mask[0] = True
    if torch.any(cumulative_probs > top_p):
        first_above = torch.where(cumulative_probs > top_p)[0][0]
        keep_mask[first_above] = True

    filtered_probs = sorted_probs[keep_mask]
    filtered_indices = sorted_indices[keep_mask]

    # renormalize
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_pos = torch.multinomial(filtered_probs, num_samples=1)

    sampled_token = filtered_indices[sampled_pos]

    return sampled_token.item()


def generate_tokens(model, prompt:list[int], context_length = 256, max_tokens = 100, temperature = 1, top_p = None):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = input_ids[:, -context_length:]  # (1, T)

            logits = model(idx_cond)                  # (1, T, vocab_size)
            logits = logits[:, -1, :]                 # (1, vocab_size)

            logits = logits / temperature
            probs = softmax(logits, dim=-1)           # (1, vocab_size)

            if top_p is not None:
                next_token_id = top_p_sample(probs[0], top_p)
                next_token = torch.tensor(
                    [[next_token_id]],
                    dtype=torch.long,
                    device=device,
                )
            else:
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if end_token is not None and next_token.item() == end_token:
                break

    return input_ids[0].tolist()
    
if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(args.checkpoint_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = TransformerLM(
        d_model = args.d_model,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        theta = args.theta,
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        num_layers = args.num_layers,
        device = device
    )

    load_checkpoint(out, model)

    completion = generate_tokens(model, list(range(1000)), args.context_length, top_p = 0.7)
    print(completion)



