import argparse
import torch
from pathlib import Path
import numpy as np
import time
import wandb
from cs336_basics.transformer import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, ScaledDotProductAttention, \
MultiHeadAttention, TransformerBlock, TransformerLM
from cs336_basics.training import cross_entropy, AdamW, learning_rate_from_cosine_annealing, gradient_clipping, \
get_batch, save_checkpoint, load_checkpoint

def get_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--dtype', type=str, default='float32')

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default= 0.9)
    parser.add_argument("--beta2", type=float, default= 0.9999)
    parser.add_argument("--max_iters", type=int, default=10000)

    parser.add_argument(
    "--from_path",
    type=str,
    default="data/TinyStoriesV2-GPT4-valid_token_ids.bin",
)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/checkpoint.pt",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
    )

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--val_path", type=str, default="data/TinyStoriesV2-GPT4-valid_token_ids.bin")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-basics")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(args.eval_iters):
            batch_x, batch_y = get_batch(
                data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )

            logits = model(batch_x)
            loss = cross_entropy(logits, batch_y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out

def main():
    args = parse_args()
    start_time = time.time()

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
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
        device = device,
        dtype = get_dtype(args.dtype)
    )

    optimizer = AdamW(
        model.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        betas = (args.beta1, args.beta2)
    )

    train_data = np.memmap(args.from_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode="r")
    model.train()
    for iter_idx in range(args.max_iters):
        batch_x, batch_y = get_batch(train_data, batch_size = args.batch_size,
                          context_length = args.context_length,
                          device = device)
        optimizer.zero_grad(set_to_none = True)
        logits = model(batch_x)
        loss = cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        #print(loss.item())

        if iter_idx % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, iter_idx, out)

        if iter_idx % args.log_every == 0:
            wall_time = time.time() - start_time
            print(
                f"iter {iter_idx}: "
                f"batch train loss = {loss.item():.4f}, "
                f"time = {wall_time:.2f}s"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "batch_train_loss": loss.item(),
                        "wall_time_sec": wall_time
                    },
                    step=iter_idx,
                )

        if iter_idx % args.eval_every == 0:
            wall_time = time.time() - start_time
            losses = estimate_loss(model, train_data, val_data, args, device)
            print(
                f"iter {iter_idx}: "
                f"train loss = {losses['train']:.4f}, "
                f"val loss = {losses['val']:.4f}"
                f"time = {wall_time:.2f}s"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "eval_train_loss": losses["train"],
                        "eval_val_loss": losses["val"],
                        "wall_time_sec": wall_time,
                    },
                    step=iter_idx,
                )
    if args.use_wandb:
        wandb.finish()
if __name__ == '__main__':
    main()