"""Step 4: train and test.

Examples:
    python train.py --name baseline  --rationality none
    python train.py --name global    --rationality global
    python train.py --name user_gcn  --rationality user --encoder lightgcn

Training objective (maximum likelihood, with a prior on per-user lambda):
    loss = - mean_k log P(i_k | u_k)  +  prior_penalty / number_of_training_events

PyTorch's cross_entropy(logits, target) is exactly - mean log softmax(logits)[target],
so it computes the first term directly from lambda_u * U(u, .).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from data import load_movielens, seen_matrix, split_by_time, to_tensors
from metrics import evaluate
from model import BehavioralRecommender, build_graph

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="global")
parser.add_argument("--encoder", choices=["free", "lightgcn"], default="free")
parser.add_argument("--rationality", choices=["none", "global", "user"], default="global")
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
out = Path("results") / args.name
out.mkdir(parents=True, exist_ok=True)

# ---- data ----
df = load_movielens()
n_users, n_items = df.user.max() + 1, df.item.max() + 1
train, val, test = split_by_time(df)
print(f"{n_users} users, {n_items} items | train {len(train)}, val {len(val)}, test {len(test)}")

train_users, train_items = to_tensors(train)
val_users, val_items = to_tensors(val)
test_users, test_items = to_tensors(test)
seen_before_val = seen_matrix(train, n_users, n_items)
seen_before_test = seen_matrix(pd.concat([train, val]), n_users, n_items)

# ---- model ----
graph = build_graph(train, n_users, n_items) if args.encoder == "lightgcn" else None
model = BehavioralRecommender(n_users, n_items, args.dim, graph, rationality=args.rationality)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ---- training loop ----
best_loglik, best_epoch, history = -float("inf"), 0, []
for epoch in range(1, args.epochs + 1):
    model.train()
    order = torch.randperm(len(train_users))
    total_loss = 0.0
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        users, items = train_users[batch], train_items[batch]

        nll = F.cross_entropy(model.logits(users), items)          # - mean log P(chosen item)
        prior = model.prior_penalty() / len(train_users)
        l2 = args.weight_decay * model.embedding.weight[users].pow(2).sum(dim=1).mean()
        loss = nll + prior + l2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += nll.item() * len(batch)

    val_metrics = evaluate(model, val_users, val_items, seen_before_val)
    row = {"epoch": epoch, "train_nll": total_loss / len(order), **{f"val_{k}": v for k, v in val_metrics.items()}}
    if args.rationality == "global":
        row["lambda"] = model.log_lambda.exp().item()
    history.append(row)
    print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    # Keep the epoch with the best validation log-likelihood; stop if no progress.
    if val_metrics["loglik"] > best_loglik:
        best_loglik, best_epoch = val_metrics["loglik"], epoch
        torch.save({"args": vars(args), "state": model.state_dict()}, out / "model.pt")
    elif epoch - best_epoch >= args.patience:
        print(f"No improvement for {args.patience} epochs, stopping.")
        break

# ---- test the best model ----
model.load_state_dict(torch.load(out / "model.pt")["state"])
test_metrics = evaluate(model, test_users, test_items, seen_before_test)
print("TEST:", {k: round(v, 4) for k, v in test_metrics.items()})

pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
with open(out / "results.json", "w") as f:
    json.dump({"args": vars(args), "best_epoch": best_epoch, "test": test_metrics}, f, indent=2)
print("Saved to", out)
