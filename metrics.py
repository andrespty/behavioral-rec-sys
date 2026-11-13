"""Evaluation.

For each held-out event (user u chose item i):
    rank      = how many items scored strictly higher than i (0 = model's top pick)
    Recall@K  = 1 if rank < K
    NDCG@K    = 1 / log2(rank + 2) if rank < K   (rewards being near the top)
    loglik    = log P(i | u)                     (how much probability the model gave i)
    ECE       = calibration error: when the model is 30% sure of its top pick,
                is that pick chosen about 30% of the time?

Ranking metrics do not change with lambda (scaling a user's scores keeps their order).
Only loglik and ECE do, so those are the metrics that judge lambda.
"""

import torch


@torch.no_grad()
def evaluate(model, users, items, seen, batch_size=1024):
    model.eval()
    totals = {"recall@10": 0.0, "recall@20": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0, "loglik": 0.0}
    confidence, correct = [], []
    emb = model.embeddings()
    for start in range(0, len(users), batch_size):
        u, i = users[start:start + batch_size], items[start:start + batch_size]
        logits = model.logits(u, emb=emb)  # shape: (batch, n_items)

        # Remove items the user already rated (but never the item being tested).
        mask = seen[u].clone()
        mask[torch.arange(len(i)), i] = False
        logits = logits.masked_fill(mask, float("-inf"))

        target = logits[torch.arange(len(i)), i]
        rank = (logits > target[:, None]).sum(dim=1)
        for k in (10, 20):
            totals[f"recall@{k}"] += (rank < k).sum().item()
            totals[f"ndcg@{k}"] += torch.where(rank < k, 1 / torch.log2(rank + 2.0), 0.0).sum().item()

        log_probs = torch.log_softmax(logits, dim=1)
        totals["loglik"] += log_probs[torch.arange(len(i)), i].sum().item()

        top_prob, top_item = log_probs.exp().max(dim=1)
        confidence.append(top_prob)
        correct.append((top_item == i).float())

    results = {k: v / len(users) for k, v in totals.items()}
    results["ece"] = calibration_error(torch.cat(confidence), torch.cat(correct))
    return results


def calibration_error(confidence, correct, n_bins=10):
    error = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            gap = (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
            error += in_bin.float().mean().item() * gap.item()
    return error
