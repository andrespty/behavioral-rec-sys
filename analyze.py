"""Step 5: look at the estimated rationality of each user.

Usage (after training with --rationality user):
    python analyze.py --name user_gcn

Prints how lambda is distributed, how it relates to user activity, and saves a plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data import load_movielens, split_by_time
from model import BehavioralRecommender, build_graph

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="user_gcn")
folder = Path("results") / parser.parse_args().name

saved = torch.load(folder / "model.pt")
args = saved["args"]
if args["rationality"] != "user":
    raise SystemExit("This analysis needs a model trained with --rationality user")

df = load_movielens()
n_users, n_items = df.user.max() + 1, df.item.max() + 1
train, _, _ = split_by_time(df)
graph = build_graph(train, n_users, n_items) if args["encoder"] == "lightgcn" else None
model = BehavioralRecommender(n_users, n_items, args["dim"], graph, rationality="user")
model.load_state_dict(saved["state"])

users = pd.DataFrame({
    "lambda": model.log_lambda.detach().exp().numpy(),
    "train_events": train.user.value_counts().reindex(range(n_users), fill_value=0).values,
})
users = users[users.train_events > 0]
users.to_csv(folder / "lambda_per_user.csv", index_label="user")

print(users["lambda"].describe().round(3))
print("\nSpearman correlation between lambda and activity:",
      round(users["lambda"].corr(users.train_events, method="spearman"), 3))
users["activity_quintile"] = pd.qcut(users.train_events.rank(method="first"), 5, labels=["Q1 least", "Q2", "Q3", "Q4", "Q5 most"])
print("\n", users.groupby("activity_quintile", observed=True)["lambda"].agg(["mean", "median", "count"]).round(3))

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.hist(users["lambda"], bins=50)
left.set(title="How rational are users?", xlabel="estimated λ", ylabel="number of users")
right.scatter(users.train_events, users["lambda"], s=4, alpha=0.4)
right.set(title="Rationality vs. activity", xlabel="training events (log scale)", ylabel="estimated λ", xscale="log")
fig.tight_layout()
fig.savefig(folder / "lambda_plot.png", dpi=150)
print("\nSaved plot to", folder / "lambda_plot.png")
