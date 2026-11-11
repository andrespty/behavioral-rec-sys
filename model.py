"""The model.

    embeddings  z_u, z_i            (free, or computed by LightGCN from the graph)
    utility     U(u, i)             standardized: mean 0, std 1 across items, per user
    choice      P(i | u) = exp(lambda_u * U(u,i)) / sum_j exp(lambda_u * U(u,j))

The last line is the logit quantal response rule; lambda_u is user u's rationality.
lambda = 0: random choice.  lambda large: always picks the highest-utility item.
"""

import math

import numpy as np
import torch
from torch import nn


def build_graph(train, n_users, n_items):
    """Normalized adjacency matrix A_hat = D^-1/2 A D^-1/2 of the user-item graph.

    Nodes: users are 0..n_users-1, items are n_users..n_users+n_items-1.
    Edge (u, i) if user u rated item i in training.
    Entry for each edge: 1 / sqrt(degree(u) * degree(i)).
    """
    pairs = np.unique(train[["user", "item"]].values, axis=0)
    u, i = pairs[:, 0], pairs[:, 1] + n_users
    rows = np.concatenate([u, i])   # both directions: the graph is undirected
    cols = np.concatenate([i, u])
    degree = np.bincount(rows, minlength=n_users + n_items)
    values = 1.0 / np.sqrt(degree[rows] * degree[cols])
    n = n_users + n_items
    return torch.sparse_coo_tensor(np.stack([rows, cols]), values, (n, n), dtype=torch.float32).coalesce()


class BehavioralRecommender(nn.Module):
    def __init__(self, n_users, n_items, dim=64, graph=None, n_layers=3, rationality="global"):
        """
        graph:       None = free embeddings (matrix factorization); a sparse matrix = LightGCN
        rationality: "none"   -> plain dot product, no lambda (black-box baseline)
                     "global" -> one lambda for everyone
                     "user"   -> one lambda per user
        """
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.graph, self.n_layers = graph, n_layers
        self.rationality = rationality

        self.embedding = nn.Embedding(n_users + n_items, dim)   # users first, then items
        nn.init.normal_(self.embedding.weight, std=0.1)

        # We learn log(lambda), so lambda = exp(...) is always positive.
        start = math.log(5.0)
        if rationality == "global":
            self.log_lambda = nn.Parameter(torch.tensor(start))
        elif rationality == "user":
            self.log_lambda = nn.Parameter(torch.full((n_users,), start))
            self.prior_mean = nn.Parameter(torch.tensor(start))   # mu in the prior below

    def embeddings(self):
        """Return (user embeddings, item embeddings)."""
        e = self.embedding.weight
        if self.graph is not None:
            # LightGCN: each layer replaces every node's vector with a weighted average
            # of its neighbors' vectors. The final embedding averages all layers.
            layers = [e]
            for _ in range(self.n_layers):
                e = torch.sparse.mm(self.graph, e)
                layers.append(e)
            e = torch.stack(layers).mean(dim=0)
        return e[: self.n_users], e[self.n_users :]

    def lam(self, users):
        if self.rationality == "global":
            return torch.exp(self.log_lambda).expand(len(users))
        if self.rationality == "user":
            return torch.exp(self.log_lambda[users])
        return torch.ones(len(users))

    def logits(self, users):
        """lambda_u * U(u, j) for every item j. Shape: (batch, n_items)."""
        z_users, z_items = self.embeddings()
        z_u = z_users[users]
        if self.rationality == "none":
            return z_u @ z_items.T

        # Standardized utility. For each user, U(u, .) has mean 0 and std 1 over items:
        #   U(u,i) = z_u . (z_i - mean) / sqrt(z_u^T Cov z_u)
        # Why: P depends only on lambda_u * U. Without fixing the spread of U, the model
        # could shrink U and grow lambda and get the same probabilities, so lambda would
        # mean nothing. Fixing the spread to 1 puts all of the scale into lambda.
        mean = z_items.mean(dim=0)
        centered = z_items - mean
        cov = centered.T @ centered / self.n_items
        spread = torch.sqrt(((z_u @ cov) * z_u).sum(dim=1) + 1e-8)
        utility = (z_u @ centered.T) / spread[:, None]
        return self.lam(users)[:, None] * utility

    def prior_penalty(self, tau=0.5):
        """Only for per-user lambda: prior log(lambda_u) ~ Normal(mu, tau^2).

        Penalty = sum_u (log lambda_u - mu)^2 / (2 tau^2). Users with little data get
        pulled toward the average; users with lots of data can move away from it.
        """
        if self.rationality != "user":
            return torch.tensor(0.0)
        return ((self.log_lambda - self.prior_mean) ** 2).sum() / (2 * tau**2)
