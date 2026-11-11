"""Load MovieLens-1M and split it by time.

Each rating is treated as a CHOICE: "user u chose movie i at time t".
The rating value (1-5 stars) is ignored.

We split by time, never at random:
    train = oldest 80% of events, validation = next 10%, test = newest 10%.
A random split would let the model learn from the future.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def load_movielens(folder="data"):
    path = Path(folder) / "ml-1m" / "ratings.dat"
    if not path.exists():
        print("Downloading MovieLens-1M ...")
        with urllib.request.urlopen(URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(folder)

    df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "rating", "time"])
    df = df.sort_values("time", kind="stable").reset_index(drop=True)

    # Embedding tables need ids 0, 1, 2, ... so we renumber users and items.
    df["user"], _ = pd.factorize(df["user"], sort=True)
    df["item"], _ = pd.factorize(df["item"], sort=True)
    return df


def split_by_time(df, val_frac=0.1, test_frac=0.1):
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # Drop val/test events with users or items never seen in training ("cold start").
    # The model has learned nothing about them, so it cannot score them.
    known_users, known_items = set(train.user), set(train.item)
    val = val[val.user.isin(known_users) & val.item.isin(known_items)]
    test = test[test.user.isin(known_users) & test.item.isin(known_items)]
    return train, val, test


def seen_matrix(df, n_users, n_items):
    """Boolean matrix (n_users x n_items): True if the user already rated the item.

    At evaluation time we remove these items from the ranking, because on
    MovieLens a user never rates the same movie twice.
    """
    seen = torch.zeros(n_users, n_items, dtype=torch.bool)
    seen[torch.tensor(df.user.values), torch.tensor(df.item.values)] = True
    return seen


def to_tensors(df):
    return torch.tensor(df.user.values), torch.tensor(df.item.values)
