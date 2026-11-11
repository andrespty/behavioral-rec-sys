# Behavioral recommender: how rational are users?

A recommender system for MovieLens-1M where users choose movies by
**quantal response**, the logit choice rule from behavioral game theory:

    P(user u picks movie i) = exp(λ_u · U(u,i)) / Σ_j exp(λ_u · U(u,j))

The model learns each user's utility U(u,i) from their history, and their
**rationality λ_u**: how strongly they favor higher-utility movies.
λ = 0 means random choice; large λ means always picking the best movie.

## Files (read them in this order)

| File | What it does |
| --- | --- |
| `data.py` | Downloads MovieLens-1M, renumbers ids, splits by time |
| `model.py` | Embeddings (free or LightGCN), standardized utility, λ |
| `metrics.py` | Recall@K, NDCG@K, log-likelihood, calibration error |
| `train.py` | Trains, keeps the best epoch, evaluates on test |
| `analyze.py` | Distribution of λ across users and λ vs. activity |

## How to run

```bash
pip install -r requirements.txt

python train.py --name baseline --rationality none                        # black box, no λ
python train.py --name global   --rationality global                      # one λ for everyone
python train.py --name user_gcn --rationality user --encoder lightgcn     # one λ per user
python analyze.py --name user_gcn
```

The first run downloads the data into `data/`. Results go to `results/<name>/`:
`model.pt`, `training_log.csv`, `results.json`, and (from analyze.py) `lambda_plot.png`.

## Questions

1. Does adding λ keep the ranking quality (Recall, NDCG) of the black-box model,
   while improving how well its probabilities fit (log-likelihood, calibration)?
2. Do users differ in rationality? Are more active users more rational?

## Why the utility is standardized

The probabilities only depend on λ_u · U(u,·). If U could have any scale, the
model could shrink U and grow λ and get identical predictions, so λ would be
meaningless. We rescale each user's utilities to mean 0 and standard deviation 1
across all movies. Then λ_u has a clear meaning: how strongly user u reacts to a
one-standard-deviation difference in their own utility.
