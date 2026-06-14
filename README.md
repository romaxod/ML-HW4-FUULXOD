# FER-2013 — Facial Expression Recognition (PyTorch + Weights & Biases)

Classifying 48×48 grayscale faces into seven emotions
(0 Angry · 1 Disgust · 2 Fear · 3 Happy · 4 Sad · 5 Surprise · 6 Neutral)
for the *Challenges in Representation Learning: Facial Expression Recognition*
challenge.

The goal here is **not** a single best score. It is a documented, reproducible
progression of architectures — starting tiny and adding layers one decision at
a time — with every experiment tracked in W&B, including deliberately
**underfit** and **overfit** models and an analysis of *why* they behave that
way.

---

## Repository structure

```
.
├── README.md                        # this file — the experiment narrative
├── requirements.txt
├── model_experiment_MLP.ipynb       # iteration 1 — MLP        (expected: underfit)
├── model_experiment_SmallCNN.ipynb  # iteration 2 — Small CNN  (expected: overfit)
├── model_experiment_DeepCNN.ipynb   # iteration 3 — Deep CNN   (expected: good fit) + W&B sweep
├── model_inference.ipynb            # load best model -> submission.csv
├── submission.csv                   # produced by the inference notebook
├── WANDB_REPORT.md                  # draft text to paste into a W&B Report (bonus)
└── pictures/                        # screenshots for this README / the report
```

One notebook per architecture, mirroring the per-model layout of the previous
ensemble assignment. Each notebook is **self-contained** (setup → data → model
→ training → tuning → analysis) so it runs top-to-bottom in Colab with no
external imports. This also matches the W&B layout below: one notebook = one
model type = one W&B group.

---

## How to run (Colab)

Each `model_experiment_*.ipynb` runs top to bottom:

1. **Setup** — installs `wandb`, imports, picks GPU.
2. **Dataset** — upload your `kaggle.json` when prompted; it downloads the
   competition data. We use **`icml_face_data.csv`** because it carries the
   official split in its `Usage` column (`Training` / `PublicTest` /
   `PrivateTest`), which we map to train / validation / test. Same split for
   every model = a fair comparison.
3. **W&B login** — paste your key from https://wandb.ai/authorize.
4. **Sanity checks** — forward + backward checks (see below).
5. **Training + manual tuning** — trains a few configs, each logged as its own
   W&B run.
6. **Analysis** — markdown cell to write up what the curves show.

Run order for a submission: `DeepCNN` notebook (it saves `deep_cnn_best.pt`) →
`model_inference.ipynb` (writes `submission.csv`).

---

## W&B logging structure (mirrors the MLflow layout)

The tracking matches the MLflow structure from the brief:

| MLflow concept                  | W&B equivalent         | Here |
|---------------------------------|------------------------|------|
| Experiment (`<Model>_Training`) | `group`                | `MLP_Training`, `SmallCNN_Training`, `DeepCNN_Training` |
| Run (`Ada_n=400, lr=0.1`)       | `wandb.init(name=...)` | e.g. `SmallCNN lr=1e-3 do=0.3 aug` |
| Params table                    | `wandb.config`         | `model_type`, `config`, hyperparams, `n_params` |
| Metrics (incl. `overfit_gap`)   | `wandb.log`            | `train/*`, `val/*`, `overfit_gap`, `lr` per epoch |

Each architecture's notebook logs into its own **group**; each hyperparameter
configuration is a separate **run** in that group. Logged per epoch:
`train/loss`, `train/acc`, `val/loss`, `val/acc`, **`overfit_gap`**
(= `train_acc − val_acc`), and `lr`. Per run we also log, for richer tracking:

- **`wandb.watch`** gradient + weight histograms (the *gradients*/*parameters*
  panels — a direct view of "how the model is learning");
- **val and test confusion matrices**;
- a **per-class precision/recall/F1 table** (`test/per_class_report`);
- summary metrics (`best_val_acc`, `test_acc`, `test_f1`) in the runs table.

`overfit_gap` is the single most useful number for this assignment: watching it
grow tells you exactly when and how hard a model is overfitting.

---

## Sanity checks (run before trusting any result)

Every training notebook calls `sanity_checks(model)` before training — the
forward/backward strategies from the lectures:

1. **Forward shape** — a dummy batch produces `(B, 7)` logits.
2. **Loss at init** — random init should give cross-entropy ≈ `ln(7) ≈ 1.95`;
   a very different value means a label/logit mismatch.
3. **Backward grads** — after one `loss.backward()`, every trainable parameter
   gets a finite, non-`None` gradient (catches disconnected layers / NaNs).

If a check fails, fix the model before spending GPU time.

---

## The architecture progression (the actual experiment)

Each iteration is the previous one plus a small, motivated change. Fill in the
numbers after your runs (they come straight from the W&B summary table).

### Iteration 1 — MLP  → *expected: underfit*
Flatten 48×48 → 2304 and run dense layers. **Why:** establish the score with
**zero visual inductive bias**. **Why it underfits:** flattening destroys 2-D
locality and dense layers don't share weights, so it can't learn good facial
features. Expect train and val accuracy **both low and close** (high bias).

> train_acc = `___` · val_acc = `___` · test_acc = `___` · gap = `___`

### Iteration 2 — Small CNN  → *expected: overfit*
Two conv blocks + small head. **Why this step:** convolutions share weights and
respect locality (the right prior for images) → big jump over the MLP. **Why it
overfits:** with the `dropout=0, no-augmentation` config it memorizes the 28k
training faces. Expect **train_acc to climb while val_acc stalls/drops** —
`overfit_gap` widens. The tuning grid then shows dropout + augmentation
shrinking that gap.

> train_acc = `___` · val_acc = `___` · test_acc = `___` · gap = `___`

### Iteration 3 — Deep CNN + BatchNorm + Dropout  → *expected: good fit*
Three VGG-style stages, global average pooling, regularized head. Each addition
answers a problem seen earlier:

| Change          | Problem it addresses |
|-----------------|----------------------|
| More depth      | iter-1 underfitting (too little capacity) |
| BatchNorm       | slow/unstable training of a deeper net |
| Dropout + L2    | iter-2 overfitting |
| Augmentation    | iter-2 overfitting (more effective data) |
| Cosine/label smoothing | calibration on an imbalanced set |

Expect the **highest accuracy** with the **smallest `overfit_gap`**.

> train_acc = `___` · val_acc = `___` · test_acc = `___` · gap = `___`

---

## Hyperparameter tuning (manual)

The assignment requires tuning hyperparameters for each architecture
("ჰიპერპარამეტრების გადარჩევა"). Each notebook does this **manually**: a short
list of configs (varying learning rate, dropout, augmentation, weight decay,
label smoothing) looped so that **each config becomes its own W&B run** in that
architecture's group. You then compare them in the W&B runs table and charts.

**Automated sweep (DeepCNN notebook).** On top of the manual loop, the DeepCNN
notebook also runs a real **W&B Sweep** on our best architecture. The search
space is defined as a Python dict **in the notebook itself** (no separate
file), registered with `wandb.sweep`, and explored with `wandb.agent` using
**Bayesian** search. Each trial is its own run in `DeepCNN_Training`, and W&B
builds the **parallel-coordinates** and **parameter-importance** panels
automatically — strong material for the tracking grade and the report.

---

## Overfit vs. underfit analysis (what the brief weighs most)

The point is reading the curves, not the leaderboard number:

- **Underfitting (MLP):** low train *and* low val accuracy, small gap → lacks
  capacity / the right prior. Fix: a CNN.
- **Overfitting (SmallCNN, no regularization):** high/rising train accuracy,
  val plateaus then drops, `overfit_gap` grows → memorizing noise. Fix:
  dropout + augmentation + weight decay.
- **Good fit (DeepCNN):** train and val high and tracking together, smallest
  gap.

Read these from the `overfit_gap` and `train/acc` vs `val/acc` panels in W&B.
The **confusion matrices** add the dataset-specific story: FER-2013 models
reliably confuse **Fear / Sad / Neutral**, and the rare **Disgust** class
(~550 train images) is the hardest.

---

## Final results (fill in from W&B)

| Model    | Params | Train acc | Val acc | Test acc | Macro F1 | overfit_gap | Verdict   |
|----------|--------|-----------|---------|----------|----------|-------------|-----------|
| MLP      | `___`  | `___`     | `___`   | `___`    | `___`    | `___`       | underfit  |
| SmallCNN | `___`  | `___`     | `___`   | `___`    | `___`    | `___`       | overfit   |
| DeepCNN  | `___`  | `___`     | `___`   | `___`    | `___`    | `___`       | good fit  |

---

## Bonus: W&B Report

A **W&B Report** is an interactive document hosted on wandb.ai that mixes
written narrative with **live panels** pulling from your runs (line charts,
runs tables, confusion matrices, sweep importance plots). It stays linked to
the experiments, so it updates as runs change — think "a mini-paper that reads
your dashboard." It's created in the web UI, not in code.

`WANDB_REPORT.md` in this repo is a ready-to-use **draft**: the full narrative
with placeholders for your numbers and notes on which panel to drop in where.
Workflow: run the notebooks → open your project on wandb.ai → **Create Report**
→ paste sections from `WANDB_REPORT.md` and add the matching panels → set
sharing to "anyone with the link" → paste the links below.

> **W&B report:** `<paste link>`
> **W&B project:** `<paste link>`
