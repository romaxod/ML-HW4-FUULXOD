# W&B Report draft — FER-2013 emotion classification

This is a **draft you paste into a W&B Report** (wandb.ai → your project →
Create Report). A Report mixes text with live panels pulled from your runs.
Below, each section has the narrative plus a note in `[[ ]]` telling you which
panel to add there. Replace every `___` with your real numbers after running.

---

## 1. Problem & setup

We classify 48×48 grayscale faces into seven emotions (Angry, Disgust, Fear,
Happy, Sad, Surprise, Neutral) from the FER-2013 dataset. We use the official
split (`Training` / `PublicTest` = validation / `PrivateTest` = test) so all
models are comparable. Goal of this report: not a single best score, but a
documented progression from an underfitting baseline to a well-regularized
model, with the reasons for each step.

`[[ Panel: a Markdown panel with this text, plus a Run Set table filtered to
   all three groups showing columns: model_type, n_params, best_val_acc,
   test_acc, test_f1, overfit_gap ]]`

---

## 2. Iteration 1 — MLP baseline (underfitting)

A fully-connected net on flattened pixels, with **zero spatial prior**. It
reaches train_acc = `___` and val_acc = `___` — both low, with a small gap.
That "low-and-close" signature is **underfitting**: the model lacks the
capacity / inductive bias to fit the data, so adding regularization wouldn't
help. The fix is a better prior → convolutions.

`[[ Panel: line plot of train/acc and val/acc for group MLP_Training ]]`

---

## 3. Iteration 2 — Small CNN (overfitting)

Two convolutional blocks add weight sharing and locality. Accuracy jumps to
val_acc = `___` (vs MLP's `___`). But with no regularization, the `do=0 noaug`
run shows train_acc climbing to `___` while val_acc stalls around `___` — the
**`overfit_gap` widens to `___`**. That is textbook **overfitting**: the model
memorizes training noise. The tuning runs with dropout + augmentation shrink
the gap to `___`, motivating iteration 3.

`[[ Panel A: line plot of overfit_gap for group SmallCNN_Training (all runs) ]]`
`[[ Panel B: line plot of train/acc vs val/acc for the do=0 noaug run only ]]`

---

## 4. Iteration 3 — Deep CNN + BatchNorm + Dropout (good fit)

Three VGG-style stages + BatchNorm + dropout + weight decay + augmentation +
label smoothing. Each addition targets a problem seen earlier (depth → iter-1
underfit; BN → stable deep training; dropout/L2/augmentation → iter-2 overfit).
Best run: val_acc = `___`, test_acc = `___`, with the **smallest
`overfit_gap` = `___`** — train and val now track together.

`[[ Panel A: line plot of train/acc vs val/acc for the best DeepCNN run ]]`
`[[ Panel B: the val and test confusion matrices for the best run ]]`
`[[ Panel C: the test/per_class_report table ]]`

The confusion matrix shows the dataset-specific story: Fear/Sad/Neutral are
most confused, and the rare **Disgust** class (~550 train images) has the
lowest per-class F1 (`___`).

---

## 5. Hyperparameter sweep (automated tuning)

We ran a Bayesian **W&B Sweep** over the DeepCNN (lr, weight decay, dropout,
batch size, augmentation, label smoothing). Best config found:
lr = `___`, dropout = `___`, weight_decay = `___`, augment = `___`, giving
val_acc = `___`. The parameter-importance panel shows **`___`** mattered most.

`[[ Panel A: the Sweep's parameter-importance plot ]]`
`[[ Panel B: the Sweep's parallel-coordinates plot ]]`

---

## 6. Summary & takeaways

| Model    | Params | Val acc | Test acc | Macro F1 | overfit_gap | Verdict  |
|----------|--------|---------|----------|----------|-------------|----------|
| MLP      | `___`  | `___`   | `___`    | `___`    | `___`       | underfit |
| SmallCNN | `___`  | `___`   | `___`    | `___`    | `___`       | overfit  |
| DeepCNN  | `___`  | `___`   | `___`    | `___`    | `___`       | good fit |
| DeepCNN (best sweep) | `___` | `___` | `___` | `___` | `___` | tuned |

Key lesson: the jump from MLP→CNN is about the **right inductive bias**
(fixes underfitting), and the jump from SmallCNN→DeepCNN is about
**regularization + more effective data** (fixes overfitting). The `overfit_gap`
metric made both transitions visible at a glance.

`[[ Panel: final Run Set table across all groups, sorted by test_acc ]]`
