# SOLUTION.md — Zero-Order Fine-Tuning of ResNet18 on CIFAR100

**SMILES 2026 Application**

---

## Reproducibility Instructions

### Environment

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch`, `torchvision`, and `tqdm`. No additional
dependencies are needed.

### Run

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json
```

Budget: 64 x 128 = 8192 samples (exactly at the limit).
CIFAR100 is downloaded automatically to `./data` on first run.
Results are written to `results.json`.

### Reproducibility note

`validate.py` calls `seed_everything(42)` before anything else, so results
are fully deterministic on the same hardware. A tolerance of ±0.5% is allowed
across machines per the assignment specification.

---

## Final Solution Description

### What was modified

| File | What changed |
|---|---|
| `zo_optimizer.py` | Replaced per-parameter estimator with SPSA, added Adam update, added curriculum layer schedule |
| `head_init.py` | Replaced Kaiming init with small-scale Xavier init |
| `augmentation.py` | Added RandomCrop, ColorJitter, RandomRotation, RandomErasing |

### zo_optimizer.py — Core fix: SPSA instead of per-parameter estimator

The skeleton uses a 2-point central-difference estimator that calls `loss_fn`
**twice per parameter**. The fc layer alone has 100 x 512 + 100 = 51,300
parameters, requiring 102,600 forward passes per step. With any realistic
n_batches this would take hours and produce terrible results because the
gradient estimate for each parameter is based on perturbing only that
parameter while all others stay at the current value — the estimate is
extremely noisy at that budget.

**SPSA (Simultaneous Perturbation Stochastic Approximation)** fixes this by
perturbing ALL parameters at once with independent random directions:

```
u_i  ~  N(0, I)   for each active parameter i
f+   =  loss( params + eps * u )   [one forward pass]
f-   =  loss( params - eps * u )   [one forward pass]
grad_i  ~  (f+ - f-) / (2 * eps) * u_i
```

This requires exactly **2 forward passes per step** regardless of how many
parameters are being tuned. The estimator is still unbiased for the gradient
projection onto each u_i.

**Adam update** replaces the vanilla SGD step. Adam adapts the learning rate
per parameter using first and second moment estimates, which is especially
helpful when the budget is small and we cannot afford many steps to find the
right scale manually.

**Layer curriculum** unlocks deeper layers at the halfway point:

- Steps 1 to 64: tune only `fc.weight` and `fc.bias`. The head is randomly
  initialised and needs the most adjustment. Keeping the parameter set small
  reduces noise in the SPSA estimate.
- Steps 65 to 128: also tune `layer4.1` (conv weights and BN parameters).
  By this point the head is roughly calibrated, so the backbone can start
  adapting to CIFAR100 features without the gradient signal being drowned out
  by a poor head.

### head_init.py — Small-scale initialization

Xavier uniform init followed by scaling weights by 0.01. This keeps the
initial logits close to zero, so the starting cross-entropy loss is near
log(100) ~ 4.6, which is the expected loss for a random uniform predictor.
Kaiming init (the skeleton) tends to produce large initial logits, which
inflates the loss and makes early SPSA gradient estimates very noisy.

### augmentation.py — Richer training pipeline

Added on top of the skeleton:
- `RandomCrop(224, padding=28)` for translation invariance
- `ColorJitter` for colour robustness
- `RandomRotation(15)` for rotational invariance
- `RandomErasing(p=0.2)` for occlusion robustness

These help because each SPSA step sees a different augmented view of the
batch, reducing overfitting to specific pixel patterns in the training set.

### Batch size and n_batches choice

`batch_size=64, n_batches=128` (product = 8192 = budget cap).
A larger batch gives a less noisy loss estimate for each SPSA call.
More steps (128 vs 32) lets Adam accumulate better moment estimates and
lets the curriculum schedule actually reach phase 2.

---

## Experiments and Failed Attempts

### Per-parameter central-difference (skeleton, abandoned)

Running the skeleton as provided is not practical. For `fc` alone (51,300
params), each step costs 102,600 + 1 forward passes. Even one step would
take several minutes on CPU. The gradient estimate quality also degrades
badly because perturbing one parameter at a time ignores interactions.

### Tuning the full backbone from the start

Unlocking all of layer4 from step 1 inflated the SPSA estimate variance
significantly — the random directions were spread across too many parameters,
so the signal-to-noise ratio of `(f+ - f-)` was too low to produce useful
updates. The head-first curriculum mitigated this.

### Uniform perturbation mode

Replaced by Gaussian because Gaussian perturbations are isotropic in
parameter space and the resulting SPSA estimator has a clean theoretical
connection to gradient estimation. Uniform directions with renormalisation
to unit norm concentrate near the surface of the hypersphere, which can
reduce coverage of the gradient direction.

### Higher learning rate (lr=0.1)

Adam with lr=0.1 caused the fc weights to diverge within 10 steps — the
loss spiked to 5.0+ and never recovered. lr=5e-3 is stable across all runs.

### Orthogonal head initialization

`nn.init.orthogonal_` produced similar results to small-scale Xavier but
was slightly harder to tune. The small-scale Xavier approach (scale by 0.01)
was simpler and more consistent.
