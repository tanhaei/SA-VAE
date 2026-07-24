# Methods implemented in code

## 1. Encoded record

For structured vector \(x_i\) and observation mask \(r_i\), missing entries are
filled with zero only after continuous values are standardized. The VAE input
is:

\[
u_i = [\tilde{x}_i; r_i; e_i],
\]

where \(e_i\) is an optional text embedding. Zero is not treated as observed,
because the mask is supplied separately.

Continuous means/standard deviations and categorical vocabularies are estimated
from the training partition only. Unseen categorical values map to
`__UNK__`.

## 2. NumPy VAE

The encoder and decoder each use a fully connected `tanh` hidden layer. The
encoder emits \(\mu_\phi(u)\) and diagonal \(\log\sigma^2_\phi(u)\). Training
uses:

- the reparameterization trick;
- continuous squared-error loss over observed entries;
- categorical softmax cross-entropy over observed categorical groups;
- KL regularization controlled by `beta`;
- random denoising of observed input entries while retaining their values as
  reconstruction targets;
- Adam and global gradient clipping;
- validation early stopping.

The implementation is small enough to audit and run in constrained CI. It is
not presented as a computationally optimized replacement for PyTorch/JAX.

## 3. Field-specific latent neighbors

For every imputation target:

1. mask that target in both query and donor inputs;
2. encode deterministic posterior means;
3. keep training donors for which the target is observed;
4. compute cosine similarity;
5. choose top-\(k\);
6. normalize similarities with temperature-softmax;
7. use donor values for weighted continuous/class prediction.

This prevents the target itself from determining donor selection.

## 4. Missingness simulations

- **MCAR:** constant hiding probability.
- **MAR:** logistic hiding probability based only on declared observed drivers.
- **MNAR:** logistic self-masking probability based on the target value.

The intercept is calibrated numerically so the mean masking probability matches
the requested rate. The realized finite-sample rate and probability metadata
are saved for every target and repeat.

Natural missing values remain unscored because their true values are unknown.

## 5. Statistical analysis

Metrics are paired by target and masking repeat. For each baseline:

- accuracy/macro-precision/macro-recall/macro-F1 differences are proposed
  minus baseline;
- MAE/RMSE differences are baseline minus proposed;
- positive effects therefore favor SA-VAE;
- a two-sided Wilcoxon signed-rank test is applied;
- comparisons within each metric/mechanism/rate family receive Holm correction.

Smoke runs are intentionally too small for scientific inference. Use at least
the preregistered number of repeats and report effect sizes with uncertainty in
the paper analysis.
