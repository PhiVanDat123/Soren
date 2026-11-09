# Approximation diagnostics during Llama pretraining

From `llama_pretraining`, run the existing Linux/CUDA launchers:

```bash
bash scripts/60M/Soren.sh
bash scripts/60M/Muon.sh
```

Configure your WandB credentials/names in the scripts as usual. Both scripts
enable diagnostics on all matrix parameters handled by the spectral optimizer
only at optimizer steps 1, 2500, and 5000 (the final update for these launchers).
The `first_middle_last` schedule follows the configured training length.
Accumulation microsteps do not increment this schedule.
Diagnostics use SVD and add work at the sampled steps. They do not replace the
SNS/NS training update with SVD or change the learning rate or momentum.

Files are append-only UTF-8 text, one JSON record per line:

* `outputs/llama_60m_soren/approximation_logs/soren_approximation_rank<RANK>.txt`
* `outputs/llama_60m_muon/approximation_logs/muon_approximation_rank<RANK>.txt`

Read all rank files: distributed optimizers assign different matrices to each
rank. Each row identifies the absolute update step, parameter name, matrix
shape, rank, dtypes and reference. Resuming uses the training loop's restored
update step. Use a new log directory for an independent experiment; rerunning
in the same directory appends records rather than overwriting prior results.

## References

The input is the actual Nesterov-adjusted momentum after gradient clipping.
The approximate output is captured inside the optimizer, in its original
training dtype, before aspect-ratio scaling, learning rate, and weight decay.
SVD diagnostics are computed in float64 on that input/output.

For input `X = U diag(s) Vh`:

* Muon: `svd_polar` means `U @ Vh`, compared against the actual five-step quintic
  NS output (including its bfloat16 arithmetic).
* Soren: `svd_sigmoid_raw` means `U diag(sigmoid(s)) Vh`, matching the existing
  `exact_matrix_sigmoid` definition, evaluated with a float64 reference.
* Soren also logs `svd_sigmoid_fro_normalized`: its coefficients are
  `sigmoid(s / (||X||_F + 1e-8))`. The current SNS code initializes BOTH Q and T
  from the Frobenius-normalized input, whereas `exact_matrix_sigmoid` uses raw
  singular values. Keep these references distinct when reporting results.

Soren runs and Muon runs have separate training trajectories. Each row compares
the approximation and reference on exactly the same input within that run;
cross-run rows do not necessarily share input gradients.

## Metrics

All relative errors are fractions: multiply by 100 for percent.

* `relative_frobenius_error`: `||approx - exact||_F / ||exact||_F` on matrices.
* `max_per_mode_relative_error`: maximum `|a_hat_i - a_i| / a_i`, using SIGNED
  diagonal coefficients of `U.T @ approx @ V`, not sorted output singular values.
  This corresponds to Eq. (14) when the output preserves the input singular
  basis. `off_diagonal_relative_error` reports departure from that structure
  caused by finite-precision arithmetic.
* `mean_per_mode_relative_error`, `mae`, `max_absolute_error`: additional summaries
  of coefficient errors in that same input basis.
* `log_centered_max_error`, `log_centered_rmse`: errors between sorted output
  singular spectra after log2 and independent median centering.
* `width_exact`, `width_approx`, `width_difference`: log2 max-minus-min ranges
  and `width_approx - width_exact`. These directly quantify spectral widening.
* `negative_coefficients`: signed mode violations that sorted spectra conceal.
* `input_frobenius_norm`, `input_sigma_min`, `input_sigma_max`, `numerical_rank`,
  `num_modes`: input diagnostics. Numerical rank uses
  `max(shape) * float64_epsilon * sigma_max`.

Logs clamp output singular values below `1e-30` before taking log2 and report
`log_clamped_modes` and `log_floor`. Interpret widths with this floor in mind
when modes are zero. References use the full reduced SVD, as in the existing
exact Soren implementation. At rank deficiency, the extension onto null modes
is basis-dependent; inspect the rank before claiming the full-rank theoretical
bound applies.

## Options

Add these flags to `torchrun_main_HTMuon.py`:

```bash
--approximation_log_dir outputs/my_run/approximation_logs \
--approximation_log_every 0 \
--approximation_log_steps 1,2500,5000 \
--approximation_log_param_regex 'layers\.7\.'
```

The optional regex above measures only the final transformer block of the
8-layer Llama-60M. Omit it to measure every spectral optimizer matrix. Embeddings,
head, and scalar parameters assigned to auxiliary AdamW do not get spectral
error rows. Set `--approximation_log_every 1` to measure every update, or `0`
for only the explicitly selected snapshots. Without `--approximation_log_dir`,
these diagnostics are disabled. Existing spectral NPZ logging stays available
and is separate from these actual optimizer-output measurements.

Verification on CPU:

```bash
python -m unittest test_approximation_metrics -v
```

Tests cover exact references, signed mode errors, log widths/zeros, scheduling,
and unchanged updates/momentum through the two optimizer classes used by the
training entry point. Distributed ownership is emulated in the unit test;
full CUDA/DDP pretraining requires the training environment and dataset.
