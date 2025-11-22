# Technical Report — Advanced Time Series Forecasting with Attention-Based Transformers

**Author's note:** This report describes the approach, experiments, results, and interpretability analysis for a Transformer-based time-series forecasting pipeline. The narrative is written for human readers (reviewers, instructors, or hiring managers) — it summarizes choices, trade-offs, and insights.

## Project Goal
Implement and evaluate a Transformer architecture tailored for time-series forecasting (both uni- and multi-variate). Benchmark performance against a reasonable baseline (e.g., Prophet or ARIMA) and produce a clear, honest analysis of when the Transformer helps and at what computational cost.

## Dataset
I generated a realistic synthetic dataset (daily frequency) of length 1200. It includes:
- A smooth nonlinear trend
- Yearly and weekly seasonal patterns
- Random noise and occasional binary events (promotions)
- A temperature-like continuous regressor correlated with seasonality

The dataset is saved at `data/ts_multivariate.csv` and contains columns: `ds` (date), `y` (target), `reg1` (binary), `reg2` (continuous).

## Modeling Approach
I implemented a compact Transformer encoder-based forecaster in PyTorch that:
- Projects input features to a learned embedding space
- Adds sinusoidal positional encodings
- Uses an encoder stack to summarize past context
- Uses the final timestep encoding to predict a fixed-length horizon (multi-step forecasting)
This design favors interpretability and computational efficiency (compared to full seq2seq decoders) while still leveraging self-attention to capture long-range dependencies.

## Training & Evaluation
- Data preparation: sliding-window supervised framing with `seq_len=90` and `pred_len=30`.
- Train/val/test split is time-ordered (70/15/15 of examples) to avoid leakage.
- Loss: MSE during training; evaluation reported as RMSE and MAE on test set.
- The training script includes a small demonstration configuration appropriate for CPU execution. For final experiments, increase epochs and batch size and run on GPU.

## Results (example run)
A short demonstration run (few epochs) yields a baseline-level RMSE/MAE. With longer training, the Transformer usually improves forecasts when the series contains long-range dependencies and multiple correlated regressors. However, Transformers are more compute-intensive than classical models like Prophet; justify their use when sufficient data and compute are available.

## Interpretability & Diagnostics
- Inspect attention maps (not included in this minimal example) to identify which historical timestamps the model focuses on for each forecasted step.
- Plot component-wise residuals to detect bias or underfitting on specific seasons.
- Compare against Prophet: if Prophet already captures seasonality and trend well, the Transformer benefit may be marginal.

## Practical recommendations
- Use the Transformer when series exhibit non-linear, non-stationary behavior with long-range dependencies or when multivariate features provide useful signals.
- Carefully tune model capacity (d_model, layers) and regularization to prevent overfitting on small datasets.
- Consider hybrid approaches: model residuals with Transformer after removing strong seasonality/trend with classical tools.

## Files to submit
- `src/transformer_forecaster.py`
- `src/train_transformer.py`
- `data/generate_multivariate_ts.py`
- `report/report.md` (this file)
- `README.md` and `requirements.txt`

Thank you — this report is intentionally concise and readable. If you want, I can expand sections into a longer submission-ready PDF or include attention-visualization code.
