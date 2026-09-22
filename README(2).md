# Trust or Short? Predicting Daily Asset Allocation Returns

This repository contains my solution notebook for the **Trust or Short?** asset allocation challenge. Given an allocation's recent returns, signed volumes, turnover, and group, the task is to predict whether its **next trading day return is positive** (`1`) or nonpositive (`0`). The competition metric is classification accuracy.

The notebook explores two complementary ideas: supervised models built from historical and cross-sectional features, and a structural reconstruction of the anonymized date order from overlapping return windows. The final notebook cell writes `improved_structural_submission.csv`. No public or private leaderboard score for that file is recorded in the notebook.

## Data and task

Each row corresponds to one `(TS, ALLOCATION)` pair, identified by `ROW_ID`. `TS` labels are anonymized and shuffled; their numerical suffixes do **not** establish chronological order. Each allocation has up to 20 prior daily returns (`RET_1` is most recent), 20 signed-volume observations, `MEDIAN_DAILY_TURNOVER`, and an anonymized `GROUP`. Training labels are real-valued next-day returns in `y_train.csv`; the prediction is `1[target > 0]`.

| Input | Role |
| --- | --- |
| `X_train.csv` | Training features; 527,073 rows in the supplied data |
| `y_train.csv` | Training next-day returns, joined by `ROW_ID` |
| `X_test.csv` | Test features; 31,870 rows in the supplied data |
| `sample_submission.csv` | Example `ROW_ID,prediction` format |

The local data used by the notebook have 44 raw feature columns, 2,522 training date labels, and 120 test date labels. Training and test share allocation identities but have no date labels in common.

## Approach

### 1. Historical and cross-sectional features

The notebook derives return and signed-volume summaries over 2, 3, 5, 10, and 20 days: means, dispersion, momentum, downside risk, trends, sign frequencies, and return-volume relationships. It also adds date-level and within-group means, standardized values, and ranks. Additional features include exponentially weighted signals, reversal, autocorrelation, historical drawdown, and turnover interactions. Date-level features use other allocations' *observed historical features* on the same date, without their target returns.

`ROW_ID` and `TS` are excluded from the supervised model matrix. `ALLOCATION` and `GROUP` are treated as categorical where supported. Train and test features are constructed separately with the same functions.

### 2. Supervised models

The notebook evaluates an SGD logistic classifier, CatBoost, LightGBM, an explainable boosting model (EBM), and MiniROCKET on two-channel return/volume histories. It uses five-fold `GroupKFold` by `TS`, so rows from a given anonymized date stay together within a fold. Test probabilities are averaged across the fitted folds. A separate date-grouped tuning/check split explores weights and a decision threshold for the logistic/CatBoost/LightGBM blend.

| Model | Notebook out-of-fold accuracy |
| --- | ---: |
| Majority class | 0.507184 |
| SGD logistic | 0.523637 |
| CatBoost | 0.527297 |
| LightGBM | 0.525468 |
| EBM | 0.523575 |
| MiniROCKET | 0.497950 |
| Three-model blend at the initial 0.5 threshold | 0.528428 |

For the subsequent blend search, the notebook reports **0.527806** accuracy on its held-out tuning/check dates. The initial blend score and tuned check score come from different selection steps and should not be read as independent final-test results. The challenge's provided LightGBM benchmark public leaderboard accuracy is **0.5079**; that is a competition benchmark, not a score for this notebook.

### 3. Inferring adjacent dates from overlapping histories

If anonymized date `u` follows date `t`, the same allocation's lagged histories should overlap:

```text
RET_1(t, S) ≈ RET_2(u, S)
RET_2(t, S) ≈ RET_3(u, S)
...
RET_19(t, S) ≈ RET_20(u, S)
```

The notebook initially averages returns across allocations on each date, standardizes the resulting 19-dimensional signatures, and uses nearest-neighbor matching to infer a successor `u`. For an allocation `S` present on that successor date, it uses the observed `RET_1(u, S)` as an estimate of the next return at `(t, S)`. This uses the **features of other rows**, including test rows, and does not use test target labels.

A later cell builds an alternative successor map using the return history of `ALLOCATION_156` as an anchor and applies it to dates with less reliable aggregate matches. The notebook reports **88.91% test-row structural coverage** for this later mapping; coverage means a successor-date return was found, **not** that 88.91% of test predictions are correct. On 38,643 comparable difficult *training* rows, the notebook reports 0.7103 accuracy for the earlier map and 0.8180 for the revised map. Those rows and the anchor choice were inspected during development, so these are diagnostic training figures, not independent validation scores.

An earlier hybrid submission uses structural matches where available and the model blend elsewhere. The final `improved_structural_submission.csv` instead uses the revised structural predictions where available and the majority training class for unmatched rows. `ROW_ID` is preserved in the original `X_test` row order.

## Reproduce the notebook

1. Place the challenge CSV files alongside the notebook and open `QRT challenge (1).ipynb` in Jupyter.
2. In its first data-loading cell, replace the original author's absolute Windows CSV paths with your own paths. The uploaded notebook refers to downloaded files with suffixes in their names; map these to the corresponding challenge files listed above.
3. Install the notebook's dependencies in your environment:

   ```bash
   python -m pip install jupyter numpy pandas scipy scikit-learn catboost lightgbm interpret aeon
   ```

4. Run cells from top to bottom. The model fits and the ensemble search can take substantial time and memory on the full data. The last cell saves `improved_structural_submission.csv` in the notebook's working directory.
5. Before uploading, confirm the CSV has exactly one `prediction` column, a `ROW_ID` index matching `X_test.csv`, 31,870 rows, and only values `0` or `1`.

The notebook also creates `ensemble_tuned_submission.csv`, `hybrid_submission.csv`, and `structural_submission.csv` in earlier cells. They represent different approaches; the final saved file is the improved structural version.

## Interpretation and limitations

- The structural approach relies on historical-window overlap and access to features from the full train/test collection. **Check the competition rules for permitted use of test features and transductive reconstruction before submitting that approach.** Its measured coverage alone does not establish a leaderboard result.
- Date-grouped cross-validation prevents the same `TS` label appearing in both sides of a supervised fold, but it is not a chronological or purged backtest: neighboring dates can still have overlapping historical windows. Further prospective validation would be needed to assess live trading performance.
- Accuracy is a sign-prediction metric. Neither these accuracy figures nor the trust/short labels account for transaction costs, borrow availability, execution, or realized trading P&L.

## Repository layout

```text
README.md                 Project overview and reproduction notes
QRT challenge (1).ipynb   Exploration, model evaluation, structural matching, and submissions
```

Challenge data and generated submissions can be kept outside the repository if their redistribution is restricted by the competition rules.
