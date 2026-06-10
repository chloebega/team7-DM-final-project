# TFT Notebook Flow Report

Reviewed sources: `/root/DM_114_FinalProject.pptx (1).pdf`, `/root/TFT_notebook.ipynb`, and the generated helper scripts in `/root`.

## Assignment Context

The attached project brief describes a Spring 2026 Data Mining final project framed as a Kaggle competition for natural disaster severity prediction. The specific disaster type is drought. The training data contains daily meteorological observations by `region_id`, with 5,480 days per region, and a `score` label from 0 to 5 that appears weekly while the other six days of each week are `NaN`. The test set contains 91 observed daily weather rows for each region and no `score` column. The required output is five predicted weekly drought severity scores for each region, corresponding to the next five consecutive weeks after the 91-day test window. Kaggle evaluates submissions with MAE, and the required submission shape follows `sample_submission.csv` with `region_id` plus `pred_week1` through `pred_week5`.

The notebook attempts to solve that task by first performing exploratory analysis and feature engineering, then trying multiple modeling strategies. Its main sequence begins with daily feature engineering and an initial Temporal Fusion Transformer (TFT) that interpolates weekly labels into daily targets. After weak public leaderboard behavior, the notebook adds a post-submission review, retries with a weekly-direct TFT that better matches the Kaggle interface, and finally pivots to a LightGBM tabular pipeline that trains five direct weekly horizon models. The final visible submission workflow is the LightGBM section, which writes `lgbm_outputs/lgbm_submission.csv`.

## Quick Overview of Cells 1-52

Cells 1-24 set up the project, install/import dependencies, load `train.csv` and `test.csv`, and perform the initial EDA. These cells inspect shapes, columns, types, missing values, duplicate rows, numeric distributions, correlation matrices, highly correlated feature pairs, and raw correlations with `score`. The key early conclusion is that raw daily weather values have limited direct correlation with weekly drought severity, so temporal aggregation and lagged drought-state features should be more useful.

Cells 25-52 build the first feature-engineering layer on the daily training data. They parse the synthetic `YYYY-MM-DD` date strings manually because years such as 3004 and 3005 are outside normal pandas datetime bounds, sort rows by region and date, create region-local day indexes, define meteorological columns, and generate rolling precipitation, dry-day, temperature, heatwave, humidity, pressure, wind, seasonal, trend, interaction, and score-lag features. These cells are still mostly model-agnostic: they prepare a richer weekly training table while preserving the idea that `score` is observed weekly and should not simply be treated as ordinary missing data.

## Cell-by-Cell Flow From Cell 53

### Cell 53 - Code

This cell creates two branches of the data. It copies the full daily engineered frame into `tft_df` for later sequence modeling, while also creating `weekly_df` by keeping only rows where the weekly `score` exists. It drops helper columns from `weekly_df`, resets the index, and reports that the full daily dataset has 12,319,040 rows and 103 columns, while the weekly target-only dataset has 1,757,936 rows and no missing target scores.

### Cell 54 - Markdown

This markdown cell introduces a check of the engineered weekly dataset. It explains that the next verification will count generated features and inspect missing values, especially in score-lag features where the first weeks of each region cannot have previous score history.

### Cell 55 - Code

This cell compares the original 17 input columns against the 99 columns in `weekly_df` after feature engineering. It then counts remaining missing values and shows that only the five lagged score columns contain missing values, with missing counts increasing from `score_lag_1w` to `score_lag_5w`. The purpose is to confirm that the engineered weekly table is mostly complete and that remaining nulls are expected from lag construction.

### Cell 56 - Markdown

This cell summarizes the first feature-engineering phase. It says the notebook has transformed daily weather observations into weekly predictors using rolling sums, means, variability measures, dry-day counts, seasonal encodings, trend features, interactions, and previous-score lags. It frames the next step as validation design and baseline modeling for the five forecast horizons.

### Cell 57 - Code

This cell computes correlations between all numeric columns in `weekly_df` and the target `score`, sorted by absolute value. It shows that previous weekly scores dominate the correlation list, with `score_lag_1w` near 0.97 and progressively weaker but still strong longer lags. It also highlights meaningful weather aggregates such as 90-day precipitation, 30-day temperature range, precipitation means, dry-day counts, and interaction features.

### Cell 58 - Code

This cell plots the top 20 engineered feature correlations with drought score as a bar chart. Its purpose is visual confirmation that temporal and lagged features are more informative than the raw daily variables inspected earlier. The plot supports the notebook's move toward time-aware modeling.

### Cell 59 - Markdown

This interpretation cell states the feature-engineering takeaway: aggregated temporal patterns show stronger relationships with drought score than instantaneous observations. It motivates using features that capture accumulated meteorological conditions instead of relying only on same-day weather values.

### Cell 60 - Markdown

This cell marks a major transition titled "Final dataset rework for TFT." It signals that the notebook is about to move away from the broader weekly feature table and reshape the data for a Temporal Fusion Transformer sequence model.

### Cell 61 - Code

This cell drops most of the rolling, lag, trend, and interaction features that were just engineered, leaving `tft_df` closer to the original daily weather table plus calendar helpers. It prints 29 remaining columns, including raw weather variables, `score`, date-derived fields, dryness and heat flags, and yearly cyclic features. The intent is to make a leaner daily sequence dataset for TFT rather than a wide tabular feature matrix.

### Cell 62 - Code

This cell adds final TFT-oriented features and removes helper columns. It computes vapor pressure deficit from temperature and dew point, log-transforms precipitation with `log1p`, adds cyclic month features, and drops raw date fields plus helper flags such as `is_hot_day`, `is_dry_day`, and `tmp_90_region`. This turns the daily frame into a compact set of weather, target, time index, and cyclic calendar variables.

### Cell 63 - Code

This cell prints the post-rework dataset shape and feature list. The dataset remains 12,319,040 daily rows, but now has 22 columns: `region_id`, 14 weather variables, `score`, `day_index_region`, day-of-year sine/cosine, `vpd`, and month sine/cosine. It verifies column alignment before scaling and model setup.

### Cell 64 - Code

This cell sorts `tft_df` by `region_id` and `day_index_region`, identifies continuous features, and performs per-region forward-fill/backward-fill imputation followed by per-region `StandardScaler` scaling. It intentionally leaves `score` untouched, so the remaining 10,561,104 missing values are the sparse daily target labels. The purpose is to normalize weather inputs within each region while preserving the unresolved weekly-target problem.

### Cell 65 - Code

This cell reprints the scaled `tft_df` shape and columns. It is a simple sanity check that the per-region scaling did not change the expected 22-column schema.

### Cell 66 - Code

This cell builds a processed `test_df` that mirrors the training frame schema. It loads `test.csv`, adds a blank `score` column, parses synthetic dates, sorts by region, creates region-local day indexes, adds cyclic day and month features, computes `vpd`, log-transforms precipitation, keeps the final 22 columns, and scales continuous features per region. This prepares the 91-day test histories for later TFT inference, although the blank test score column remains conceptually difficult because the trained model later expects score-like history.

### Cell 67 - Code

This cell checks that `tft_df` and `test_df` have identical columns and compares representative feature distributions. The column alignment check returns no train-only or test-only columns, and descriptive statistics show that scaled train features have near-zero means and unit standard deviations. The purpose is to catch schema mismatches before creating model datasets.

### Cell 68 - Code

This cell counts rows per region in the processed training frame. It confirms there are 2,248 regions and shows that each listed region has 5,480 daily training rows. This verifies the expected panel structure before sequence modeling.

### Cell 69 - Markdown

This markdown cell introduces the TFT implementation, architecture, and training section. It separates the preprocessing work from the deep-learning experiment that follows.

### Cell 70 - Code

This cell installs PyTorch Lightning and PyTorch Forecasting if needed, then imports PyTorch, dataset/dataloader utilities, Lightning callbacks, `TimeSeriesDataSet`, `TemporalFusionTransformer`, `QuantileLoss`, `GroupNormalizer`, and hyperparameter tuning helpers. It prepares the environment for the first TFT attempt.

### Cell 71 - Code

This cell resolves the missing daily target problem by linearly interpolating `score` within each region. It reports 10,561,104 missing scores before interpolation and zero after interpolation. This creates a smooth daily target path, but it is also the key modeling compromise of the first TFT attempt because the Kaggle task asks for five weekly labels, not 35 interpolated daily labels.

### Cell 72 - Markdown

This cell introduces the train/validation split. It is a short marker before constructing PyTorch Forecasting datasets.

### Cell 73 - Code

This cell slices `tft_df` into training rows through day index 5114 and validation rows after day index 5023, giving validation enough history for a 91-day encoder. It defines a 91-day encoder and 35-day prediction length, sets `region_id` as a static categorical, calendar variables as known reals, and weather plus `score` as unknown reals. It then builds `TimeSeriesDataSet` objects for training and validation, producing 11,217,520 training samples and 2,248 validation samples.

### Cell 74 - Code

This cell searches `optuna_tuning` for saved checkpoints from earlier tuning runs. No checkpoints are found in that directory, so it reports that trials likely hung or crashed before completing an epoch. The purpose is diagnostic: it checks whether a previous tuning attempt left usable artifacts.

### Cell 75 - Code

This cell verifies missing values in the train and validation dataframes after interpolation. It prints zero missing values for both, confirming that the sequence dataset input is numerically complete.

### Cell 76 - Code

This cell prints the total row count in `train_df`, which is 11,498,520 rows. It is a quick scale check before more expensive training operations.

### Cell 77 - Code

This cell checks for duplicate `(region_id, day_index_region)` pairs in `tft_df`. It finds zero duplicates and reports that every row is a unique region-day combination. This guards against accidental data multiplication during feature engineering or joins.

### Cell 78 - Code

This cell installs `pyarrow`. The immediate purpose is to enable fast Parquet serialization of the large processed train and test dataframes.

### Cell 79 - Code

This cell creates `data/checkpoints`, saves `tft_df` and `test_df` as Parquet files, and reports that the state is safe from kernel restarts. It materializes the expensive preprocessing output so later scripts can reload it without rerunning all notebook cells.

### Cell 80 - Code

This cell reloads the processed Parquet checkpoints back into `tft_df` and `test_df`. It is a recovery convenience cell, allowing the notebook to resume from saved preprocessing artifacts.

### Cell 81 - Code

This large `%%writefile` cell creates `multi_gpu_sweep.py`, a standalone DDP-safe Optuna sweep script for the TFT model. The script loads the processed Parquet data, samples a configurable number of regions, builds PyTorch Forecasting dataloaders, defines an Optuna objective over TFT hyperparameters, runs distributed training with Lightning, logs to TensorBoard, saves per-trial checkpoints, and stores trial results in SQLite. It is meant to move tuning out of the notebook and into a more stable multi-GPU command-line workflow.

### Cell 82 - Markdown

This cell labels the next execution as a GPU DDP smoke test. Its role is to separate a small functional distributed run from full hyperparameter tuning.

### Cell 83 - Code

This bash cell launches `multi_gpu_sweep.py` with `torch.distributed.run` on two GPUs for one trial, one epoch, eight sampled regions, and small dataloaders. It sets CUDA, NCCL, thread, and fault-handler environment variables. The intent is to verify that the DDP training path, data loading, model construction, and checkpointing work before launching an expensive sweep.

### Cell 84 - Markdown

This cell marks the full GPU Optuna sweep section. It indicates that the next run is intended to search hyperparameters at larger scale.

### Cell 85 - Code

This bash cell runs the full two-GPU Optuna sweep using 25 trials, up to 10 epochs, 300 sampled regions, a large batch size, and multiple workers. It writes to `optuna_tuning_gpu_full`, uses the study name `tft_full_gpu_sweep`, and logs under `optuna_gpu_full`. In the notebook narrative, this run was later stopped early because it was too time-consuming and trial 4 looked better than subsequent trials.

### Cell 86 - Markdown

This markdown cell explains that the full Optuna sweep was force-stopped because 25 trials, and even 15 trials, were too expensive. It records the decision to continue with trial 4 after observing that it outperformed two later trials.

### Cell 87 - Code

This large `%%writefile` cell creates `train_best_tft_trial4.py`, a standalone final-training script for the best Optuna trial. The script loads trial 4 parameters from `optuna_tuning_gpu_full/tft_ddp_sweep.db`, rebuilds train and validation dataloaders over all 2,248 regions, constructs the TFT with the chosen hyperparameters, and trains it with Lightning DDP, checkpointing, early stopping, TensorBoard logging, and mixed precision. It saves final checkpoints under `tft_best_trial4_final`.

### Cell 88 - Code

This bash cell starts `train_best_tft_trial4.py` inside a detached `tmux` session named `tft_trial4_train`. It writes timestamped logs, maintains a latest-log symlink, sets CUDA/NCCL/threading environment variables, runs `torch.distributed.run` with two processes, and mirrors the log output into the notebook. The purpose is to let long final TFT training continue even if the notebook connection drops.

### Cell 89 - Markdown

This markdown note explains that the server connection or output pipe crashed while training, but the `tmux` session allowed the training process to continue. It introduces a separate live-monitoring cell to reconnect to the detached training output.

### Cell 90 - Code

This bash cell attaches a notebook-side monitor to the existing `tft_trial4_train` tmux pane using a FIFO and `tmux pipe-pane`. It prints recent output and then streams live output while making clear that interrupting the notebook cell stops only the monitor, not the tmux training process. Its purpose is operational resilience during long GPU training.

### Cell 91 - Code

This is an empty code cell. It appears to function as a spacer or an abandoned execution slot, and it has no computational effect on the notebook.

### Cell 92 - Markdown

This cell introduces the TFT test inference section and documents the compromises needed for the first TFT model. Because the model was trained with weather variables and `score` in `time_varying_unknown_reals`, inference must fabricate 35 future rows per region even though `test.csv` only provides 91 observed days. The cell states the approximations: future weather is filled from recent test averages, missing test score history is proxied from recent interpolated training scores, `day_index_region` continues after training, and scaling is refit from raw train data per region.

### Cell 93 - Code

This large `%%writefile` cell creates `predict_tft_trial4_test.py`, a standalone inference script for the first TFT. The script loads raw train/test data, extends each test region with 35 synthetic future daily rows, fills future weather using recent averages, creates score proxies, applies calendar/weather features and train-based per-region scaling, rebuilds training dataset metadata, loads the trial 4 checkpoint, predicts 35 daily decoder outputs, extracts days 7, 14, 21, 28, and 35, clips predictions to 0-5, and saves daily, weekly-wide, and weekly-long diagnostic prediction files.

### Cell 94 - Code

This bash cell starts `predict_tft_trial4_test.py` in a detached tmux session named `tft_trial4_predict`, using one GPU and writing timestamped logs. The recorded output shows a successful inference run that produced 2,248 prediction samples and saved files under `tft_trial4_inference_outputs`, but the preview predictions were almost all zeros or near-zero values. A later rerun was interrupted, which produced a nonzero notebook-cell exit status while the earlier successful artifacts still existed.

### Cell 95 - Code

This bash cell is a tmux live monitor for the TFT inference session, analogous to the training monitor in cell 90. It tries to attach to `tft_trial4_predict` and stream pane output through a FIFO. In the saved notebook output, the session was already stopped, so the cell reports that the tmux session is not running and exits with an error.

### Cell 96 - Code

This is another empty code cell. It does not alter state and likely served as a placeholder during experimentation.

### Cell 97 - Code

This cell opens `submission.csv`, displays the first 15 rows, rounds all prediction columns to one decimal place, and writes the rounded file back to disk. At this stage the first TFT submission contained values that were mostly zero or extremely close to zero, so rounding collapsed the visible predictions to zeros. The purpose was submission formatting, but it also made the weakness of the first TFT output obvious.

### Cell 98 - Code

This cell reloads `submission.csv` after rounding and displays the first 15 rows. It verifies that all shown predictions have become `0.0` after rounding. This is a final inspection of the first TFT submission file.

### Cell 99 - Markdown

This markdown cell contains an embedded image attachment. Based on the surrounding cells, it is most likely a screenshot used to document the first TFT submission result or Kaggle feedback, but the notebook source only preserves the attachment reference rather than descriptive text.

### Cell 100 - Markdown

This cell is a post-submission review of the first TFT pipeline. It records a public leaderboard MAE of `1.0864` and argues that the notebook did not train in the same shape that Kaggle evaluates: the model learned daily interpolated sequences and then forced outputs into five weekly predictions. It identifies the key issues as target interpolation, synthetic decoder weather and score proxies at inference time, validation mismatch, and the loss of direct alignment with the five-week submission objective.

### Cell 101 - Markdown

This cell describes what a more correct solution design would look like. It recommends building training, validation, and inference around the exact Kaggle interface: 91 observed daily rows in and five weekly scores out. It suggests either direct multi-output forecasting or a 35-day decoder evaluated only at weekly offsets, with validation that simulates missing future weather and score information.

### Cell 102 - Markdown

This cell introduces a "Kaggle-shaped TFT retry" using weekly direct forecasting. The goal is to convert daily observations into weekly rows, use a 13-week encoder equivalent to 91 days, predict a 5-week decoder directly, and keep future decoder inputs limited to known calendar or relative-time features. It states that the next script will support both training and prediction modes and will write `weekly_tft_direct_submission.csv`.

### Cell 103 - Code

This large `%%writefile` cell creates `weekly_tft_direct.py`, a standalone TFT retry script. It parses synthetic dates, builds weekly endpoint rows from daily data, creates trailing weather summaries over a 13-week encoder, fills features, splits train/validation using the last five weekly labels, optionally subsamples training windows, builds a weekly `TimeSeriesDataSet`, trains a TFT with direct five-step weekly prediction, and predicts the Kaggle-format test output from weekly aggregates. This script is the notebook's attempt to keep the TFT architecture while correcting the task shape.

### Cell 104 - Code

This bash cell starts weekly-direct TFT training inside a detached tmux session named `weekly_tft_direct_train`. It runs `weekly_tft_direct.py train` on two GPUs, uses up to 500,000 training samples, trains for up to eight epochs, and logs to `weekly_tft_direct_outputs`. The saved logs show training completed successfully, with the best checkpoint copied to `weekly_tft_direct_outputs/checkpoints/best.ckpt` and validation loss around 0.22784.

### Cell 105 - Markdown

This markdown cell contains another embedded image attachment. In context, it likely documents the weekly-direct TFT training or leaderboard result, but the notebook source only includes the image reference.

### Cell 106 - Code

This bash cell starts weekly-direct TFT inference in a detached tmux session named `weekly_tft_direct_predict`. It verifies that `weekly_tft_direct.py` and `best.ckpt` exist, then runs `weekly_tft_direct.py predict` on one GPU, writing logs and the submission file to `weekly_tft_direct_outputs/weekly_tft_direct_submission.csv`. The purpose is to generate a Kaggle-format submission from the weekly-direct TFT checkpoint.

### Cell 107 - Code

This cell displays the head of `weekly_tft_direct_outputs/weekly_tft_direct_submission.csv`, rounds the five prediction columns to one decimal place, and saves the file. The pre-rounding values are much more varied than the first TFT output, including plausible scores near 0, 1, 3, and 4. It is a formatting step before submission.

### Cell 108 - Code

This cell reloads the rounded weekly-direct TFT submission and displays its first 15 rows. It verifies the one-decimal predictions, such as `R3` around 4.0 to 3.6 and `R8` around 4.3 to 4.1. This confirms that the weekly-direct retry produced a nontrivial Kaggle-format file.

### Cell 109 - Markdown

This markdown cell contains another image attachment. Given its position after the weekly-direct submission inspection, it likely records the second TFT submission result or a related screenshot.

### Cell 110 - Markdown

This cell reviews the weekly-direct TFT result and reports that it improved only slightly, with public leaderboard MAE around `1.0152`. It diagnoses likely remaining problems: the model lacks recent drought-state inputs because test has no score history, weekly aggregation may discard useful daily patterns, validation remains limited, and TFT may be inefficient for this tabular-style forecasting task. This review motivates the later pivot to a stronger tabular approach.

### Cell 111 - Code

This cell installs architecture visualization tools and attempts to render diagrams for both TFT models: the initial daily interpolated TFT and the weekly-direct TFT. It loads checkpoints, inspects real and categorical input variables, tries `torchview` and `torchviz`, and saves DOT graph files under `architecture_renders` when PNG rendering fails because Graphviz executables are unavailable. The purpose is to generate architecture artifacts for later analysis or reporting.

### Cell 112 - Markdown

This note explains that the architecture-rendering idea was abandoned because the generated `torchviz` representations were over 10,000 lines long. It clarifies why the notebook does not continue with architecture diagrams.

### Cell 113 - Markdown

This cell introduces the final LightGBM direct weekly drought model. It describes the new section as a handoff-plan implementation: rebuild features from raw CSV files, avoid pandas datetime parsing for synthetic years, nowcast recent drought state, train direct horizon models, validate on rolling late origins, and write `lgbm_outputs/lgbm_submission.csv`.

### Cell 114 - Code

This cell installs LightGBM. It prepares the environment for the final tabular modeling pipeline.

### Cell 115 - Code

This very large `%%writefile` cell creates `lgbm_drought_pipeline.py`, the final tabular pipeline. The script loads train, test, and sample submission data; parses synthetic date parts with string slicing; verifies or sorts region/date order; derives daily weather features; extracts weekly endpoint rows; builds rolling weather, climatology, trend, and score-state features; trains a stage-1 LightGBM nowcast model for recent unlabeled drought state; trains five horizon-specific LightGBM regressors; writes validation predictions, feature importances, model files, feature-column metadata, run config, and a Kaggle-format submission. This cell is the main implementation of the final approach.

### Cell 116 - Code

This bash cell runs a LightGBM smoke test on 96 regions with fewer boosting rounds. It verifies that the full pipeline can load data, create features, train the nowcast model, train all five horizon models, write validation outputs, and generate `lgbm_outputs/smoke/lgbm_submission_smoke.csv`. The saved smoke validation MAE is about 0.441 overall, which is sufficient as a functional test rather than a final performance estimate.

### Cell 117 - Code

This bash cell runs the full LightGBM training and submission pipeline over all 2,248 regions. It uses 1,200 boosting rounds and 150 early-stopping rounds, writing logs to `logs/lgbm_full_latest.log` and final output to `lgbm_outputs/lgbm_submission.csv`. The saved validation summary reports MAE of about 0.2697, 0.2734, 0.2822, 0.2890, and 0.3120 for horizons 1 through 5, with an overall validation MAE around 0.2853.

### Cell 118 - Code

This cell loads the final LightGBM submission and validation summary for inspection. It checks that the submission file exists, prints its shape as 2,248 rows by 6 columns, displays the first rows and descriptive statistics, and displays the validation MAE by horizon if available. Its purpose is final sanity checking of shape, value range, and validation artifacts.

### Cell 119 - Code

This cell rounds the final LightGBM submission predictions to one decimal place and overwrites `lgbm_outputs/lgbm_submission.csv`. It displays the unrounded head first, then saves the rounded version. This is a submission-formatting decision, presumably to reduce noisy precision or match the target's integer/low-precision nature.

### Cell 120 - Markdown

This markdown cell contains an embedded image attachment. In context, it likely records the LightGBM submission result, a leaderboard screenshot, or a final diagnostic image, but the notebook source itself does not include explanatory text.

### Cell 121 - Markdown

This final markdown cell provides an overview of the LightGBM section. It explains that the tabular pipeline replaced the earlier TFT attempts by turning each weekly forecast origin into one feature row and training five direct LightGBM regressors, one for each forecast week. It also records the intended run order, output locations, and an important caveat: the LightGBM public result improved substantially, but internal rolling validation may still be optimistic because score-lag and validation logic may not fully simulate the public test gap.

## Overall Flow Assessment

The notebook is best understood as an iterative modeling log rather than a clean linear solution. It begins with sound EDA and useful drought-oriented feature engineering, then experiments with a daily interpolated TFT that does not match the Kaggle target shape well. The later weekly-direct TFT fixes the output shape but still struggles with missing recent drought-state information and possible overcompression of daily signals. The final LightGBM section is the most task-aligned and operationally complete part of the notebook: it trains direct five-horizon models, produces validation artifacts, saves model outputs, and writes the final submission file.
