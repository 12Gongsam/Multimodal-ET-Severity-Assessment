# Multimodal ET Severity Assessment

This project includes data augmentation, model definition, training, inference, and analysis used in “End-to-End Multimodal Learning for Objective Assessment of Essential Tremor Severity Using Wearable Sensors”, submitted to JBHI.

## Model Figure

![Model figure](figure/model.png)


## Repository Layout

- `colab_train.py`: Colab-friendly command-line entry point for single- and multimodal LOSO training
- `notebook/Main.ipynb`: main training notebook for data preparation, LOSO training, and checkpoint export
- `notebook/Table_and_Figure.ipynb`: analysis notebook for tables, prediction summaries, and paper-style figures
- `notebook/*_legacy.ipynb`: older notebook versions kept for reference; these are attached as-is from the actual Colab experiments
- `src/et_severity/data`: manifest building, segmentation, augmentation, and dataloaders
- `src/et_severity/models`: LSTM, ResNet18-1D, TimesNet, MyWaveNet, and MIL fusion models
- `src/et_severity/training`: training loops, evaluation, and LOSO workflows
- `src/et_severity/analysis`: patient-level metrics, prediction summaries, and table builders
- `src/et_severity/visualization`: plotting utilities used by the analysis notebook
- `figure/model.png`: model overview figure

## Implemented Workflows

- Multimodal MIL training for severity prediction
- Single-modality MIL training for severity or task prediction
- Leave-one-subject-out evaluation
- Bag-level augmentation for training
- Prediction summary reports from saved CSV outputs
- Table and figure generation used by the paper notebooks

Available encoder names in the current codebase:

- `LSTM`
- `ResNet18`
- `TimesNet`
- `MyWaveNet`

## Installation

Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To run the notebooks, use Jupyter Notebook, JupyterLab, or the VS Code notebook runner. The notebooks already append `../src` to `sys.path`, so no separate package installation step is required.

## Colab Quick Start

The training entry point defaults to data under `/content/data`, so a Colab
runtime can train immediately after cloning the repository and installing the
requirements:

```python
!git clone https://github.com/12Gongsam/Multimodal-ET-Severity-Assessment.git
%cd Multimodal-ET-Severity-Assessment
!pip install -q -r requirements.txt
```

Expected files:

```text
/content/data/
  relabel_md_k5.csv
  ... sensor CSV files referenced by relabel_md_k5.csv ...
```

### Train from Colab cells

Build modality-specific loaders once, then pass a fold's train and validation
loaders directly to the training function:

```python
from et_severity import (
    EncoderConfig,
    MultimodalModelConfig,
    SingleModelConfig,
    TrainingConfig,
    build_LOSO_loaders,
    build_manifest,
    fit_multimodal,
    fit_single_modality,
)

manifest = build_manifest(
    "/content/data/relabel_md_k5.csv",
    "/content/data",
    target_col="target_k5",
)
training_config = TrainingConfig(
    epochs=100,
    learning_rate=3e-4,
    optimizer="adamw",
    monitor="macro_f1",
    early_stopping_patience=20,
)
```

Acceleration-only training:

```python
acc_loaders = build_LOSO_loaders(
    manifest,
    modality="acc",
    batch_size=16,
    target_per_class=200,
    num_workers=2,
)
acc_train_loader, acc_valid_loader = acc_loaders["1"]

single_config = SingleModelConfig(
    encoder=EncoderConfig(
        "LSTM",
        {"hidden_size": 192, "num_layers": 3, "dropout": 0.2},
    ),
    num_classes=4,
    d_model=128,
    mil_attn_dim=64,
)
single_run = fit_single_modality(
    acc_train_loader,
    acc_valid_loader,
    modality="acc",
    target="severity",
    model_config=single_config,
    training_config=training_config,
)

single_run.model
single_run.history
single_run.metrics["acc"], single_run.metrics["macro_f1"]
```

Multimodal training:

```python
multimodal_loaders = build_LOSO_loaders(
    manifest,
    modality="multimodal",
    batch_size=16,
    target_per_class=200,
    num_workers=2,
)
multimodal_train_loader, multimodal_valid_loader = multimodal_loaders["1"]

multimodal_config = MultimodalModelConfig(
    acc_encoder=EncoderConfig(
        "LSTM",
        {"hidden_size": 192, "num_layers": 3, "dropout": 0.2},
    ),
    traj_encoder=EncoderConfig(
        "TimesNet",
        {"feature_dim": 128, "d_ff": 256, "e_layers": 2, "top_k": 3},
    ),
    num_classes=4,
    d_model=128,
    cross_attention_heads=8,
    mil_attn_dim=64,
)
multimodal_run = fit_multimodal(
    multimodal_train_loader,
    multimodal_valid_loader,
    model_config=multimodal_config,
    training_config=training_config,
)
```

Additional metrics receive the complete evaluation result, including
`y_true`, `y_pred`, `logits`, `probabilities`, and bag metadata:

```python
from sklearn.metrics import f1_score

def weighted_f1(result):
    return f1_score(
        result["y_true"].numpy(),
        result["y_pred"].numpy(),
        average="weighted",
        zero_division=0,
    )

single_run = fit_single_modality(
    acc_train_loader,
    acc_valid_loader,
    modality="acc",
    model_config=single_config,
    training_config=training_config,
    metric_fns={"weighted_f1": weighted_f1},
)
single_run.metrics["weighted_f1"]
```

Convert evaluation results to a prediction table and continue into
patient-level metrics:

```python
from et_severity import (
    compute_patient_level_metrics,
    evaluation_to_prediction_frame,
)

prediction_df = evaluation_to_prediction_frame(
    single_run.metrics,
    target="severity",
)
patient_metrics = compute_patient_level_metrics(prediction_df)
display(prediction_df.head())
display(patient_metrics)
```

`TrainingRun` keeps the trained `model`, epoch `history`, final `metrics`,
`model_config`, and `training_config`. Multimodal training currently targets
severity; single-modality training accepts `target="severity"` or
`target="task"`.

Run one fold for one epoch first to verify the data paths and GPU runtime:

```python
!python colab_train.py \
  --experiments multimodal \
  --patient-ids 1 \
  --epochs 1 \
  --target-per-class 2 \
  --run-name smoke_test
```

Run full multimodal severity training:

```python
!python colab_train.py \
  --experiments multimodal \
  --epochs 100 \
  --batch-size 16 \
  --run-name multimodal_baseline
```

Run acceleration-only and trajectory-only severity experiments:

```python
!python colab_train.py \
  --experiments acc traj \
  --target severity \
  --epochs 100 \
  --run-name single_modal_baselines
```

Model parameters can be changed without editing source files:

```python
!python colab_train.py \
  --experiments multimodal \
  --acc-encoder LSTM \
  --acc-encoder-params '{"hidden_size":192,"num_layers":3,"dropout":0.2}' \
  --traj-encoder TimesNet \
  --traj-encoder-params '{"feature_dim":128,"d_ff":256,"e_layers":2,"top_k":3}' \
  --d-model 128 \
  --cross-attention-heads 8 \
  --learning-rate 0.0003 \
  --optimizer adamw \
  --monitor macro_f1 \
  --run-name multimodal_lstm_timesnet
```

Outputs are written under `/content/et_severity_runs/<run-name>/`. Each
experiment directory contains fold checkpoints, `fold_results.csv`,
`summary.json`, `histories.json`, and `run_config.json`. To keep outputs after
the Colab runtime ends, mount Google Drive and pass a Drive path:

```python
from google.colab import drive
drive.mount("/content/drive")

!python colab_train.py \
  --experiments multimodal \
  --output-dir /content/drive/MyDrive/et_severity_runs \
  --run-name multimodal_baseline
```

## Expected Local Data and Output Folders

This repository does not include raw data, prediction CSVs, or checkpoint outputs. The current notebooks expect local directories like the following next to the repository root:

```text
data/
  relabel_md_k5.csv
  calibration_power_summary.csv
  ...
predictions/
  multi/
  single/
    severity/
    task/
checkpoint3/
checkpoint4/
```

`notebook/Main.ipynb` uses:

- `data/relabel_md_k5.csv`
- raw sensor CSV files referenced by the `file` column in the label CSV
- `checkpoint3/` for multimodal checkpoints
- `checkpoint4/` for single-modality checkpoints

`notebook/Table_and_Figure.ipynb` uses:

- `data/relabel_md_k5.csv`
- `data/calibration_power_summary.csv`
- `predictions/`
- `figure/` for generated outputs such as `Table4.csv`

## Required CSV Columns

For training manifest creation, the label CSV must contain at least:

- `file`
- `patient_id`
- `session`
- `task`
- `target_k5` or another target column passed to `prepare_training_data()`

If a path in `file` is relative, it is resolved against `DATA_ROOT`.

For the analysis notebook, additional columns may be required depending on the figure being generated. The current plotting and table helpers use columns such as:

- `tetras_score`
- `log_power`
- `log_rms_delta_r`

`calibration_power_summary.csv` should include `patient_id` and either `log_power` or `normalized_power` because the notebook derives `log_power` from `normalized_power` when needed.

## Training Workflow

Open `notebook/Main.ipynb` and configure:

- `DATA_ROOT`
- `LABEL_CSV`
- `TARGET_COL`
- `FILTER_TASK`
- `TARGET_PER_CLASS`
- `BATCH_SIZE`

In Colab, the notebook uses `/content/data` when that directory exists. Expected
training files therefore live under:

```text
/content/data/
  relabel_md_k5.csv
  ... sensor CSV files referenced by the label CSV ...
```

The main notebook then uses the package entry points below:

```python
from et_severity import (
    DEFAULT_DEVICE,
    EncoderConfig,
    MultimodalModelConfig,
    SingleModelConfig,
    TrainingConfig,
    build_LOSO_loaders,
    prepare_training_data,
    run_multimodal_severity_loso,
    run_single_modality_loso,
)
```

Build loaders for the modality used by each experiment:

```python
# Multimodal: reads acceleration and trajectory columns.
manifest, multimodal_loaders, _ = prepare_training_data(
    root_dir="/content/data",
    label_csv_path="/content/data/relabel_md_k5.csv",
    modality="multimodal",
    split_multidirect=False,
)

# Acceleration only: reads accel_x, accel_y, accel_z only.
acc_loaders = build_LOSO_loaders(manifest, modality="acc")

# Trajectory only: reads coor_x, coor_y only.
traj_loaders = build_LOSO_loaders(manifest, modality="traj")
```

Model and training parameters are grouped into explicit configs:

```python
multimodal_model = MultimodalModelConfig(
    acc_encoder=EncoderConfig(
        "LSTM",
        {"hidden_size": 192, "num_layers": 3, "dropout": 0.2},
    ),
    traj_encoder=EncoderConfig(
        "TimesNet",
        {"feature_dim": 128, "d_ff": 256, "e_layers": 2, "top_k": 3},
    ),
    d_model=128,
    cross_attention_heads=8,
    mil_attn_dim=96,
)
training = TrainingConfig(
    epochs=100,
    learning_rate=3e-4,
    optimizer="adamw",
    monitor="macro_f1",
)

multi_results, multi_summary, _ = run_multimodal_severity_loso(
    multimodal_loaders,
    model_config=multimodal_model,
    training_config=training,
    device=DEFAULT_DEVICE,
)

single_model = SingleModelConfig(
    encoder=EncoderConfig(
        "MyWaveNet",
        {"residual_channels": 96, "skip_channels": 160, "n_stacks": 3},
    ),
    num_classes=4,
    d_model=128,
)
single_results, single_summary, _ = run_single_modality_loso(
    acc_loaders,
    model_config=single_model,
    modality="acc",
    target="severity",
    training_config=training,
    device=DEFAULT_DEVICE,
)
```

Encoder feature dimensions may differ from the MIL `d_model`; the model inserts
a learned projection automatically. Unsupported parameter names raise an error
instead of being silently ignored.

Supported encoder parameters:

- `LSTM`: `hidden_size`, `num_layers`, `dropout`
- `ResNet18`: `feature_dim`
- `TimesNet`: `feature_dim`, `seq_len`, `d_ff`, `e_layers`, `top_k`, `f_min`, `dropout`
- `MyWaveNet`: `residual_channels`, `skip_channels`, `dilations`, `n_stacks`

Typical flow:

1. `prepare_training_data()` builds the manifest and LOSO dataloaders.
2. `run_multimodal_severity_loso()` trains multimodal severity models.
3. `run_single_modality_loso()` trains single-modality severity or task models.

`split_multidirect=False` is the safe default. Enabling it scans `MultiDirect`
CSV files, creates `_1`, `_2`, and `_3` split files, and deletes an original
file after a successful split.

## Analysis Workflow

Open `notebook/Table_and_Figure.ipynb` and configure:

- `DATA_DIR`
- `PREDICTIONS_DIR`
- `FIGURE_DIR`

The analysis notebook currently performs:

- session-level table generation with `build_session_severity_table_from_csv()`
- multimodal prediction summary collection
- single-modality prediction summary collection
- single-vs-multimodal accuracy comparison
- scatter, boxplot, and GMM-based figure generation

The prediction summary helpers expect filenames like:

```text
predictions/multi/ACC_<ACC_MODEL>_Traj_<TRAJ_MODEL>_val_predictions.csv
predictions/single/<target>/<MODEL>_<modality>_<target>_val_predictions.csv
```
