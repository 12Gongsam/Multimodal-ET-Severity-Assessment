# Multimodal ET Severity Assessment

This project includes data augmentation, model definition, training, inference, and analysis used in “End-to-End Multimodal Learning for Objective Assessment of Essential Tremor Severity Using Wearable Sensors”, submitted to JBHI.

## Model Figure

![Model figure](figure/model.png)


## Repository Layout

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

The main notebook then uses the package entry points below:

```python
from et_severity import (
    DEFAULT_DEVICE,
    prepare_training_data,
    run_multimodal_severity_loso,
    run_single_modality_loso,
)
```

Typical flow:

1. `prepare_training_data()` builds the manifest and LOSO dataloaders.
2. `run_multimodal_severity_loso()` trains multimodal severity models.
3. `run_single_modality_loso()` trains single-modality severity or task models.

Important: `prepare_training_data()` enables `split_multidirect=True` by default. This preprocessing step scans `MultiDirect` CSV files, creates split files with `_1`, `_2`, `_3` suffixes, and deletes the original file after a successful split. If you do not want the training preparation step to modify raw files, set `split_multidirect=False`.

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
