# Early Prediction of Lifestyle Diseases Using Explainable Machine Learning (XAI)

A local Streamlit research application built from `Lifestyle_Disease_XAI_Experiment.ipynb` and the supplied paper, *Early Prediction of Lifestyle Diseases Using Explainable Machine Learning*. It retains the notebook's diabetes and Cleveland heart disease workflows, trains the six requested classifiers from the original datasets, and includes the paper's original Figures 4-6 and reported test/CV/SHAP results as clearly labeled references.

> **Educational use only.** This is not a diagnostic or screening tool. Historical-data model probabilities are not confirmed medical risks. Consult a qualified healthcare professional for medical decisions.

## Requirements

- Windows 10/11
- Python 3.10 or newer (Python 3.12 recommended)
- Internet access for the initial open-source package installation and dataset download
- The two source dataset files below

## Dataset setup

Place the files in the project's `datasets` directory:

| Required filename | Dataset | Source |
|---|---|---|
| `pima-indians-diabetes.data.txt` | Pima Indians Diabetes, 768 records, 9 comma-separated columns including `Outcome` | [Kaggle dataset page (UCIML)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database); a commonly used data mirror is [Jason Brownlee's dataset repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv). Download the headerless data and save/rename it to the required filename. |
| `processed.cleveland.txt` | UCI Cleveland processed heart disease data, 303 records, 14 comma-separated columns | [Original UCI data file](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data). Download and save/rename it to the required filename. |

The app validates column count, numeric values, target labels, and class counts. It does not create or substitute synthetic medical data. If a file is missing or invalid, the app explains what to fix and provides its source link.

## Install on Windows

Open PowerShell or a VS Code terminal in this project folder.

Option A — run the included setup script:

```bat
setup.bat
```

Option B — run commands manually:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## Train and run

After both source files are in `datasets`:

```powershell
python -m src.train_models
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`) in your browser. The app also trains models automatically on its first start when valid dataset files are present. Later starts reuse saved model and evaluation files.

Train just one task:

```powershell
python -m src.train_models --disease diabetes
python -m src.train_models --disease heart
```

## Notebook methodology retained

- Pima target remains `Outcome` (0/1); Cleveland target values 0-4 become 0 = no disease and 1 = any disease.
- Both tasks use a stratified 80/20 train/test split with `random_state=42`.
- Diabetes zero values for glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing.
- Every estimator uses median imputation; Logistic Regression also uses `StandardScaler`.
- Logistic Regression (`max_iter=1000`), Random Forest (150 trees, depth 10, balanced class weights), and XGBoost (100 trees, depth 3, learning rate 0.1, subsample/column sample 0.8, log-loss) use the notebook's parameters.
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, classification reports, and stratified five-fold F1 scores are measured from actual runs. No test split is used to fit a model.
- SHAP uses the trained Random Forest pipeline and held-out examples. SHAP attributions explain model output, not causation.

The original paper's Figures 4-6 are available under `assets/paper/`. Reported values from the paper are presented separately from local model results; local runs can differ due to dependency versions or experiment environments. Models (`models/*.pkl`), evaluation data, and datasets are local files and are excluded from version control by `.gitignore`.

## Project layout

```text
Lifestyle_Disease_XAI/
├── app.py
├── requirements.txt
├── setup.bat
├── datasets/
├── models/                  # generated after training
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── explainability.py
├── ui_pages/
├── assets/
│   ├── style.css
│   └── paper/               # original figures extracted from the supplied PDF
└── tests/
```

## Limitations

These are small, historical benchmark datasets and may not represent present-day or local populations. Model probabilities are not calibrated risk estimates. The app has no clinical validation and must not be used to diagnose or make treatment decisions.
