# Parkinson's MLOps Project

End-to-end Parkinson's disease prediction project using a classical ML model (`RandomForestClassifier`) with a Gradio interface and GitHub Actions-based CI/CD.

This repository currently provides:
- Model training workflow (Colab script in `notebooks/`)
- Saved model artifact (`model.pkl`)
- Interactive web inference app (`app.py`)
- Modular inference utilities in `src/`
- CI checks and auto-deployment to a Hugging Face Space

## Project Goal

Predict whether a voice sample indicates Parkinson's disease (`1`) or not (`0`) from biomedical voice features.

## Repository Structure

```text
.
|-- app.py                    # Gradio application (loads model.pkl and serves UI)
|-- model.pkl                 # Trained RandomForest model artifact
|-- requirements.txt          # Python dependencies
|-- notebooks/
|   `-- train_colab.py        # Colab-oriented training script
|-- src/
|   |-- inference.py          # Programmatic inference pipeline (config-driven)
|   |-- model.py              # Model wrapper class (joblib + yaml config)
|   `-- preprocess.py         # Input preprocessing helper
`-- .github/workflows/
    |-- ci.yml                # Lint/test workflow
    `-- deploy.yml            # Deploys app/model/readme to Hugging Face Space
```

## How the Project Works

### 1) Training

Training is done in `notebooks/train_colab.py`:
- Upload Parkinson's dataset (Kaggle voice dataset)
- Rename uploaded file to `parkinsons.csv`
- Drop `name` column
- Use `status` as target
- Train `RandomForestClassifier`
- Evaluate with accuracy score
- Save trained model as `model.pkl`

### 2) Inference (UI Path)

`app.py`:
- Loads `model.pkl` with `joblib`
- Accepts a comma-separated string of numeric features
- Converts values to `float`
- Runs `model.predict([input_list])`
- Returns human-readable output:
  - `0 (No Parkinson's detected)`
  - `1 (Parkinson's detected)`

### 3) Inference (Code Path)

`src/inference.py` + `src/model.py` + `src/preprocess.py` define a modular inference pipeline:
- Reads feature list and model path from `config/config.yaml`
- Builds a DataFrame in expected feature order
- Predicts via `ParkinsonModel`

Note: This config file is not yet included in the repository, so the `src/` pipeline is a scaffold for structured inference and extension.

## Core Features

- Binary Parkinson's prediction from voice features
- Lightweight and fast model artifact (`joblib` serialized)
- Simple Gradio frontend for quick manual inference
- Reusable Python inference modules for integration
- CI workflow for linting/testing checks
- Deployment workflow to Hugging Face Spaces

## Expected Input Format

The current Gradio app expects a single comma-separated row of numeric values in the same order as the training features.

Typical feature order for this dataset is:

1. MDVP:Fo(Hz)
2. MDVP:Fhi(Hz)
3. MDVP:Flo(Hz)
4. MDVP:Jitter(%)
5. MDVP:Jitter(Abs)
6. MDVP:RAP
7. MDVP:PPQ
8. Jitter:DDP
9. MDVP:Shimmer
10. MDVP:Shimmer(dB)
11. Shimmer:APQ3
12. Shimmer:APQ5
13. MDVP:APQ
14. Shimmer:DDA
15. NHR
16. HNR
17. RPDE
18. DFA
19. spread1
20. spread2
21. D2
22. PPE

Example input string:

```text
119.992,157.302,74.997,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.426,0.02182,0.03130,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654
```

## Local Setup

### Prerequisites

- Python 3.10+
- `pip`

### Install

```bash
git clone https://github.com/Hena685/parkinsons-mlops.git
cd parkinsons-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## CI/CD Workflows

### CI (`.github/workflows/ci.yml`)

On push/PR to `main`:
- Runs formatting check (`black --check --diff .`)
- Runs lint (`flake8`)
- Runs tests (`pytest`)

Current workflow uses `|| true` in these commands, so failures do not break the pipeline yet.

### Deploy (`.github/workflows/deploy.yml`)

On changes to app/deployment-relevant files:
- Authenticates with Hugging Face via secrets
- Clones target Space: `HF_USERNAME/parkinsons-detection`
- Copies `app.py`, `requirements.txt`, `model.pkl`, and `README.md`
- Commits and pushes updates to Space `main`

Required repository secrets:
- `HF_USERNAME`
- `HF_TOKEN`

## Current Limitations

- `app.py` accepts free-form comma-separated text without strict schema validation
- `src/` inference path requires a missing `config/config.yaml`
- No dataset versioning or experiment tracking yet
- No committed automated tests in this repository currently

## Suggested Next Improvements

1. Add `config/config.yaml` and unify app + `src/` inference flow
2. Add robust input validation and clearer UI guidance
3. Add unit tests for preprocessing and inference
4. Remove `|| true` from CI to enforce quality gates
5. Add model metrics/version metadata in repository

## Author

- GitHub: [@hemangsengar](https://github.com/hemangsengar)

## License

No license file is currently included. Add a `LICENSE` file to define usage terms.