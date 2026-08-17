````
# Customer Intent Classification

An end-to-end Machine Learning and MLOps project for classifying customer-service utterances into predefined intent categories.

The project covers the complete lifecycle of a text-classification model:

- Data validation and exploratory data analysis
- Text preprocessing
- TF-IDF feature engineering
- Classical ML model training
- Transformer fine-tuning
- MLflow experiment tracking
- Model comparison and selection
- Final test-set evaluation
- Model serialization and inference
- FastAPI deployment
- Docker containerization
- Automated testing
- Prediction logging
- Model monitoring and drift detection
- Retraining decision logic

---

# 1. Project Objective

The objective of this project is to classify customer-service messages into the correct intent so that downstream systems can automatically route or process customer requests.

Example:

```text
Customer:
"Where is my order?"

        ↓

Intent Classifier

        ↓

track_order
```

Another example:

```text
Customer:
"I forgot my password"

        ↓

Intent Classifier

        ↓

recover_password
```

The project also addresses an important production ML problem:

> What happens when customer language changes after the model has been deployed?

For this reason, the project includes a monitoring framework for detecting language/input drift, prediction-distribution drift, and model-performance degradation.

---

# 2. Dataset

The project uses predefined training, validation, and testing datasets.

Files:

```text
data/raw/
├── Customer_Service_Training_Dataset.csv
├── Customer_Service_Validation_Dataset.csv
└── Customer_Service_Testing_Dataset.csv
```

Dataset sizes:

| Dataset | Rows |
|---|---:|
| Training | 6,539 |
| Validation | 818 |
| Testing | 818 |

Each dataset contains:

```text
utterance
intent
category
tags
```

The classification target is:

```text
intent
```

There are **27 unique intent classes**.

Examples include:

```text
cancel_order
change_order
change_shipping_address
check_cancellation_fee
check_invoice
check_payment_methods
check_refund_policy
complaint
contact_customer_service
contact_human_agent
create_account
delete_account
delivery_options
delivery_period
edit_account
get_invoice
get_refund
newsletter_subscription
payment_issue
place_order
recover_password
registration_problems
review
set_up_shipping_address
switch_account
track_order
track_refund
```

---

# 3. Project Structure

```text
intent-classifier/
│
├── data/
│   ├── raw/
│   │   ├── Customer_Service_Training_Dataset.csv
│   │   ├── Customer_Service_Validation_Dataset.csv
│   │   └── Customer_Service_Testing_Dataset.csv
│   │
│   ├── drift/
│   │   └── Customer_Service_Drift_Dataset.csv
│   │
│   └── monitoring/
│       └── prediction_log.csv
│
├── models/
│   └── linear_svm_pipeline.joblib
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   └── intent_classifier/
│       │
│       ├── __init__.py
│       │
│       ├── api/
│       │   └── main.py
│       │
│       ├── data/
│       │   ├── inspect_data.py
│       │   └── validate_data.py
│       │
│       ├── features/
│       │   ├── text_preprocessing.py
│       │   └── tfidf_features.py
│       │
│       ├── inference/
│       │   └── predict.py
│       │
│       ├── models/
│       │   ├── train_baseline.py
│       │   ├── train_linear_svm.py
│       │   ├── train_distilbert.py
│       │   ├── analyze_distilbert.py
│       │   ├── compare_models.py
│       │   └── evaluate_final_model.py
│       │
│       └── monitoring/
│           ├── __init__.py
│           ├── prediction_logger.py
│           ├── evaluate_drift.py
│           ├── detect_input_drift.py
│           ├── detect_prediction_drift.py
│           └── generate_monitoring_report.py
│
├── tests/
│   ├── ...
│   └── test_monitoring.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated artifacts such as saved models, runtime monitoring logs, and local MLflow data may be excluded from Git.

---

# 4. Environment and Dependency Management

The project uses **uv** for Python dependency and virtual-environment management.

Install uv if it is not already available.

After cloning the repository:

```bash
git clone <repository-url>
cd intent-classifier
```

Install and synchronize dependencies:

```bash
uv sync
```

`uv` reads:

```text
pyproject.toml
uv.lock
```

and creates/reuses the project virtual environment.

Python scripts should normally be executed using:

```bash
uv run python <path-to-script>
```

For example:

```bash
uv run python src/intent_classifier/models/train_linear_svm.py
```

---

# 5. Data Validation and EDA

Initial analysis verifies:

- Dataset dimensions
- Missing values
- Duplicate rows
- Empty utterances
- Intent distribution
- Category distribution
- Text length
- Word count
- Punctuation
- Numbers
- Upper/lowercase patterns
- Label consistency across train/validation/test splits

Important observations:

- Training rows: 6,539
- Validation rows: 818
- Testing rows: 818
- Unique intents: 27
- Missing values: 0
- Duplicate rows: 0
- Empty utterances: 0
- Intent sets are consistent across train, validation, and test data

The data was therefore already relatively clean.

---

# 6. Text Preprocessing

A lightweight text-cleaning strategy is used.

The objective is to normalize obvious noise while preserving words that may be important for intent classification.

Heavy preprocessing such as aggressive stemming, lemmatization, or stop-word removal was intentionally avoided.

This is particularly useful because phrases such as:

```text
cancel order
track refund
change address
payment issue
```

contain combinations of words that are highly informative for intent classification.

---

# 7. TF-IDF Feature Engineering

Classical models use TF-IDF to convert customer utterances into numeric feature vectors.

Configuration:

```python
TfidfVectorizer(
    preprocessor=clean_text,
    ngram_range=(1, 2),
    max_features=5000,
)
```

Both unigrams and bigrams are used.

For example:

```text
"cancel my order"
```

can generate features such as:

```text
cancel
my
order
cancel my
my order
```

This allows the model to learn both individual words and short phrases.

---

# 8. Models Evaluated

Three main models were evaluated.

## Logistic Regression

The first baseline was:

```text
TF-IDF
   ↓
Logistic Regression
```

Validation results:

```text
Accuracy : 0.9963
Macro F1 : 0.9965
```

Only 3 of 818 validation examples were misclassified.

Logistic Regression provided a strong, fast, interpretable baseline.

---

## Linear SVM

The second classical model was:

```text
TF-IDF
   ↓
Linear SVM
```

The TF-IDF configuration was intentionally kept the same as the Logistic Regression experiment so that the classifier was the main variable being changed.

Validation results:

```text
Accuracy : 0.9976
Macro F1 : 0.9978
```

Misclassified examples:

```text
2 / 818
```

Linear SVM produced the best validation Macro F1.

---

## DistilBERT

A transformer-based model was also fine-tuned.

Architecture:

```text
Customer utterance
       ↓
Tokenizer
       ↓
DistilBERT
       ↓
Classification head
       ↓
27 intent probabilities
```

DistilBERT was selected as the transformer experiment because it provides contextual language understanding while being smaller and faster than full BERT.

It is useful for comparing a contextual deep-learning model against classical TF-IDF approaches.

After fine-tuning:

```text
Validation Accuracy : 0.9976
Validation Macro F1 : 0.9976
```

Misclassified examples:

```text
2 / 818
```

---

# 9. MLflow Experiment Tracking

MLflow is used to track model experiments.

MLflow allows experiments to record information such as:

- Model type
- Parameters
- Accuracy
- Macro F1
- Training configuration
- Model artifacts
- Comparison between runs

The project experiment is:

```text
customer-intent-classification
```

Start the MLflow UI with:

```bash
uv run mlflow ui
```

Then open the local MLflow address shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

MLflow makes it possible to compare model experiments without relying only on terminal output.

---

# 10. Model Comparison

The three models produced:

| Model | Feature Method | Validation Accuracy | Validation Macro F1 | Errors |
|---|---|---:|---:|---:|
| Linear SVM | TF-IDF | 0.9976 | **0.9978** | 2 |
| DistilBERT | Transformer | 0.9976 | 0.9976 | 2 |
| Logistic Regression | TF-IDF | 0.9963 | 0.9965 | 3 |

Linear SVM was selected as the final model.

Although DistilBERT performed almost identically, Linear SVM offered:

- Highest validation Macro F1
- Very low training cost
- Very low inference cost
- Simpler deployment
- Smaller operational footprint
- Faster experimentation

Therefore, the additional transformer complexity was not justified by the validation results for this dataset.

---

# 11. Final Test Evaluation

The untouched testing dataset was used only after model selection.

Final Linear SVM results:

```text
Test Accuracy : 0.9976
Test Macro F1 : 0.9975
```

Only **2 of 818 test examples** were misclassified.

Examples included confusion between closely related intents such as:

```text
delete_account
vs
cancel_order
```

and:

```text
track_order
vs
delivery_period
```

The test result confirmed that the selected model generalized very well to the predefined test distribution.

---

# 12. Saved Production Model

The selected sklearn pipeline is serialized as:

```text
models/linear_svm_pipeline.joblib
```

The complete pipeline is saved rather than only the SVM classifier.

This is important because inference requires exactly the same:

```text
clean_text
   ↓
TF-IDF transformation
   ↓
Linear SVM
```

used during training.

The model artifact is generated rather than treated as source code and can therefore be excluded from Git.

---

# 13. Inference Layer

The inference module loads the saved pipeline and exposes reusable prediction functionality.

Conceptually:

```text
Raw customer text
       ↓
predict_intent()
       ↓
Saved sklearn Pipeline
       ↓
Predicted intent
```

This separates model inference from the API layer and makes the prediction logic reusable.

---

# 14. FastAPI Service

The trained classifier is exposed through FastAPI.

Run locally:

```bash
uv run uvicorn intent_classifier.api.main:app --app-dir src --reload
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Health endpoint

```http
GET /health
```

Used to confirm that the API is running.

## Prediction endpoint

```http
POST /predict
```

Example request:

```json
{
  "text": "where is my order?"
}
```

Example response:

```json
{
  "intent": "track_order"
}
```

Pydantic validation is used to validate incoming requests.

---

# 15. Docker

Docker packages the API, application code, Python environment, dependencies, and required model artifact into a reproducible container image.

Docker Desktop must be running when using Docker locally on Windows.

Build:

```bash
docker build -t intent-classifier .
```

Run:

```bash
docker run -p 8000:8000 intent-classifier
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Model artifact and Docker

The saved `.joblib` model is not committed to Git.

Before building the Docker image, generate the model locally using the final model training/evaluation workflow.

The Docker build context can then include the locally generated model artifact even though Git does not track it, provided `.dockerignore` does not exclude the required artifact.

Therefore:

```text
Git repository
     ≠
Docker build context
```

A file can be ignored by Git while still being copied into a Docker image.

---

# 16. Automated Testing

The project includes automated tests for the API and monitoring logic.

Run all tests:

```bash
uv run pytest -v
```

The API tests validate behaviour such as:

- Health endpoint
- Valid prediction request
- Empty/invalid input handling
- Response schema
- Expected prediction behaviour

The monitoring tests validate the operational decision logic.

---

# 17. Prediction Logging

Production monitoring begins by recording successful predictions.

The `/predict` flow is:

```text
POST /predict
      ↓
Pydantic validation
      ↓
Linear SVM
   ┌──┴─────────────┐
   ↓                ↓
Response      Prediction Log
```

Predictions are written to:

```text
data/monitoring/prediction_log.csv
```

Each record contains:

```text
timestamp
text
predicted_intent
```

Example:

```text
2026-08-17T18:55:30+00:00,where is my order?,track_order
```

The monitoring log is runtime data and should not normally be committed to Git.

In a production environment, this would typically be replaced with durable logging or database/event-stream storage.

---

# 18. Model Drift Monitoring

A production model can deteriorate even when its original test performance was excellent.

The monitoring framework therefore evaluates three signals:

```text
                 Production Traffic
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Input Drift    Prediction     Performance
                      Drift           Drift
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Decision Logic
                         ▼
             NO_ACTION / REVIEW / RETRAIN
```

---

# 19. Controlled Drift Dataset

A controlled drift dataset was created:

```text
data/drift/Customer_Service_Drift_Dataset.csv
```

It contains:

```text
27 intents
×
5 examples per intent
=
135 examples
```

The dataset simulates changed customer language using:

- Slang
- Abbreviations
- Informal grammar
- Shortened chat-style messages
- Paraphrasing
- Vocabulary changes
- Typos

The intent definitions remain unchanged.

Therefore, the experiment primarily simulates **input/language drift**, rather than proving pure concept drift.

---

# 20. Performance Under Changed Language

The final Linear SVM was evaluated against the controlled drift dataset.

Results:

| Metric | Original Test | Drift Dataset |
|---|---:|---:|
| Accuracy | 0.9976 | **0.6074** |
| Macro F1 | 0.9975 | **0.5929** |
| Misclassified | 2 / 818 | **53 / 135** |

Accuracy dropped by:

```text
0.3902
```

Macro F1 dropped by:

```text
0.4046
```

This demonstrates that extremely high test performance does not guarantee robustness when real customer language changes.

---

# 21. Label-Free Input Drift Detection

Production traffic usually does not immediately contain the correct intent label.

Therefore, input drift is detected without using `y_true`.

The fitted production TF-IDF vectorizer is used to transform:

```text
training text
validation text
new production text
```

For each validation utterance, cosine similarity is calculated against the training corpus and the closest training example is retained.

The **5th percentile** of normal validation similarity becomes the lower reference boundary.

Observed threshold:

```text
0.5340
```

Normal validation traffic:

```text
Mean max similarity : 0.8161
Below threshold     : 5.01%
```

Drift traffic:

```text
Mean max similarity : 0.5570
Below threshold     : 48.89%
```

Result:

```text
Input Drift Status : HIGH
```

This signal does **not require true production labels**.

---

# 22. Prediction Distribution Drift

The second label-free monitoring signal examines whether the model starts predicting intents at substantially different rates.

Total Variation Distance (TVD) is used.

Conceptually:

```text
TVD = 0
```

means identical prediction distributions.

Larger values indicate increasing differences.

Observed result:

```text
Total Variation Distance : 0.1872
Prediction Drift Status  : MODERATE
```

Some notable changes were:

| Intent | Reference | Drift Traffic |
|---|---:|---:|
| edit_account | 3.18% | 9.63% |
| switch_account | 3.18% | 7.41% |
| delivery_options | 4.16% | 0.00% |

Prediction drift is treated as a warning rather than proof that the model is wrong.

---

# 23. Performance Monitoring

When reviewed ground-truth labels become available, model quality can be measured directly.

The drift experiment produced:

```text
Baseline Test Accuracy : 0.9976
Current Accuracy       : 0.6074

Baseline Test Macro F1 : 0.9975
Current Macro F1       : 0.5929
```

Performance status:

```text
HIGH
```

This provides strong evidence that the changed language is affecting model quality.

---

# 24. Monitoring Decision

The final monitoring report combines the independent signals.

Observed result:

```text
Input Drift       : HIGH
Prediction Drift  : MODERATE
Performance Drift : HIGH

Recommended Action: RETRAIN
```

The system intentionally does **not** recommend retraining because of a single warning.

The decision policy is approximately:

```text
Stable signals
        ↓
NO_ACTION


Concerning signal without enough evidence
        ↓
REVIEW


HIGH performance degradation
+
HIGH input or prediction drift
        ↓
RETRAIN
```

Retraining means creating and evaluating a new candidate model. It does not mean automatically replacing the deployed model.

---

# 25. Monitoring Tests

Six unit tests validate the monitoring decision logic.

Scenarios include:

```text
LOW + LOW + LOW
→ NO_ACTION
```

```text
MODERATE input drift
→ REVIEW
```

```text
MODERATE prediction drift
→ REVIEW
```

```text
HIGH performance degradation alone
→ REVIEW
```

```text
HIGH input drift + HIGH performance degradation
→ RETRAIN
```

```text
HIGH prediction drift + HIGH performance degradation
→ RETRAIN
```

All six monitoring tests pass.

---

# 26. Production Drift Workflow

A realistic production workflow would be:

```text
Customer requests
       ↓
Prediction API
       ↓
Prediction logging
       ↓
Recent traffic window
       ↓
Input drift detection
+
Prediction distribution monitoring
       ↓
Warning if behaviour changes
       ↓
Collect / review labelled examples
       ↓
Measure Accuracy / Macro F1
       ↓
NO_ACTION / REVIEW / RETRAIN
```

If retraining is justified:

```text
Historical training data
        +
Reviewed recent production examples
        ↓
Retrain candidate models
        ↓
MLflow experiment comparison
        ↓
Historical evaluation
        +
Recent-language evaluation
        ↓
Acceptance criteria
        ↓
Deploy new version
        ↓
Continue monitoring
```

A model should not be retrained blindly on only the latest production batch because doing so may reduce performance on historical language.

---

# 27. Common Commands

## Install dependencies

```bash
uv sync
```

## Run a Python module/script

```bash
uv run python <path-to-file>
```

Example:

```bash
uv run python src/intent_classifier/models/train_linear_svm.py
```

## Start MLflow

```bash
uv run mlflow ui
```

## Start FastAPI locally

```bash
uv run uvicorn intent_classifier.api.main:app --app-dir src --reload
```

## Run tests

```bash
uv run pytest -v
```

## Run drift evaluation

```bash
uv run python src/intent_classifier/monitoring/evaluate_drift.py
```

## Run input drift detection

```bash
uv run python src/intent_classifier/monitoring/detect_input_drift.py
```

## Run prediction drift detection

```bash
uv run python src/intent_classifier/monitoring/detect_prediction_drift.py
```

## Generate complete monitoring report

```bash
uv run python src/intent_classifier/monitoring/generate_monitoring_report.py
```

## Build Docker image

```bash
docker build -t intent-classifier .
```

## Run Docker container

```bash
docker run -p 8000:8000 intent-classifier
```

---

# 28. Setup for a New Developer

A teammate can reproduce the project using the following workflow.

### 1. Install uv

Install the `uv` package manager.

### 2. Clone the repository

```bash
git clone <repository-url>
cd intent-classifier
```

### 3. Place required datasets

Ensure the raw CSV files are available under:

```text
data/raw/
```

### 4. Synchronize dependencies

```bash
uv sync
```

### 5. Run tests

```bash
uv run pytest -v
```

### 6. Run MLflow if experiment inspection is required

```bash
uv run mlflow ui
```

### 7. Generate the final model artifact if it is not present

Run the project's final model training/evaluation workflow to generate:

```text
models/linear_svm_pipeline.joblib
```

### 8. Run the API

```bash
uv run uvicorn intent_classifier.api.main:app --app-dir src --reload
```

### 9. Test using Swagger

Open:

```text
http://127.0.0.1:8000/docs
```

### 10. Run Docker if required

Ensure Docker Desktop is running.

```bash
docker build -t intent-classifier .
docker run -p 8000:8000 intent-classifier
```

---

# 29. Generated and Ignored Artifacts

Some project files are intentionally generated locally rather than stored in Git.

Examples may include:

```text
.venv/
mlruns/
mlflow.db
models/*.joblib
data/monitoring/
__pycache__/
.pytest_cache/
```

Reasons include:

- Virtual environments are machine-specific.
- MLflow runtime artifacts can become large.
- Model binaries are generated artifacts.
- Prediction logs are runtime data.
- Python caches do not belong in source control.

The exact ignore rules are defined in `.gitignore`.

---

# 30. Current Final Model

```text
Model:
Linear SVM

Features:
TF-IDF unigrams + bigrams

Validation Accuracy:
0.9976

Validation Macro F1:
0.9978

Final Test Accuracy:
0.9976

Final Test Macro F1:
0.9975
```

The model was selected over Logistic Regression and DistilBERT because it produced the best validation Macro F1 while remaining computationally inexpensive and straightforward to deploy.

---

# 31. Limitations

The current project has several important limitations.

The predefined dataset appears relatively easy for the evaluated models, as shown by near-perfect validation and test performance.

The controlled drift dataset is manually constructed and is intended to demonstrate monitoring behaviour rather than represent genuine production traffic.

The monitoring thresholds are project-level operational thresholds and have not been calibrated using long-term historical production data.

Prediction logging currently uses a CSV file. A production system would use more durable and scalable storage.

Ground-truth production labels are assumed to become available through review or another feedback process.

The project demonstrates simulated language/input drift, not confirmed pure concept drift.

---

# 32. Future Improvements

Possible future enhancements include:

- Collect real production utterances
- Introduce human feedback and reviewed labels
- Calibrate drift thresholds from stable production history
- Use rolling monitoring windows
- Add model-version information to prediction logs
- Store monitoring data in a database
- Build a monitoring dashboard
- Add alerting
- Evaluate transformer robustness under real drift
- Add confidence/calibration analysis
- Retrain using reviewed recent examples
- Compare retrained candidates through MLflow
- Introduce a model registry
- Add CI/CD
- Deploy to a cloud environment
- Add production observability for latency and API errors

---

# 33. End-to-End Project Summary

The project implements the following ML lifecycle:

```text
Raw customer-service data
        ↓
Data validation + EDA
        ↓
Text preprocessing
        ↓
TF-IDF feature engineering
        ↓
Logistic Regression
Linear SVM
DistilBERT
        ↓
MLflow experiment tracking
        ↓
Model comparison
        ↓
Linear SVM selected
        ↓
Untouched test evaluation
        ↓
Save production pipeline
        ↓
Inference layer
        ↓
FastAPI
        ↓
Docker
        ↓
Automated tests
        ↓
Prediction logging
        ↓
Drift monitoring
        ↓
Input + Prediction + Performance signals
        ↓
Operational decision
        ↓
NO_ACTION / REVIEW / RETRAIN
```

The final result is therefore not only a trained text-classification model, but an end-to-end example of how a machine-learning model can be **trained, evaluated, tracked, deployed, tested, monitored, and prepared for future retraining**.
````
---

# 34. Dataset Versioning with DVC

The project requirement includes dataset versioning. **DVC (Data Version Control)** was introduced to track dataset versions separately from source code while keeping lightweight DVC metadata under Git version control.

The intended workflow was:

```text
Raw CSV datasets
      ↓
DVC tracking
      ↓
.dvc metadata committed to Git
      ↓
Dataset contents stored outside normal Git history
```

A Google Drive DVC remote was also attempted so that team members could retrieve the same dataset version. However, authentication to the remote was blocked by institutional OAuth restrictions in the development environment.

As a result, the project retains the DVC/versioning approach and metadata, but collaborators currently obtain the source CSV files separately and place them under:

```text
data/raw/
```

This limitation should be distinguished from the generated drift dataset under `data/drift/`, which is part of the project implementation and can be version-controlled with the source code.

---

# 35. Alignment with the Assignment Drift Requirement

The assignment asks the project to simulate concept drift as language and topics evolve. The implemented experiment changes customer language using slang, abbreviations, paraphrases, typos, and informal phrasing while keeping the existing 27 intent definitions unchanged.

Technically, this primarily represents **input/language drift**, meaning the distribution of incoming text `P(X)` changes. Pure **concept drift** would mean that the relationship between an utterance and its correct label `P(Y|X)` changes, for example because the business redefines, merges, or introduces support intents.

The project therefore uses the assignment's new-language drift scenario to demonstrate how changed production inputs can lead to performance degradation and how a production monitoring framework can detect the change, request review, and recommend retraining when sufficient evidence is available.

---

# 36. References and Acknowledgements

## Dataset

This project uses a customer-service intent-classification dataset obtained from Kaggle for academic work. The dataset provides predefined training, validation, and testing splits containing customer utterances and intent labels.

## Dataset Reference

- **Dataset:** customer-support-intent-dataset
- **Author/Owner:** Tara Prasad Pandey
- **Source:** Kaggle – Customer Support Intent Dataset
- **Accessed:** 18 August 2026
```

## Open-Source Libraries and Tools

The implementation uses open-source software including:

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- MLflow
- FastAPI
- Uvicorn
- Pydantic
- joblib
- pytest
- Docker
- DVC
- uv

These libraries provide the underlying numerical computing, NLP, machine-learning, experiment-tracking, serving, testing, containerization, dependency-management, and data-versioning capabilities used by the project. The project-specific preprocessing, training workflows, model comparison, API integration, drift simulation, monitoring logic, and retraining decision framework were implemented for this academic project.

---

# 37. Requirement Coverage

| Assignment Requirement | Project Implementation |
|---|---|
| Week 1 - ingest raw text | Train, validation, and test customer-service CSV datasets |
| Clean/tokenize | Lightweight text normalization; transformer tokenization for DistilBERT |
| TF-IDF/embeddings | TF-IDF unigram + bigram pipeline and DistilBERT contextual representations |
| Version dataset | DVC introduced; remote sharing limited by institutional OAuth restrictions |
| Week 2 - classical ML | Logistic Regression and Linear SVM |
| Fine-tuned transformer | DistilBERT fine-tuned for 27-class intent classification |
| Track experiments | MLflow experiment `customer-intent-classification` |
| Week 3 - package model | Complete Linear SVM sklearn pipeline serialized with joblib |
| REST API | FastAPI `/health` and `/predict` endpoints |
| Handle malformed/empty input | Pydantic/API validation and automated API tests |
| Package application | Dockerfile and `.dockerignore` |
| Week 4 - log predictions | Runtime `prediction_log.csv` |
| Simulate changing language/topics | 135-example controlled slang/paraphrase drift dataset |
| Monitor performance | Accuracy and Macro F1 compared with final test baseline |
| Detect drift without labels | TF-IDF/cosine-similarity input monitoring and TVD prediction monitoring |
| Design retraining triggers | `NO_ACTION`, `REVIEW`, and `RETRAIN` policy with unit tests |
| Version-control code incrementally | Weekly implementation committed through Git |
| Justify design decisions | Week-wise documentation plus this README |

