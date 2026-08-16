import pandas as pd


# ---------------------------------------------------------
# 1. Define validation results collected from our experiments
# ---------------------------------------------------------

# These are the validation results obtained from the three
# models we trained and tracked during Week 2.
model_results = [
    {
        "model": "Logistic Regression",
        "feature_method": "TF-IDF",
        "validation_accuracy": 0.9963,
        "validation_macro_f1": 0.9965,
        "misclassified_examples": 3,
        "training_cost": "Very Low",
        "inference_cost": "Very Low",
        "native_probability_support": "Yes",
        "deployment_complexity": "Low",
    },
    {
        "model": "Linear SVM",
        "feature_method": "TF-IDF",
        "validation_accuracy": 0.9976,
        "validation_macro_f1": 0.9978,
        "misclassified_examples": 2,
        "training_cost": "Very Low",
        "inference_cost": "Very Low",
        "native_probability_support": "No",
        "deployment_complexity": "Low",
    },
    {
        "model": "DistilBERT",
        "feature_method": "Transformer",
        "validation_accuracy": 0.9976,
        "validation_macro_f1": 0.9976,
        "misclassified_examples": 2,
        "training_cost": "High",
        "inference_cost": "Higher",
        "native_probability_support": "Yes",
        "deployment_complexity": "Higher",
    },
]


# ---------------------------------------------------------
# 2. Convert comparison data into a dataframe
# ---------------------------------------------------------

comparison_df = pd.DataFrame(model_results)


# ---------------------------------------------------------
# 3. Sort primarily by Macro F1
# ---------------------------------------------------------

# Macro F1 is our primary comparison metric because it gives
# equal importance to all 27 intent classes.
comparison_df = comparison_df.sort_values(
    by="validation_macro_f1",
    ascending=False,
)


# ---------------------------------------------------------
# 4. Display the comparison
# ---------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

print("\n" + "=" * 120)
print("MODEL COMPARISON")
print("=" * 120)

print(
    comparison_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 5. Print the current best validation model
# ---------------------------------------------------------

best_model = comparison_df.iloc[0]

print("\n" + "=" * 120)
print("CURRENT BEST VALIDATION MODEL")
print("=" * 120)

print(
    f"Model       : {best_model['model']}"
)

print(
    f"Accuracy    : {best_model['validation_accuracy']:.4f}"
)

print(
    f"Macro F1    : {best_model['validation_macro_f1']:.4f}"
)

print(
    f"Errors      : {best_model['misclassified_examples']}"
)