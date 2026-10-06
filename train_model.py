import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/customer_churn.csv")

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 70)

print("\nDataset loaded successfully!")
print(f"Original shape: {df.shape}")


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where TotalCharges could not be converted
df.dropna(subset=["TotalCharges"], inplace=True)

# Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert target variable
# No = 0
# Yes = 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# ============================================================
# 7. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_pipeline.fit(X_train, y_train)

logistic_predictions = logistic_pipeline.predict(X_test)


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest_pipeline.fit(X_train, y_train)

random_forest_predictions = random_forest_pipeline.predict(X_test)


# ============================================================
# 9. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 10. EVALUATE BOTH MODELS
# ============================================================

logistic_results = evaluate_model(
    "LOGISTIC REGRESSION RESULTS",
    y_test,
    logistic_predictions
)

random_forest_results = evaluate_model(
    "RANDOM FOREST RESULTS",
    y_test,
    random_forest_predictions
)


# ============================================================
# 11. COMPARE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"\n{'Metric':<15}"
    f"{'Logistic Regression':<22}"
    f"{'Random Forest':<15}"
)

print("-" * 52)

print(
    f"{'Accuracy':<15}"
    f"{logistic_results['accuracy']:<22.4f}"
    f"{random_forest_results['accuracy']:.4f}"
)

print(
    f"{'Precision':<15}"
    f"{logistic_results['precision']:<22.4f}"
    f"{random_forest_results['precision']:.4f}"
)

print(
    f"{'Recall':<15}"
    f"{logistic_results['recall']:<22.4f}"
    f"{random_forest_results['recall']:.4f}"
)

print(
    f"{'F1 Score':<15}"
    f"{logistic_results['f1']:<22.4f}"
    f"{random_forest_results['f1']:.4f}"
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

if random_forest_results["f1"] > logistic_results["f1"]:
    best_model = random_forest_pipeline
    best_model_name = "Random Forest"
    best_f1 = random_forest_results["f1"]

else:
    best_model = logistic_pipeline
    best_model_name = "Logistic Regression"
    best_f1 = logistic_results["f1"]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"\nSelected Model: {best_model_name}")
print(f"F1 Score: {best_f1:.4f}")


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

model_path = "model/churn_model.pkl"

joblib.dump(best_model, model_path)

print(f"\nBest model saved successfully!")
print(f"Location: {model_path}")


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)