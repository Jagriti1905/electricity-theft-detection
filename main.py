import pandas as pd

df = pd.read_csv("data/data set.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 10 Columns:")
print(df.columns[:10])

print("\nLast 10 Columns:")
print(df.columns[-10:])

print("\nFirst 5 Rows:")
print(df.head())

print("\nFLAG Distribution:")
print(df["FLAG"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nTop 10 Columns with Missing Values:")

missing = df.isnull().sum()
print(missing.sort_values(ascending=False).head(10))

# Remove columns with more than 50% missing values
threshold = len(df) * 0.5

df_clean = df.dropna(axis=1, thresh=threshold)

print("\nOriginal Shape:")
print(df.shape)

print("\nShape after removing highly missing columns:")
print(df_clean.shape)

print("\nRemaining Missing Values:")
print(df_clean.isnull().sum().sum())

print("\nData Types:")
print(df_clean.dtypes.value_counts())

# Fill missing values in consumption columns with their median
consumption_cols = df_clean.select_dtypes(include="float64").columns

df_clean[consumption_cols] = df_clean[consumption_cols].fillna(
    df_clean[consumption_cols].median()
)

print("\nMissing Values After Filling:")
print(df_clean.isnull().sum().sum())

# Save cleaned dataset
df_clean.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Separate features and target

X = df_clean.drop(columns=["FLAG", "CONS_NO"])
y = df_clean["FLAG"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Target:")
print(y_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")
# Make predictions on test data
y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test.iloc[:10].values)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

# Train
tree_model.fit(X_train, y_train)

# Predict
tree_pred = tree_model.predict(X_test)

# Evaluate
tree_accuracy = accuracy_score(y_test, tree_pred)
tree_precision = precision_score(y_test, tree_pred)
tree_recall = recall_score(y_test, tree_pred)
tree_f1 = f1_score(y_test, tree_pred)

print("\n--- Decision Tree Evaluation ---")
print("Accuracy:", tree_accuracy)
print("Precision:", tree_precision)
print("Recall:", tree_recall)
print("F1 Score:", tree_f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train
forest_model.fit(X_train, y_train)

# Predict
forest_pred = forest_model.predict(X_test)

# Evaluate
forest_accuracy = accuracy_score(y_test, forest_pred)
forest_precision = precision_score(y_test, forest_pred)
forest_recall = recall_score(y_test, forest_pred)
forest_f1 = f1_score(y_test, forest_pred)

print("\n--- Random Forest Evaluation ---")
print("Accuracy:", forest_accuracy)
print("Precision:", forest_precision)
print("Recall:", forest_recall)
print("F1 Score:", forest_f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, forest_pred))

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy, tree_accuracy, forest_accuracy],
    "Precision": [precision, tree_precision, forest_precision],
    "Recall": [recall, tree_recall, forest_recall],
    "F1 Score": [f1, tree_f1, forest_f1]
})

print("\n--- Model Comparison ---")
print(results.round(4).to_string(index=False))

import joblib

# Save the trained Random Forest model
joblib.dump(forest_model, "electricity_theft_model.pkl")

print("\nModel saved successfully!")