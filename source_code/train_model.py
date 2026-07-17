import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Load larger dataset
data = pd.read_csv("../dataset/student_data_large.csv")


# Features
X = data[
    [
        "Hours_Studied",
        "Attendance",
        "Assignments",
        "Previous_Score"
    ]
]


# Target
y = data["Result"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# Testing
y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("Model Accuracy:", accuracy * 100, "%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# Save model
joblib.dump(
    model,
    "model.pkl"
)


print("\nModel saved successfully!")