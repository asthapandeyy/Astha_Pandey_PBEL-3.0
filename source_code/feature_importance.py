import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# Load dataset

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


y = data["Result"]


# Train model

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X, y)


# Feature importance

importance = model.feature_importances_


features = [
"Hours Studied",
"Attendance",
"Assignments",
"Previous Score"
]


importance_data = pd.DataFrame(
{
"Feature": features,
"Importance": importance
}
)


importance_data = importance_data.sort_values(
by="Importance",
ascending=False
)


print(importance_data)


# Plot graph

plt.figure(figsize=(8,5))

plt.bar(
importance_data["Feature"],
importance_data["Importance"]
)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.title(
"Factors Affecting Student Performance"
)

plt.xticks(
rotation=45
)

plt.tight_layout()


plt.savefig(
"feature_importance.png"
)


print(
"Feature importance graph saved!"
)