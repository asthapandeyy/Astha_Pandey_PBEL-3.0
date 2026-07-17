import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("../dataset/student_data_large.csv")


X = data[
[
"Hours_Studied",
"Attendance",
"Assignments",
"Previous_Score"
]
]


y = data["Result"]
# Split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
# Train

model = DecisionTreeClassifier(
random_state=42
)

model.fit(
X_train,
y_train
)
# Prediction

prediction = model.predict(
X_test
)


accuracy = accuracy_score(
y_test,
prediction
)


print(
"Accuracy:",
accuracy*100,
"%"
)
# Confusion Matrix

cm = confusion_matrix(
y_test,
prediction
)

plt.figure(figsize=(5,4))

sns.heatmap(
cm,
annot=True,
fmt="d"
)

plt.xlabel(
"Predicted"
)

plt.ylabel(
"Actual"
)

plt.title(
"Confusion Matrix"
)

plt.savefig(
"confusion_matrix.png"
)

print(
"Confusion matrix saved!"
)