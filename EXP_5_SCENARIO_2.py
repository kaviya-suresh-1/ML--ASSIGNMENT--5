import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
zip_path = r"C:\Users\Kaviya\Desktop\Machine_learning_lab\train_u6lujuX_CVtuZ9i (1) (2).csv"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Files inside ZIP:", zip_ref.namelist())
    csv_name = zip_ref.namelist()[0]
    df = pd.read_csv(zip_ref.open(csv_name))
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values Before:")
print(df.isnull().sum())

df.fillna(df.mode().iloc[0], inplace=True)

print("\nMissing Values After:")
print(df.isnull().sum())
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
X = df[['ApplicantIncome','LoanAmount','Credit_History','Education','Property_Area']]
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
importances = dt.feature_importances_
features = X.columns
plt.figure()
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()
plt.figure(figsize=(14,8))
plot_tree(dt, feature_names=X.columns, filled=True)
plt.title("Decision Tree Structure")
plt.show()          
depths = [2,4,6,8,10]
train_acc = []
test_acc = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure()
plt.plot(depths, train_acc, label="Train Accuracy")
plt.plot(depths, test_acc, label="Test Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Overfitting Analysis")
plt.show()