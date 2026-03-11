import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.colors import ListedColormap


def run_knn_scenario():

    print("--- SCENARIO: K-NEAREST NEIGHBORS (KNN) ---")

    data_path = "breast-cancer (1).csv"
    df = pd.read_csv(r"C:\Users\Kaviya\Desktop\Machine_learning_lab\breast-cancer (1).csv")

    print("\nDataset Info:")
    df.info()

    print("\nFirst 5 rows:")
    print(df.head())

    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    X = df[features]
    y = df['diagnosis']

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    k_values = range(1, 21)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig("knn_accuracy_vs_k.png")
    plt.show()

    best_k = k_values[np.argmax(accuracies)]
    print(f"\nBest K identified: {best_k}")

    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred_best = knn_best.predict(X_test)

    print("\nFinal Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_best):.4f}")

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix (K={best_k})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("knn_confusion_matrix.png")
    plt.show()

    misclassified = np.where(y_test != y_pred_best)[0]
    print(f"\nNumber of misclassified cases: {len(misclassified)}")
    if len(misclassified) > 0:
        print("Sample misclassified indices:", misclassified[:10])

    X_vis = X_scaled[:, :2]

    knn_vis = KNeighborsClassifier(n_neighbors=best_k)
    knn_vis.fit(X_vis, y)

    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_vis[:, 0], X_vis[:, 1],
                c=y, cmap=cmap_bold,
                edgecolor='k', s=20)

    plt.xlabel('Radius Mean (scaled)')
    plt.ylabel('Texture Mean (scaled)')
    plt.title('Decision Boundary (First Two Features)')
    plt.savefig("knn_decision_boundary.png")
    plt.show()


if __name__ == "__main__":
    run_knn_scenario()