

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
 
df = pd.read_csv("C:/Users/rohit/Downloads/airpollution.csv")
print("Dataset loaded successfully")
print(df.head())

#Clean pollutant_avg
df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
df["pollutant_avg"] = df["pollutant_avg"].fillna(df["pollutant_avg"].mean())

#Long -> Wide (one row per station)
pivot_df = df.pivot_table(
    index=["state", "city", "station"],
    columns="pollutant_id",
    values="pollutant_avg"
).reset_index()

pivot_df = pivot_df.fillna(pivot_df.mean(numeric_only=True))
print("\nWide-format data created")
print(pivot_df.head())

#Simple AQI calculation
pivot_df["AQI"] = (
    pivot_df["PM10"] * 0.4 +
    pivot_df["NO2"]  * 0.3 +
    pivot_df["SO2"]  * 0.2 +
    pivot_df["CO"]   * 0.1
)

print("\nAQI column created")
print(pivot_df[["state", "city", "station", "AQI"]].head())

#Basic statistics and plots
print("\nPM2.5 statistics:")
print(pivot_df["PM2.5"].describe())

print("\nAQI statistics:")
print(pivot_df["AQI"].describe())

print("\nTop 5 states by number of stations:")
print(pivot_df["state"].value_counts().head(5))

sns.set_style("whitegrid")

plt.figure()
sns.histplot(pivot_df["PM2.5"], kde=True)
plt.title("PM2.5 Distribution across Stations")
plt.xlabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(pivot_df["AQI"], kde=True)
plt.title("AQI Distribution across Stations")
plt.xlabel("AQI (simple weighted value)")
plt.tight_layout()
plt.show()

#Regression: predict AQI (Linear Regression)
X_reg = pivot_df[["PM10", "NO2", "SO2", "CO"]]
y_reg = pivot_df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_pred = reg_model.predict(X_test)

reg_mae = mean_absolute_error(y_test, reg_pred)
print("\nLinear Regression MAE (AQI):", reg_mae)

#Linear Regression accuracy (AQI classes)
y_test_reg_class = pd.cut(
    y_test,
    bins=[-1, 50, 100, np.inf],
    labels=[0, 1, 2]
)

reg_pred_class = pd.cut(
    reg_pred,
    bins=[-1, 50, 100, np.inf],
    labels=[0, 1, 2]
)

reg_acc = accuracy_score(y_test_reg_class, reg_pred_class)
print("Linear Regression Accuracy (AQI class):", reg_acc)

#Classification: AQI categories (Logistic + KNN)
pivot_df["AQI_Category"] = pd.cut(
    pivot_df["AQI"],
    bins=[-1, 50, 100, np.inf],
    labels=[0, 1, 2]   # 0 = low, 1 = moderate, 2 = high
)

X_cls = pivot_df[["PM10", "NO2", "SO2", "CO"]]
y_cls = pivot_df["AQI_Category"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

print("\nFirst 10 AQI classes:", y_cls.head(10))

#Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_c, y_train_c)

log_pred = log_model.predict(X_test_c)
log_acc = accuracy_score(y_test_c, log_pred)
print("Logistic Regression Accuracy:", log_acc)

#KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_c, y_train_c)

knn_pred = knn_model.predict(X_test_c)
knn_acc = accuracy_score(y_test_c, knn_pred)
print("KNN Accuracy:", knn_acc)

#Clustering: K-Means
X_cluster = pivot_df[["PM10", "NO2", "SO2", "CO"]]

kmeans = KMeans(n_clusters=3, random_state=42)
pivot_df["Cluster"] = kmeans.fit_predict(X_cluster)

plt.figure()
plt.scatter(pivot_df["PM10"], pivot_df["AQI"],
            c=pivot_df["Cluster"], cmap="viridis")
plt.xlabel("PM10")
plt.ylabel("AQI")
plt.title("K-Means Clusters of Stations by Pollution")
plt.tight_layout()
plt.show()

#Visual comparison of model accuracies
model_names = ["LinearReg", "Logistic", "KNN"]
model_accs  = [reg_acc,       log_acc,    knn_acc]

plt.figure()
plt.bar(model_names, model_accs, color=["skyblue", "orange", "green"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("AQI Class Accuracy for Different Models")
plt.tight_layout()
plt.show()

# 10. Final summary print
print("\nMODEL COMPARISON SUMMARY")
print("Linear Regression Accuracy (AQI class) :", reg_acc)
print("Logistic Regression Accuracy           :", log_acc)
print("KNN Accuracy                           :", knn_acc)

