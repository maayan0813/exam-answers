import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_classification

# Simulated dataset
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_clusters_per_class=1, random_state=42)

# Split the dataset
X_labeled, X_unlabeled_test, y_labeled, y_unlabeled_test = train_test_split(
    X, y, test_size=0.8, stratify=y, random_state=42
)

# Ensure test set has exactly 500 samples, with remaining **800** as unlabeled
X_unlabeled, X_test, y_unlabeled, y_test = train_test_split(
    X_unlabeled_test, y_unlabeled_test, test_size=500/800, stratify=y_unlabeled_test, random_state=42
)

# Apply K-Means Clustering on 800 Unlabeled Data Points
num_clusters = 10  
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_unlabeled)

# Assign Pseudo-Labels Based on Nearest Labeled Sample
pseudo_labels = np.zeros(len(X_unlabeled))
for cluster_id in range(num_clusters):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        continue  

    # Find the nearest labeled sample to the cluster center
    nearest_labeled_index = np.argmin(np.linalg.norm(X_labeled - kmeans.cluster_centers_[cluster_id], axis=1))
    
    # Assign pseudo-labels based on the nearest labeled sample
    pseudo_labels[cluster_indices] = y_labeled[nearest_labeled_index]

# STrain a Random Forest on Labeled + Pseudo-Labeled Data
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.hstack((y_labeled, pseudo_labels))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, X_combined, y_combined, cv=5, scoring="roc_auc")
print(f"Cross-Validation AUC: {np.mean(cv_scores):.4f}")

# Train Final Model and Evaluate on Test Set
rf_model.fit(X_combined, y_combined)
y_pred = rf_model.predict(X_test)
final_auc = roc_auc_score(y_test, y_pred)
final_acc = accuracy_score(y_test, y_pred)

print(f"Final Model AUC on Test Set: {final_auc:.4f}")
print(f"Final Model Accuracy on Test Set: {final_acc:.4f}")

