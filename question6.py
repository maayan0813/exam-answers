import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with all features (50 iterations)

rf_model = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Compute initial error rates
rf_error_initial = 1 - accuracy_score(y_test, rf_model.predict(X_test))
gb_error_initial = 1 - accuracy_score(y_test, gb_model.predict(X_test))

# Get feature importance values
rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_

# Find the most important feature
rf_top_feature_idx = np.argmax(rf_importances)
gb_top_feature_idx = np.argmax(gb_importances)

rf_top_feature = feature_names[rf_top_feature_idx]
gb_top_feature = feature_names[gb_top_feature_idx]

# Display most important features
print("\n Most Important Feature:")
print(f"   - Random Forest: {rf_top_feature}")
print(f"   - Gradient Boosting: {gb_top_feature}")


# Train models without the most important feature (50 more iterations)

# Remove the most important feature
X_train_mod = np.delete(X_train, rf_top_feature_idx, axis=1)
X_test_mod = np.delete(X_test, rf_top_feature_idx, axis=1)

# Train the models again without the removed feature
rf_model_mod = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)
gb_model_mod = GradientBoostingClassifier(n_estimators=50, random_state=42)

rf_model_mod.fit(X_train_mod, y_train)
gb_model_mod.fit(X_train_mod, y_train)

# Compute error rates after removing the feature
rf_error_mod = 1 - accuracy_score(y_test, rf_model_mod.predict(X_test_mod))
gb_error_mod = 1 - accuracy_score(y_test, gb_model_mod.predict(X_test_mod))


# Compare results before and after feature removal

print("\n Comparative Results (Error rate before features removal):")
print(f"   - Random Forest: {rf_error_initial:.4f}")
print(f"   - Gradient Boosting: {gb_error_initial:.4f}")

print("\n Comparative Results (Error rate after feature removal): ")
print(f"   - Random Forest: {rf_error_mod:.4f}")
print(f"   - Gradient Boosting: {gb_error_mod:.4f}")


# Visualizing feature importance changes

plt.figure(figsize=(12, 5))

# Feature importance in Random Forest before removal
plt.subplot(1, 2, 1)
plt.barh(feature_names, rf_importances, color="blue", alpha=0.6, label="Before")
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()

# Feature importance in Gradient Boosting before removal
plt.subplot(1, 2, 2)
plt.barh(feature_names, gb_importances, color="green", alpha=0.6, label="Before")
plt.xlabel("Importance")
plt.title("Feature Importance - Gradient Boosting")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()








