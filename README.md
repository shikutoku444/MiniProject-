# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('robot_kinematics_dataset.csv')
df.head()

# Show correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

#defines the features (joint angles q1 to q6) and the target variable (end-effector positions x, y, z)
X = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']]
y = df[['x', 'y', 'z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

# Baseline model: Predict the average end-effector position
baseline_pred = [y_train.mean()] * len(y_test)

# Calculate evaluation metrics for baseline model
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

# Print baseline evaluation metrics
print(f'Baseline Mean Absolute Error (MAE): {baseline_mae}')
print(f'Baseline Mean Squared Error (MSE): {baseline_mse}')
print(f'Baseline R-squared (R²): {baseline_r2}')

# Plot predicted vs actual values for each target variable
plt.figure(figsize=(15, 5))
for i, col in enumerate(y.columns):
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test[col], y_pred[:, i], alpha=0.7)
    plt.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Predicted vs Actual {col}")
plt.tight_layout()
plt.show()

# Convert y_test and y_pred to DataFrame for proper indexing
residuals = pd.DataFrame(y_test.values - y_pred, columns=['x', 'y', 'z'])

# Plot residuals for each target variable
plt.figure(figsize=(15, 5))
for i, col in enumerate(['x', 'y', 'z']):  # Columns for end-effector positions
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test[col], residuals[col], alpha=0.7)
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.xlabel(f"Actual {col}")
    plt.ylabel("Residuals")
    plt.title(f"Residuals for {col}")
plt.tight_layout()
plt.show()

# After training the model
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(feature_importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()
