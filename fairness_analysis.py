import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import cm

# Load the dataset, ensure the path is correct
data = pd.read_csv('D:/data.csv')

# Fill missing numeric values with the mean
data['numeric_column'].fillna(data['numeric_column'].mean(), inplace=True)

# Fill missing categorical values with the mode
data['categorical_column'].fillna(data['categorical_column'].mode()[0], inplace=True)

# Handle infinite values by replacing them with NaN
data = data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)

# Separate the dependent and independent variables
X = data.drop(columns=['fairness'])
y = data['fairness']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)

# Train the GradientBoostingRegressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=53)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Calculate the feature importance ranking
feature_importance = np.abs(shap_values).mean(0)
sorted_indices = np.argsort(feature_importance)[::-1]

# Extract the top twenty features and their corresponding importances
top_twenty_features = X.columns[sorted_indices[:20]]
top_twenty_importance = feature_importance[sorted_indices[:20]]

# Use the viridis colormap
cmap = plt.get_cmap('viridis')

# Select colors based on feature importance, ensuring the 'other' variable is the darkest
sorted_importance_indices = np.argsort(top_twenty_importance)
colors = [cmap(val) for val in np.linspace(1, 0.2, len(top_twenty_importance))]
colors = np.array(colors)[sorted_importance_indices]

# Create a figure with one subplot
fig, ax1 = plt.subplots(figsize=(8, 6))

# Set font configuration
plt.rcParams['font.sans-serif'] = ['Arial']  # English font
plt.rcParams['axes.unicode_minus'] = False  # Handle minus sign

# Plot the global feature importance bar chart
bars = ax1.barh(top_twenty_features[::-1], top_twenty_importance[::-1], color=colors[::-1])
ax1.set_xlabel('SHAP Values (mean)')
ax1.set_ylabel('Features')

# Adjust the left subplot borders
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Set consistent font size for the axes
ax1.tick_params(axis='both', which='major', labelsize=10)

# Add common variable name
fig.text(0.5, 0.04, 'Features', ha='center', fontsize=12)

# Adjust the layout
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Show the first plot
plt.show()

# Create a new plot for the SHAP summary
shap.summary_plot(shap_values, X, plot_size=(8, 6), max_display=20, color_bar=True, alpha=0.2)
